#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions to verify and log an experiment's hyperparameters.
Outputs class accumulates results throughout an epoch and saves it.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import csv
import json
from pytorch_lightning.loggers import CometLogger
from torch import Tensor

from icr.dataloader import CodrawData


def log_all(logger: CometLogger, params: argparse.Namespace,
            datasets: Dict[str, CodrawData]) -> None:
    """Log CLI arguments to the logger and to a local folder."""
    # create directory where experiment results and metadata will be saved
    path_name = Path(f'{params.path_to_outputs}{logger.version}')
    os.mkdir(path_name)
    # log all to comet
    params_dic = {k: v for k, v in params.__dict__.items() if k != 'comet_key'}
    logger.experiment.log_parameters(params_dic)
    logger.experiment.log_code('icr/dataloader.py')
    logger.experiment.log_code('icr/logs.py')
    logger.experiment.log_code('main.py')
    logger.experiment.log_code('icr/metrics.py')
    logger.experiment.log_code('icr/model.py')
    logger.experiment.log_code('icr/plmodel.py')
    logger.experiment.log_code(folder='icr/structs/')
    logger.experiment.log_others(datasets['train'].sizes)
    logger.experiment.log_others(datasets['val'].sizes)
    logger.experiment.log_others(datasets['test'].sizes)
    # also log hyperparameter configuration to local
    out_path = Path(f'{params.path_to_outputs}{logger.version}/config.json')
    with open(out_path, 'w') as file:
        json.dump(params_dic, file)


def filter_params(config: argparse.Namespace) -> argparse.Namespace:
    """Remove or verify hyperparameters according to architecture."""
    img_models = ('resnet101', 'resnet101_resized_centered', 'vgg16')
    if config.img_pretrained in img_models:
        assert config.img_input_dim == 2048
    text_models = ('all-mpnet-base-v2',)
    if config.text_pretrained in text_models:
        assert config.context_input_dim == 768
    if config.lr_scheduler == 'exp':
        config.lr_step = None
    if config.delay_scenes:
        assert config.task == 'drawer'
    # at least one source of input is necessary
    assert not config.no_image or not config.no_context or not config.no_msg
    return config


class Outputs:
    """Accumulate and save a model's outputs in an epoch."""
    def __init__(self, split: str, datapoints: Dict[int, Tuple[int, int]]):
        self.split = split
        self.datapoints = datapoints
        self.predictions: Dict[int, int] = {}
        self.labels: Dict[int, int] = {}
        self.probs: Dict[int, int] = {}
        self.counter: int = 0

    def update(self, batch_idxs: Tensor, batch_pred: Tensor,
               batch_label: Tensor, batch_prob: Tensor) -> None:
        """Accumulate outputs for a batch."""
        for idx, pred, label, prob in zip(batch_idxs, batch_pred,
                                          batch_label, batch_prob):
            self.counter += 1
            self.predictions[idx.item()] = pred.item()
            self.labels[idx.item()] = label.item()
            self.probs[idx.item()] = prob.item()

    def reset(self) -> None:
        """Restart accumulation."""
        self.predictions = {}
        self.labels = {}
        self.probs = {}
        self.counter = 0

    def save(self, epoch: int, logger: CometLogger,
             out_path: str) -> None:
        """Log outputs into a local csv file and send to comet."""
        # sanity checking that no additional data was added by mistake
        # at the end of the epoch, the number of instances summing over batches
        # should match the number of datapoints in this split
        # besides, the indexes are also checked in the loop below
        assert len(self.predictions) == len(self.datapoints)
        assert len(self.labels) == len(self.datapoints)
        assert len(self.probs) == len(self.datapoints)
        assert self.counter == len(self.datapoints)
        fname = Path(f'{out_path}{logger.version}/{self.split}_{epoch}.csv')
        with open(fname, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            header = ['idx', 'game_id', 'turn', 'label', 'pred', 'prob']
            writer.writerow(header)
            for idx, (game_id, turn) in self.datapoints.items():
                writer.writerow([idx, game_id, turn,
                                 self.labels[idx], self.predictions[idx],
                                 self.probs[idx]])
        logger.experiment.log_table(fname)
