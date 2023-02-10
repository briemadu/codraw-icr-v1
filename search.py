#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An adapted copy of main.py which is used to perform hyperparameter search
using comet.ml.
"""

import warnings
from pathlib import Path

import comet_ml
from comet_ml import Optimizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from icr.config import args, LABEL_MAP
from icr.dataloader import CodrawData
from icr.logs import log_all, filter_params
from icr.plmodel import LitClassifier

# ignore Lightning warning about increasing the number of workers
warnings.filterwarnings("ignore", ".*does not have many workers.*")

params = filter_params(args())

######
import os
os.mkdir('outputs_search')

params.comet_project = 'cr-codraw-optimizer-search'
params.path_to_outputs = './outputs_search/'

optim_config = {
    "algorithm": "bayes",
    "parameters": {
        "random_seed": {"type": "integer", "min": 1, "max": 54321},
        "dim": {"type": "discrete", "values": [32, 64, 128, 256, 512, 1024]},
        "hidden_dim": {"type": "discrete", "values": [32, 64, 128, 256, 512, 1024]},
        "batch_size": {"type": "discrete", "values": [32, 64, 128, 256, 512, 1024]},
        "lr": {"type": "discrete", "values": [0.1, 0.01, 0.001, 0.0001, 0.003, 0.0003, 0.00001, 0.0005]},
        "dropout": {"type": "discrete", "values": [0.1, 0.2, 0.3, 0.5]},
        "clip": {"type": "discrete", "values": [0, 0.25, 0.5, 1, 2.5, 5, 10]},
        "weight_cr": {"type": "float", "scaling_type": "uniform", "min": 1, "max": 10},
        "accumulate_grad": {"type": "discrete", "values": [1, 2, 5, 10, 25]},
        "gamma": {"type": "discrete", "values": [0.1, 0.5, 0.9, 0.99, 1]},
        "lr_scheduler": {"type": "categorical", "values": ['none', 'exp', 'step']},
        "weight_decay": {"type": "discrete", "values": [1, 0.1, 0.01, 0.001, 0.0001]},
        "lr_step": {"type": "integer", "min": 1, "max": 5},
    },
    "spec": {
        "metric": "val_BinaryAveragePrecision",
        "objective": "maximize",
    },
}
opt = Optimizer(optim_config)

for experiment in opt.get_experiments(
        api_key=params.comet_key,
        workspace=params.comet_workspace,
        project_name=params.comet_project,
        auto_output_logging="simple",
        log_git_metadata=False,
        log_git_patch=False):

    # Extract optim experiment config
    params.random_seed = experiment.get_parameter('random_seed')
    params.batch_size = experiment.get_parameter('batch_size')
    params.img_embedding_dim = experiment.get_parameter('dim')
    params.context_embedding_dim = experiment.get_parameter('dim')
    params.last_msg_embedding_dim = experiment.get_parameter('dim')
    params.img_embedding_dim = experiment.get_parameter('dim')
    params.hidden_dim = experiment.get_parameter('hidden_dim')
    params.lr = experiment.get_parameter('lr')
    params.dropout = experiment.get_parameter('dropout')
    params.clip = experiment.get_parameter('clip')
    params.weight_cr = experiment.get_parameter('weight_cr')
    params.accumulate_grad = experiment.get_parameter('accumulate_grad')
    params.gamma = experiment.get_parameter('gamma')
    params.lr_scheduler = experiment.get_parameter('lr_scheduler')
    params.weight_decay = experiment.get_parameter('weight_decay')
    params.lr_step = experiment.get_parameter('lr_step')
######

    pl.seed_everything(params.random_seed)

    SPLITS = ('train', 'val', 'test')
    datasets = {split: CodrawData(split, params) for split in SPLITS}
    params.n_labels = len(LABEL_MAP)

    # because the dataloader can call random, we reset it here just so the models
    # are initialised in a similar state
    pl.seed_everything(params.random_seed)
    model = LitClassifier(datasets=datasets, config=params)

    logger = CometLogger(
        api_key=params.comet_key,
        workspace=params.comet_workspace,
        save_dir="comet-logs/",
        project_name=params.comet_project,
        disabled=params.ignore_comet,
        experiment_key=experiment.get_key(),
        auto_metric_logging=False
    )
    log_all(logger, params, datasets)

    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(params.path_to_checkpoints) / f'model_{logger.version}',
        filename='model-{epoch}-{val_BinaryAveragePrecision:.5f}',
        monitor='val_BinaryAveragePrecision',
        mode='max',
        save_top_k=1,
        )

    trainer = pl.Trainer(
        accelerator=params.device,
        devices=[1],
        max_epochs=params.n_epochs,
        logger=logger,
        gradient_clip_val=params.clip if params.clip > 0 else None,
        accumulate_grad_batches=params.accumulate_grad,
        callbacks=[lr_monitor, checkpoint_callback],
        deterministic=True,
        )

    trainer.fit(model=model)
    #trainer.test(model=model, ckpt_path=checkpoint_callback.best_model_path)
