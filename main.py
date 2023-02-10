#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run experiments for the Instruction Clarification Requests
EACL 2023 paper.

The model is build according to the CLI arguments, then trained and tested.

It uses Pytorch Lightning to structure the experiment and comet.ml as a logger. 
Outputs and checkpoints are also saved locally.
"""

import warnings
from pathlib import Path

import comet_ml
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
trainer.test(model=model, ckpt_path=checkpoint_callback.best_model_path)
