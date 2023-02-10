#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytorch Lightining module to structure and manage the experiment.
"""

import argparse
import re
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from icr.config import LABEL_MAP
from icr.dataloader import CodrawData
from icr.metrics import LitMetrics
from icr.model import Classifier

REDUCTION = 'sum'


class LitClassifier(pl.LightningModule):
    """Lightining experiment."""
    def __init__(self, datasets: Dict[str, CodrawData],
                 config: argparse.Namespace):
        super().__init__()
        self.datasets = datasets
        self.config = config
        self.model = Classifier(config)
        self.labels_dic = LABEL_MAP
        self._define_metrics()
        self._define_loss()

    def _define_metrics(self) -> None:
        prop_icr_train = self.datasets['train'].sizes['train: %_icr']
        prop_icr_val = self.datasets['val'].sizes['val: %_icr']
        prop_icr_test = self.datasets['test'].sizes['test: %_icr']

        threshold = self.config.decision_threshold
        n_labels = self.config.n_labels

        self.train_metrics = LitMetrics(
            split='train', threshold=threshold, n_labels=n_labels,
            datapoints=self.datasets['train'].datapoints,
            prop_icr=prop_icr_train)

        self.val_metrics = LitMetrics(
            split='val', threshold=threshold, n_labels=n_labels,
            datapoints=self.datasets['val'].datapoints,
            prop_icr=prop_icr_val)

        self.test_metrics = LitMetrics(
            split='test', threshold=threshold, n_labels=n_labels,
            datapoints=self.datasets['test'].datapoints,
            prop_icr=prop_icr_test)

    def _define_loss(self) -> None:
        """Define the cross entropy loss."""
        weight_cr = self.config.weight_cr
        if weight_cr == 0:
            n_positive = self.datasets['train'].sizes['train: n_cr']
            n_negative = self.datasets['train'].sizes['train: n_other']
            # set weight according to Pytorch documentation
            # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html?highlight=bce#torch.nn.BCEWithLogitsLoss
            weight_cr = n_negative / n_positive
        pos_class_weight = torch.tensor(weight_cr)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_class_weight,
                                              reduction=REDUCTION)
        print(f'\nUsing weight {weight_cr} for CR class.\n')

    def forward(self, context: Tensor, last_msg: Tensor,
                image: Tensor) -> Tensor:
        """Call the model's forward pass."""
        output = self.model(context=context, last_msg=last_msg, image=image)
        return output

    def _call_forward(self, batch):
        """Get the predictions in a batch."""
        # the loader returns a batch containing
        # idx, datap.context, datap.last_msg, datap.img, datap.label
        idxs, context, last_msg, image, label = batch
        logits = self(context=context, last_msg=last_msg, image=image)

        loss = self.criterion(logits.squeeze(1), label.float())
        probs_icr = torch.sigmoid(logits).view(-1)
        pred = (probs_icr >= self.config.decision_threshold).long()

        return idxs, probs_icr, pred, label, loss

    def training_step(self, batch, batch_idx):
        """Training step in one batch."""
        idxs, probs, preds, labels, loss = self._call_forward(batch)
        self.train_metrics.update(idxs, probs, preds, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step in one batch."""
        idxs, probs, preds, labels, loss = self._call_forward(batch)
        self.val_metrics.update(idxs, probs, preds, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step in one batch."""
        idxs, probs, preds, labels, loss = self._call_forward(batch)
        self.test_metrics.update(idxs, probs, preds, labels)
        self.log('test_loss', loss, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs) -> None:
        """Finalize training epoch, collect metrics."""
        # for training, we collect binary and multiclass metrics only
        self._log_metrics('train', self.train_metrics)
        self.train_metrics.reset()

    def validation_epoch_end(self, outputs) -> None:
        """Finalize validation, collect metrics."""
        if not self.trainer.sanity_checking:
            # sanity check that metric gets rebooted
            # assert sum([len(x) for x in self.val_roc.target]) == 7714
            self._log_metrics('val', self.val_metrics)
        self.val_metrics.reset()

    def test_epoch_end(self, outputs) -> None:
        """Finalize test, collect metrics."""
        self._log_metrics('test', self.test_metrics)
        self.test_metrics.reset()

    def _log_metrics(self, split: str, metrics: LitMetrics) -> None:
        """Log metrics for a given split."""
        macro_f1, acc_icr, acc_not_icr, results, other = metrics.compute()
        self.log(f'{split}_macro-f1_epoch', macro_f1)
        self.log(f'{split}_acc_cr_epoch', acc_icr)
        self.log(f'{split}_acc_not_cr_epoch', acc_not_icr)
        self.log_dict(results)
        # further metrics and images are only logged during inference
        if split in ('val', 'test'):
            # collect outputs
            metrics.outputs.save(self.current_epoch, self.logger,
                                 self.config.path_to_outputs)
            # collect figures and related metrics
            auc, fig_roc, conf_matrix, auc_pr, fig_pr = other
            self.log(f'{split}_auc_epoch', auc)
            self.logger.experiment.log_figure(
                    figure=fig_roc,
                    figure_name=f'roc_{split}_{self.current_epoch}')
            id2label = {value: key for key, value in LABEL_MAP.items()}
            self.logger.experiment.log_confusion_matrix(
                matrix=conf_matrix,
                labels=[id2label[0], id2label[1]],
                epoch=self.current_epoch,
                file_name=f'cf_{split}_{self.current_epoch}')
            self.log('val_auc_pr_epoch', auc_pr)
            self.logger.experiment.log_figure(
                figure=fig_pr,
                figure_name=f'precrec_{split}_{self.current_epoch}')

    def on_train_end(self):
        """Log the best model."""
        version = self.logger.version
        ckpts = Path(self.config.path_to_checkpoints)
        ckpt_dir = ckpts / f'model_{version}'
        self.logger.experiment.log_asset_folder(ckpt_dir)
        best_epoch = self._get_best_epoch()
        self.logger.experiment.log_metric('best_epoch', best_epoch)

        fname = Path(f'{self.config.path_to_outputs}{version}/best-epoch.txt')
        with open(fname, 'w') as file:
            file.write(str(best_epoch))

    def configure_optimizers(self):
        """Define optimizer (and scheduler)."""
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            )
        if self.config.lr_scheduler != 'none':
            lr_scheduler = self._define_lr_scheduler(optimizer)
            return [optimizer], [lr_scheduler]
        return [optimizer]

    def _define_lr_scheduler(self, optimizer):
        """Define LR scheduler."""
        gamma = self.config.gamma
        if self.config.lr_scheduler == 'step':
            step = self.config.lr_step
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=step, gamma=gamma)
        elif self.config.lr_scheduler == 'exp':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma)
        return lr_scheduler

    def train_dataloader(self):
        """Define train data loader."""
        return DataLoader(self.datasets['train'],
                          batch_size=self.config.batch_size, shuffle=True)

    def val_dataloader(self):
        """Define val data loader."""
        return DataLoader(self.datasets['val'],
                          batch_size=self.config.batch_size, shuffle=False)

    def test_dataloader(self):
        """Define test data loader."""
        return DataLoader(self.datasets['test'],
                          batch_size=self.config.batch_size, shuffle=False)

    def _get_best_epoch(self):
        """Hack the best epoch."""
        path = self.trainer.checkpoint_callback.best_model_path
        begin = re.search('model-epoch=', path).span()[1]
        end = re.search('-val_BinaryAver', path).span()[0]
        return int(path[begin: end])
