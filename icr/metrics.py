#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class to manage metrics for the Pytorch Lightining module.
"""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn import metrics
from torchmetrics import MetricCollection, Accuracy,  F1Score
from torchmetrics.classification import (
    BinaryAccuracy, BinaryF1Score, BinaryCohenKappa, BinaryROC,
    BinaryPrecisionRecallCurve, BinaryPrecision, BinaryRecall,
    BinaryAveragePrecision, BinaryConfusionMatrix)

from icr.config import LABEL_MAP
from icr.logs import Outputs

NON_SCALAR = ['BinaryROC', 'BinaryConfusionMatrix',
              'BinaryPrecisionRecallCurve']


class LitMetrics(torch.nn.Module):
    """Contain all the metrics to be tracked in an experiment."""
    def __init__(self, split: str, threshold: float, n_labels: int,
                 datapoints: Dict[int, Tuple[int, int]], prop_icr: float):
        super().__init__()
        self.split = split
        self.threshold = threshold
        self.n_labels = n_labels
        self.n_datapoints = len(datapoints)
        self.prop_icr = prop_icr
        self.build(datapoints)

    def forward(self):
        """This method is not necessary."""
        # This class has to inherit from torch.nn.Module just so
        # PyTorch lightning recognises the metrics correctly.
        pass

    def create_collections(self) -> Tuple[MetricCollection, MetricCollection]:
        """Initialise PytorchLightning collections of metrics."""
        # Single-value binary metrics
        collection = MetricCollection([
            BinaryAccuracy(threshold=self.threshold),
            BinaryPrecision(threshold=self.threshold),
            BinaryRecall(threshold=self.threshold),
            BinaryAveragePrecision(),
            BinaryF1Score(threshold=self.threshold),
            BinaryCohenKappa(threshold=self.threshold),
            BinaryROC(),
            BinaryConfusionMatrix(threshold=self.threshold),
            BinaryPrecisionRecallCurve()
        ])
        # Multi-class metrics
        collection_multiclass = MetricCollection([
            Accuracy(num_classes=self.n_labels, average='none'),
            F1Score(num_classes=self.n_labels, average='macro',
                    threshold=self.threshold),
        ])
        return collection, collection_multiclass

    def build(self, dtpnts: Dict[int, Tuple[int, int]]) -> None:
        """Initialize all metrics."""
        collection, col_multiclass = self.create_collections()
        self.metrics = collection.clone(prefix=f'{self.split}_')
        self.multiclass = col_multiclass.clone(prefix=f'{self.split}_')
        self.outputs = None
        if self.split in ('val', 'test'):
            self.outputs = Outputs(self.split, dtpnts)

    def update(self, idxs, probs, preds, labels) -> None:
        """Add predictions from one batch."""
        self.metrics.update(probs, labels)
        self.multiclass.update(preds, labels)
        if self.outputs:
            self.outputs.update(idxs, preds, labels, probs)

    def compute(self):
        """Compute all the metrics being tracked."""
        # Multiclass metrics
        results_multiclass = self.multiclass.compute()
        f1_macro = results_multiclass[f'{self.split}_F1Score']
        acc_class = results_multiclass[f'{self.split}_Accuracy']
        acc_icr, acc_not_icr = self._split_metrics(acc_class)
        # Binary metrics
        results = self.metrics.compute()
        filtered = {k: v for k, v in results.items()
                    if k.split('_')[1] not in NON_SCALAR}

        if self.split == 'train':
            return f1_macro, acc_icr, acc_not_icr, filtered, None

        # ROC curve and area under the curve
        fpr, tpr, _ = results[f'{self.split}_BinaryROC']
        auc, fig_roc = self._plot_roc(fpr, tpr)
        # confusion matrix
        conf_matrix = results[f'{self.split}_BinaryConfusionMatrix']
        conf_matrix = conf_matrix.cpu().numpy()
        # precision-recall curve and area under the curve
        prec, recall, _ = results[f'{self.split}_BinaryPrecisionRecallCurve']
        avp, auc_pr, fig_pr = self._plot_prcurve(
            prec, recall, self.metrics['BinaryPrecisionRecallCurve'])
        return (f1_macro, acc_icr, acc_not_icr, filtered, (auc, fig_roc,
                conf_matrix, auc_pr, fig_pr))

    def reset(self) -> None:
        """Clear all internal states."""
        self.metrics.reset()
        self.multiclass.reset()
        if self.outputs is not None:
            self.outputs.reset()
        plt.close('all')
        # double check that metrics got reset
        assert not self.metrics['BinaryAveragePrecision'].target
        assert self.multiclass['Accuracy'].tp.sum() == 0

    @staticmethod
    def _plot_roc(fpr, tpr):
        """Return AUC and ROC plot."""
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                          pos_label=LABEL_MAP['icr'])
        display.plot()
        display.ax_.plot([0, 1], [0, 1], 'r--')
        return roc_auc, display.figure_

    def _plot_prcurve(self, precision, recall, pr_obj):
        """Return AVP, AUC-PR and PR plot."""
        precision = precision.cpu().numpy()
        recall = recall.cpu().numpy()
        target = [x.cpu().tolist() for batch in pr_obj.target for x in batch]
        pred = [x.cpu().tolist() for batch in pr_obj.preds for x in batch]
        assert len(target) == self.n_datapoints
        assert len(pred) == self.n_datapoints
        avp = metrics.average_precision_score(target, pred)
        display = metrics.PrecisionRecallDisplay(precision, recall,
                                                 average_precision=avp)
        display.plot()
        auc = metrics.auc(recall, precision)
        rand_line = self.prop_icr / 100
        display.ax_.axhline(rand_line, color='r', ls='--')
        return avp, auc, display.figure_

    @staticmethod
    def _split_metrics(accs):
        """Extract metrics per class."""
        acc_cr = accs[LABEL_MAP['icr']]
        acc_not_cr = accs[LABEL_MAP['not_icr']]
        return acc_cr, acc_not_cr
