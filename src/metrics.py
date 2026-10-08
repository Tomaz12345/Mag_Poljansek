# author: jonathan prexl

import logging
from typing import Any, cast

import numpy as np
from sklearn.metrics import confusion_matrix

from .constants import NORM_CONFIG, SPECIAL_VALUES, VIZ_CONFIG
from .utils import log_and_raise

logger = logging.getLogger(__name__)


class ConfMatrix:
    def __init__(self, num_classes: int) -> None:
        if num_classes <= 0:
            msg = f"num_classes must be a positive integer, got {num_classes}"
            log_and_raise(ValueError, msg, logger_instance=logger)

        self.num_classes: int = num_classes
        self.state: np.ndarray = np.zeros(
            (self.num_classes, self.num_classes),
            dtype=np.float64,
        )

    def calc(self, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """calcs and returns the CM without saving it to state"""
        cm = confusion_matrix(
            gt.flatten(),
            pred.flatten(),
            labels=np.arange(self.num_classes),
        )
        return cast(np.ndarray, cm)

    def get_existing_classes(self) -> int:
        return int(sum(np.sum(self.state, axis=1) > 0))

    def add(self, gt: np.ndarray, pred: np.ndarray) -> None:
        """adds one label mask to the confusion matrix"""

        if gt.shape != pred.shape:
            msg = f"Ground truth and prediction shapes must match. Got gt.shape={gt.shape}, pred.shape={pred.shape}"
            log_and_raise(ValueError, msg, logger_instance=logger)
        if gt.shape != VIZ_CONFIG["EXPECTED_TILE_SIZE"]:
            msg = (
                f"Expected arrays with shape {VIZ_CONFIG['EXPECTED_TILE_SIZE']}, got shape {gt.shape}. "
                "If using different tile sizes, consider updating the validation."
            )
            log_and_raise(ValueError, msg, logger_instance=logger)

        gt = gt.flatten()
        pred = pred.flatten()
        pred = pred[gt != SPECIAL_VALUES["IGNORE_LABEL"]]
        gt = gt[gt != SPECIAL_VALUES["IGNORE_LABEL"]]

        if gt.size != 0:
            cm = confusion_matrix(gt, pred, labels=np.arange(self.num_classes))
            self.state += cast(np.ndarray, cm.astype(np.float64, copy=False))

    def add_batch(self, gt: Any, pred: Any) -> None:
        """adds a batch of label masks to the confusion matrix (vectorized)"""

        # convert pytorch tensors to numpy arrays
        if not isinstance(gt, np.ndarray):
            gt = gt.cpu().numpy()
            pred = pred.cpu().numpy()

        if len(gt.shape) != 3:
            msg = f"Expected batch input with 3 dimensions (batch, height, width), got {len(gt.shape)} dimensions with shape {gt.shape}"
            log_and_raise(ValueError, msg, logger_instance=logger)

        # Vectorized batch processing: flatten entire batch at once
        # Shape: (batch, H, W) -> (batch * H * W,)
        gt_flat = gt.reshape(-1)
        pred_flat = pred.reshape(-1)

        # Remove ignore pixels from both
        valid_mask = gt_flat != SPECIAL_VALUES["IGNORE_LABEL"]
        gt_valid = gt_flat[valid_mask]
        pred_valid = pred_flat[valid_mask]

        if gt_valid.size > 0:
            cm = confusion_matrix(gt_valid, pred_valid, labels=np.arange(self.num_classes))
            self.state += cast(np.ndarray, cm.astype(np.float64, copy=False))

        return None

    def norm_on_lines(self) -> np.ndarray:
        """norms along the lines of the matrix"""

        a = self.state
        b = np.sum(self.state, axis=1)[:, None]
        return cast(np.ndarray, np.divide(a, b, out=np.zeros_like(a), where=b != 0))

    def get_aa(self) -> float:
        confmatrix = self.norm_on_lines()
        existing = self.get_existing_classes()
        if existing == 0:
            return 0.0
        return float(np.diagonal(confmatrix).sum() / existing)

    def get_IoU(self) -> np.ndarray:
        """Calculate per-class IoU (vectorized)."""
        cm = self.state
        # Diagonal: true positives per class
        tp = np.diag(cm)
        # Row sums: actual class totals (TP + FN)
        row_sums = cm.sum(axis=1)
        # Column sums: predicted class totals (TP + FP)
        col_sums = cm.sum(axis=0)
        # Union = TP + FN + FP = row_sum + col_sum - TP
        union = row_sums + col_sums - tp
        # IoU = TP / Union, with safe division
        res = np.divide(tp, union, out=np.zeros(self.num_classes, dtype=np.float64), where=union != 0)
        return cast(np.ndarray, res)

    def get_mIoU(self) -> float:
        return float(np.mean(self.get_IoU()))

    def get_precision(self) -> np.ndarray:
        """Calculate per-class precision: TP / (TP + FP) (vectorized)"""
        cm = self.state
        tp = np.diag(cm)
        # Column sums = TP + FP (all predictions for each class)
        col_sums = cm.sum(axis=0)
        res = np.divide(tp, col_sums, out=np.zeros(self.num_classes, dtype=np.float64), where=col_sums != 0)
        return cast(np.ndarray, res)

    def get_recall(self) -> np.ndarray:
        """Calculate per-class recall: TP / (TP + FN) (vectorized)"""
        cm = self.state
        tp = np.diag(cm)
        # Row sums = TP + FN (all actual instances for each class)
        row_sums = cm.sum(axis=1)
        res = np.divide(tp, row_sums, out=np.zeros(self.num_classes, dtype=np.float64), where=row_sums != 0)
        return cast(np.ndarray, res)

    def get_f1(self) -> np.ndarray:
        """Calculate per-class F1 score: 2 * (precision * recall) / (precision + recall)"""
        precision = self.get_precision()
        recall = self.get_recall()
        f1 = 2 * (precision * recall) / (precision + recall + NORM_CONFIG["EPSILON"])
        return cast(np.ndarray, f1)

    def get_mean_f1(self) -> float:
        """Calculate mean F1 score across all classes"""
        return float(np.mean(self.get_f1()))

    def get_accuracy(self) -> float:
        """Calculate overall accuracy: sum of diagonal / sum of all elements"""
        total = np.sum(self.state)
        correct = np.sum(np.diagonal(self.state))
        return float(np.divide(correct, total, out=np.zeros_like(correct), where=total != 0))

    def get_per_class_accuracy(self) -> np.ndarray:
        """Calculate per-class accuracy (same as recall)"""
        return self.get_recall()

    def get_per_class_iou(self) -> np.ndarray:
        """Get per-class IoU scores."""
        return self.get_IoU()

    def get_per_class_f1(self) -> np.ndarray:
        """Get per-class F1 scores."""
        return self.get_f1()

    @property
    def n_classes(self) -> int:
        return self.num_classes

    @property
    def overall_conf_mat(self) -> np.ndarray:
        """The confusion matrix state as numpy array."""
        return self.state

    def to_dict(self) -> dict[str, float | np.ndarray]:
        """Export all metrics as a dictionary"""
        return {
            "AA": self.get_aa(),
            "mIoU": self.get_mIoU(),
            "IoU": self.get_IoU(),
            "accuracy": self.get_accuracy(),
            "mean_f1": self.get_mean_f1(),
            "f1": self.get_f1(),
            "precision": self.get_precision(),
            "recall": self.get_recall(),
            "per_class_accuracy": self.get_per_class_accuracy(),
        }
