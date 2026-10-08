import argparse
import csv
import json
import logging
import os
from collections.abc import Iterable, Mapping
from os import PathLike
from typing import Any

import numpy as np

from .messages import RESULT_SAVER_ERRORS
from .utils import log_and_raise

logger = logging.getLogger(__name__)


def _coerce_float(value: Any) -> float:
    return float(value)


class MetricsLogger:
    def __init__(self, save_dir: str | PathLike[str], experiment_name: str) -> None:
        """Initialize metrics logger.

        Args:
            save_dir: Directory to save metrics files
            experiment_name: Name of the experiment

        Raises:
            ValueError: If save_dir or experiment_name are empty or invalid.
        """
        if not save_dir:
            msg = RESULT_SAVER_ERRORS.INVALID_SAVE_DIR.format(value=save_dir)
            log_and_raise(ValueError, msg, logger_instance=logger)
        if not experiment_name:
            msg = f"experiment_name must be a non-empty string, got {experiment_name!r}"
            log_and_raise(ValueError, msg, logger_instance=logger)

        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.metrics_dir = os.path.join(save_dir, experiment_name, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Storage for metrics over time
        self.train_metrics: list[dict[str, Any]] = []
        self.val_metrics: list[dict[str, Any]] = []

    def log_train_step(self, step: int, loss: float, epoch: int | None = None, **kwargs: Any) -> None:
        """Log training step metrics.

        Args:
            step: Training step number
            loss: Training loss value
            epoch: Optional epoch number
            **kwargs: Additional metrics to log
        """
        metrics: dict[str, Any] = {"step": step, "loss": loss}
        if epoch is not None:
            metrics["epoch"] = epoch
        metrics.update(kwargs)
        self.train_metrics.append(metrics)

    def log_val_step(
        self,
        step: int,
        metrics_dict: Mapping[str, Any] | Iterable[tuple[str, Any]],
    ) -> None:
        """Log validation step metrics.

        Args:
            step: Training step number
            metrics_dict: Dictionary of validation metrics
        """
        metrics_payload: dict[str, Any] = {"step": step}
        metrics_payload.update(dict(metrics_dict))
        self.val_metrics.append(metrics_payload)

    def save_to_csv(self) -> None:
        """Save all metrics to CSV files."""
        # Save training metrics
        if self.train_metrics:
            train_csv_path = os.path.join(self.metrics_dir, "train_metrics.csv")
            with open(train_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.train_metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.train_metrics)

        # Save validation metrics
        if self.val_metrics:
            val_csv_path = os.path.join(self.metrics_dir, "val_metrics.csv")
            with open(val_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.val_metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.val_metrics)

    def save_to_json(self) -> None:
        """Save all metrics to JSON files."""
        # Save training metrics
        if self.train_metrics:
            train_json_path = os.path.join(self.metrics_dir, "train_metrics.json")
            with open(train_json_path, "w") as f:
                json.dump(self.train_metrics, f, indent=2)

        # Save validation metrics
        if self.val_metrics:
            val_json_path = os.path.join(self.metrics_dir, "val_metrics.json")
            with open(val_json_path, "w") as f:
                json.dump(self.val_metrics, f, indent=2)

    def save_all(self) -> None:
        """Save metrics in all formats."""
        self.save_to_csv()
        self.save_to_json()


class ConfigSaver:
    @staticmethod
    def save_args_summary(args: argparse.Namespace, save_dir: str, experiment_name: str) -> None:
        """Save training arguments as JSON and TXT.

        Args:
            args: Training arguments namespace
            save_dir: Directory to save config files
            experiment_name: Name of the experiment
        """
        config_dir = os.path.join(save_dir, experiment_name, "config")
        os.makedirs(config_dir, exist_ok=True)

        from src.logging_utils import filter_disabled_children

        args_dict = filter_disabled_children(args)

        # Save as JSON
        json_path = os.path.join(config_dir, "config.json")
        with open(json_path, "w") as f:
            json.dump(args_dict, f, indent=2, default=str)

        # Save as human-readable TXT
        txt_path = os.path.join(config_dir, "config.txt")
        with open(txt_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Training Configuration: {experiment_name}\n")
            f.write("=" * 80 + "\n\n")

            for key, value in sorted(args_dict.items()):
                f.write(f"{key:.<30} {value}\n")


class PredictionSaver:
    def __init__(self, save_dir: str) -> None:
        """Initialize prediction saver.

        Raises:
            ValueError: If save_dir is empty.
        """
        if not save_dir:
            msg = RESULT_SAVER_ERRORS.INVALID_SAVE_DIR.format(value=save_dir)
            log_and_raise(ValueError, msg, logger_instance=logger)

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Storage for per-element metrics
        self.element_metrics: list[dict[str, Any]] = []

    def log_element(self, element_id: str, metrics: dict[str, float], save_prediction: bool = True) -> None:
        """Log metrics for a single element.

        Args:
            element_id: Unique identifier for the element
            metrics: Dictionary of metrics for this element
            save_prediction: Whether prediction was saved for this element
        """
        element_data: dict[str, Any] = {
            "element_id": element_id,
            "saved": save_prediction,
            **metrics,
        }
        self.element_metrics.append(element_data)

    def save_confusion_matrix(self, confusion_matrix: np.ndarray, filename: str = "confusion_matrix.csv") -> None:
        """Save confusion matrix as CSV.

        Args:
            confusion_matrix: Confusion matrix array (must be 2D and square)
            filename: Output filename

        Raises:
            ValueError: If confusion_matrix is not 2D or not square.
        """
        if confusion_matrix.ndim != 2:
            msg = f"Confusion matrix must be a 2D array, got {confusion_matrix.ndim}D array with shape {confusion_matrix.shape}"
            log_and_raise(ValueError, msg, logger_instance=logger)
        if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
            msg = f"Confusion matrix must be square, got shape {confusion_matrix.shape}"
            log_and_raise(ValueError, msg, logger_instance=logger)

        csv_path = os.path.join(self.save_dir, filename)
        np.savetxt(csv_path, confusion_matrix, delimiter=",", fmt="%.6f")

    def save_element_metrics(self, filename: str = "element_metrics.csv") -> None:
        """Save per-element metrics to CSV."""
        if not self.element_metrics:
            return

        csv_path = os.path.join(self.save_dir, filename)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.element_metrics[0].keys())
            writer.writeheader()
            writer.writerows(self.element_metrics)

    def save_summary_metrics(self, metrics: dict[str, Any], filename: str = "summary_metrics.json") -> None:
        # Convert numpy arrays to lists for JSON serialization
        json_metrics: dict[str, float | list[float] | str | bool | None] = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                json_metrics[key] = [_coerce_float(v) for v in value.tolist()]
            elif isinstance(value, (np.floating, np.integer)):
                json_metrics[key] = _coerce_float(value)
            elif isinstance(value, list):
                json_metrics[key] = [_coerce_float(v) for v in value]
            elif isinstance(value, (int, float)):
                json_metrics[key] = _coerce_float(value)
            else:
                json_metrics[key] = str(value)

        json_path = os.path.join(self.save_dir, filename)
        with open(json_path, "w") as f:
            json.dump(json_metrics, f, indent=2)

        # Also save as TXT for easy viewing
        txt_path = os.path.join(self.save_dir, "summary_metrics.txt")
        with open(txt_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("Prediction Summary Metrics\n")
            f.write("=" * 80 + "\n\n")

            for key in sorted(json_metrics):
                value = json_metrics[key]
                if isinstance(value, list):
                    values = [_coerce_float(v) for v in value]
                    f.write(f"{key}:\n")
                    for i, v in enumerate(values):
                        f.write(f"  Class {i}: {v:.6f}\n")
                elif isinstance(value, (int, float)):
                    f.write(f"{key:.<30} {_coerce_float(value):.6f}\n")
                else:
                    f.write(f"{key:.<30} {value}\n")


def select_random_subset(
    total_elements: int,
    subset_ratio: float = 0.1,
    seed: int = 42,
) -> set[int]:
    """Select a random subset of element indices.

    Args:
        total_elements: Total number of elements (must be positive)
        subset_ratio: Fraction of elements to select (must be in (0.0, 1.0])
        seed: Random seed for reproducibility

    Returns:
        Set of selected indices

    Raises:
        ValueError: If total_elements <= 0 or subset_ratio not in (0.0, 1.0].
    """
    if total_elements <= 0:
        msg = f"total_elements must be a positive integer, got {total_elements}"
        log_and_raise(ValueError, msg, logger_instance=logger)
    if subset_ratio <= 0.0 or subset_ratio > 1.0:
        msg = (
            f"subset_ratio must be in range (0.0, 1.0], got {subset_ratio}. Use 1.0 to include all elements, or a smaller value for random sampling."
        )
        log_and_raise(ValueError, msg, logger_instance=logger)

    rng = np.random.RandomState(seed)
    n_selected = max(1, int(total_elements * subset_ratio))
    selected_indices = rng.choice(total_elements, size=n_selected, replace=False)
    return set(selected_indices.tolist())
