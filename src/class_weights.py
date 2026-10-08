import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from .utils import log_and_raise

logger = logging.getLogger(__name__)


def compute_class_weights(
    class_counts: np.ndarray,
    method: str = "effective",
    beta: float = 0.9999,
) -> torch.Tensor:
    """Compute class weights from class counts using the specified method.

    Args:
        class_counts: Array of pixel counts per class (shape: [n_classes])
        method: Method for computing weights (inverse, effective, median)
        beta: Beta parameter for effective number method (default: 0.9999)

    Returns:
        Tensor of class weights (shape: [n_classes])

    Raises:
        ValueError: If class_counts contains zeros or negative values

    Example:
        >>> counts = np.array([10000, 5000, 100, 50])  # Class distribution
        >>> weights = compute_class_weights(counts, method="effective")
        >>> print(weights)
        tensor([0.5, 1.0, 20.0, 30.0])

    """
    if np.any(class_counts <= 0):
        log_and_raise(ValueError, f"class_counts must contain only positive values, got: {class_counts}", logger_instance=logger)

    n_classes = len(class_counts)

    if method == "inverse":
        # Inverse frequency: weight = 1 / count
        weights = 1.0 / class_counts

    elif method == "effective":
        # Effective number: prevents extreme weights for rare classes
        # E_n = (1 - beta^n) / (1 - beta)
        # weight = (1 - beta) / (1 - beta^n)
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num

    elif method == "median":
        # Median frequency balancing
        median_freq = np.median(class_counts)
        weights = median_freq / class_counts

    else:
        log_and_raise(ValueError, f"Unknown weight method: {method}", logger_instance=logger)

    # Normalize weights so they sum to n_classes (average weight = 1.0)
    weights = weights / weights.sum() * n_classes

    return torch.tensor(weights, dtype=torch.float32)


def compute_class_weights_from_dataset(
    dataset: Iterable[dict[str, Any]],
    n_classes: int,
    method: str = "effective",
    beta: float = 0.9999,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Compute class weights by iterating through a dataset.

    Args:
        dataset: PyTorch dataset with segmentation masks
        n_classes: Number of classes
        method: Method for computing weights
        beta: Beta parameter for effective number method
        ignore_index: Label index to ignore in computation

    Returns:
        Tensor of class weights (shape: [n_classes])

    Note:
        This can be slow for large datasets. Consider using pre-computed statistics
        or a subset of the data for faster computation.

    """
    class_counts = np.zeros(n_classes, dtype=np.int64)

    for sample in dataset:
        mask: torch.Tensor | np.ndarray = sample["label"]
        if isinstance(mask, torch.Tensor):
            mask_np: np.ndarray = mask.numpy()
        else:
            mask_np = mask

        # Exclude ignore index
        valid_mask: np.ndarray = mask_np != ignore_index
        valid_pixels: np.ndarray = mask_np[valid_mask]

        # Count pixels per class
        for c in range(n_classes):
            class_counts[c] += int(np.sum(valid_pixels == c))

    return compute_class_weights(class_counts, method=method, beta=beta)


# Pre-computed class distributions for DFC2020 dataset (approximate)
# These can be used as a starting point or fallback
DFC2020_CLASS_COUNTS_10CLASS = np.array(
    [
        3500000,  # Forest
        2500000,  # Shrubland
        800000,  # Savanna
        2200000,  # Grassland
        200000,  # Wetlands
        1800000,  # Croplands
        300000,  # Urban/Built-up
        100000,  # Snow/Ice
        1500000,  # Barren
        1200000,  # Water
    ]
)

DFC2020_CLASS_COUNTS_9CLASS = np.array(
    [
        3500000,  # Forest
        2500000,  # Shrubland
        2200000,  # Grassland (merged with savanna)
        200000,  # Wetlands
        1800000,  # Croplands
        300000,  # Urban/Built-up
        100000,  # Snow/Ice
        1500000,  # Barren
        1200000,  # Water
    ]
)


def get_dfc2020_class_weights(
    n_classes: int = 10,
    method: str = "effective",
    beta: float = 0.9999,
) -> torch.Tensor:
    """Get pre-computed class weights for DFC2020 dataset.

    Args:
        n_classes: Number of classes (9 or 10)
        method: Method for computing weights
        beta: Beta parameter for effective number method

    Returns:
        Tensor of class weights

    Raises:
        ValueError: If n_classes is not 9 or 10

    """
    if n_classes == 10:
        counts = DFC2020_CLASS_COUNTS_10CLASS
    elif n_classes == 9:
        counts = DFC2020_CLASS_COUNTS_9CLASS
    else:
        log_and_raise(ValueError, f"n_classes must be 9 or 10 for DFC2020, got {n_classes}", logger_instance=logger)

    return compute_class_weights(counts, method=method, beta=beta)
