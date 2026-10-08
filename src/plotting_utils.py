from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import compute_per_class_metrics_from_cm


def create_metrics_distribution_plot(
    element_df: pd.DataFrame,
    metrics: Sequence[tuple[str, str]],
    *,
    bins: int,
    figure_size: tuple[float, float],
    title: str,
    n_cols: int = 2,
    show_mean: bool = False,
    show_median: bool = False,
) -> plt.Figure:
    """Create a grid of metric distribution histograms."""
    n_metrics = len(metrics)
    n_cols = max(1, min(n_cols, n_metrics))
    n_rows = max(1, math.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figure_size)
    axes_flat = [axes] if n_metrics == 1 else np.array(axes).flatten().tolist()

    for idx, (metric, label) in enumerate(metrics):
        ax = axes_flat[idx]
        if metric not in element_df.columns:
            ax.set_visible(False)
            continue

        ax.hist(element_df[metric], bins=bins, edgecolor="black", alpha=0.7)
        if show_mean:
            ax.axvline(
                element_df[metric].mean(),
                color="red",
                linestyle="--",
                linewidth=2,
                label="Mean",
            )
        if show_median:
            ax.axvline(
                element_df[metric].median(),
                color="green",
                linestyle="--",
                linewidth=2,
                label="Median",
            )

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Distribution of {label}", fontsize=12, fontweight="bold")
        if show_mean or show_median:
            ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def create_confusion_matrix_plot(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str],
    *,
    figure_size: tuple[float, float],
    show_normalized: bool = False,
) -> plt.Figure:
    if show_normalized:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            ax=ax1,
            xticklabels=class_names,
            yticklabels=class_names,
        )
        ax1.set_xlabel("Predicted Class", fontsize=12)
        ax1.set_ylabel("Actual Class", fontsize=12)
        ax1.set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")

        cm_norm = confusion_matrix / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-12)
        sns.heatmap(
            cm_norm * 100,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            ax=ax2,
            vmin=0,
            vmax=100,
            xticklabels=class_names,
            yticklabels=class_names,
        )
        ax2.set_xlabel("Predicted Class", fontsize=12)
        ax2.set_ylabel("Actual Class", fontsize=12)
        ax2.set_title(
            "Confusion Matrix (Normalized by Row)",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=figure_size)
    im = ax.imshow(confusion_matrix, cmap="Blues")
    num_classes = confusion_matrix.shape[0]
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                str(int(confusion_matrix[i, j])),
                ha="center",
                va="center",
                color="black" if confusion_matrix[i, j] < confusion_matrix.max() / 2 else "white",
            )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def create_per_class_iou_plot(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str],
    *,
    figure_size: tuple[float, float],
) -> plt.Figure:
    """Create per-class IoU bar chart from confusion matrix."""
    num_classes = confusion_matrix.shape[0]
    ious = compute_per_class_metrics_from_cm(confusion_matrix)["iou"]

    fig, ax = plt.subplots(figsize=figure_size)
    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.8, num_classes))
    bars = ax.bar(range(num_classes), ious, color=colors, edgecolor="black", alpha=0.8)

    for bar, iou in zip(bars, ious, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{iou:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("IoU")
    ax.set_title("Per-Class IoU")
    ax.set_ylim(0, min(1.0, max(ious) * 1.15) if ious else 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    mean_iou = float(np.mean(ious)) if ious else 0.0
    ax.axhline(
        y=mean_iou,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Mean IoU: {mean_iou:.3f}",
    )
    ax.legend()

    fig.tight_layout()
    return fig
