import argparse
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_args_from_file, validate_analysis_args
from src.constants import ANALYSIS_DEFAULTS, IGBP_CLASS_NAMES, AnalysisArgDefaults
from src.logging_utils import log_args_config, setup_logging
from src.messages import ANALYSIS_INFO
from src.plotting_utils import create_confusion_matrix_plot, create_metrics_distribution_plot
from src.utils import log_and_raise

logger = logging.getLogger(__name__)


def make_metric_stats(series: pd.Series) -> dict[str, float]:
    """Summary statistics for a metric series."""
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
    }


def make_metric_summary(name: str, stats: dict[str, float]) -> dict[str, Any]:
    return {"name": name, "stats": stats}


def make_element_performance(element_id: str, accuracy: float, f1: float) -> dict[str, Any]:
    """Element-level performance summary."""
    return {"element_id": element_id, "accuracy": float(accuracy), "f1": float(f1)}


def make_element_metrics_analysis() -> dict[str, list[dict[str, Any]]]:
    return {"summary": [], "worst": [], "best": []}


def make_confusion_matrix_class_recall(class_name: str, recall: float, samples: int) -> dict[str, Any]:
    return {"class_name": class_name, "recall": float(recall), "samples": int(samples)}


def make_confusion_matrix_analysis() -> dict[str, list[dict[str, Any]]]:
    return {"per_class_recall": []}


class PredictionAnalyzer:
    def __init__(
        self,
        args: argparse.Namespace,
        class_names: list[str] | None = None,
    ) -> None:
        self.args = args
        self.class_names = class_names or list(IGBP_CLASS_NAMES["ALL_CLASSES"])
        self.output_dir: Path | None = None

    def set_output_dir(self, output_dir: Path) -> None:
        """Set the output directory for saving plots."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze_element_metrics(self, df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        """Analyze per-element metrics and identify problematic samples."""
        if self.output_dir is None:
            msg = "Output directory not set. Call set_output_dir() before running analysis."
            log_and_raise(ValueError, msg, logger_instance=logger)

        results = make_element_metrics_analysis()

        logger.info("Per-element metrics analysis")

        # Summary statistics
        logger.info("Summary Statistics:")
        for metric in ("accuracy", "mean_f1", "AA", "mIoU"):
            if metric not in df.columns:
                continue

            stats = make_metric_stats(df[metric])
            results["summary"].append(make_metric_summary(name=metric, stats=stats))

            logger.info(f"\n{metric}:")
            logger.info(f"  Mean:   {stats['mean']:.4f}")
            logger.info(f"  Std:    {stats['std']:.4f}")
            logger.info(f"  Min:    {stats['min']:.4f}")
            logger.info(f"  Max:    {stats['max']:.4f}")
            logger.info(f"  Median: {stats['median']:.4f}")

        # Find worst/best performing elements
        n_elements = 10

        logger.info(f"{n_elements} worst predictions (by accuracy):")

        worst_df = df.nsmallest(n_elements, "accuracy")
        for _, row in worst_df.iterrows():
            results["worst"].append(
                make_element_performance(
                    element_id=cast(str, row["element_id"]),
                    accuracy=float(row["accuracy"]),
                    f1=float(row["mean_f1"]),
                )
            )
            logger.info(f"{row['element_id']:60s}  Acc: {row['accuracy']:.4f}  F1: {row['mean_f1']:.4f}")

        logger.info(f"{n_elements} best predictions (by accuracy):")

        best_df = df.nlargest(n_elements, "accuracy")
        for _, row in best_df.iterrows():
            results["best"].append(
                make_element_performance(
                    element_id=cast(str, row["element_id"]),
                    accuracy=float(row["accuracy"]),
                    f1=float(row["mean_f1"]),
                )
            )
            logger.info(f"{row['element_id']:60s}  Acc: {row['accuracy']:.4f}  F1: {row['mean_f1']:.4f}")

        # Generate plots
        self._plot_metric_distributions(df)
        self._plot_accuracy_vs_f1(df)
        return results

    def _plot_metric_distributions(self, df: pd.DataFrame) -> None:
        if self.output_dir is None:
            return
        metrics_config = [
            ("accuracy", "Accuracy"),
            ("mean_f1", "Mean F1 Score"),
            ("AA", "Average Accuracy"),
            ("mIoU", "Mean IoU"),
        ]
        fig = create_metrics_distribution_plot(
            df,
            metrics_config,
            bins=30,
            figure_size=(12, 10),
            title="Per-Element Metrics Distributions",
            n_cols=2,
            show_mean=True,
            show_median=True,
        )

        save_path = self.output_dir / "element_metrics_distributions.png"
        fig.savefig(save_path, dpi=ANALYSIS_DEFAULTS["FIGURE_DPI"], bbox_inches="tight")
        logger.info(ANALYSIS_INFO.SAVED_PLOT.format(name="distribution plot", path=save_path))
        plt.close(fig)

    def _plot_accuracy_vs_f1(self, df: pd.DataFrame) -> None:
        """Plot accuracy vs F1 score scatter plot."""
        if self.output_dir is None:
            return

        if "accuracy" not in df.columns or "mean_f1" not in df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(df["accuracy"], df["mean_f1"], alpha=0.5, s=30)
        ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect agreement")
        ax.set_xlabel("Accuracy", fontsize=12)
        ax.set_ylabel("Mean F1 Score", fontsize=12)
        ax.set_title("Accuracy vs F1 Score", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / "accuracy_vs_f1.png"
        plt.savefig(save_path, dpi=ANALYSIS_DEFAULTS["FIGURE_DPI"], bbox_inches="tight")
        logger.info(ANALYSIS_INFO.SAVED_PLOT.format(name="scatter plot", path=save_path))
        plt.close()

    def visualize_confusion_matrix(self, cm: np.ndarray) -> dict[str, list[dict[str, Any]]]:
        if self.output_dir is None:
            msg = "Output directory not set. Call set_output_dir() before running analysis."
            log_and_raise(ValueError, msg, logger_instance=logger)

        logger.info("Confusion matrix analysis")

        n_classes = cm.shape[0]
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

        results = make_confusion_matrix_analysis()

        logger.info("Per-Class Accuracy (Recall):")
        for i in range(n_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
            recall = float(cm_norm[i, i])
            sample_count = int(cm[i, :].sum())
            results["per_class_recall"].append(
                make_confusion_matrix_class_recall(
                    class_name=class_name,
                    recall=recall,
                    samples=sample_count,
                )
            )
            logger.info(f"  {class_name:20s}: {recall:.4f} ({sample_count} samples)")

        fig = create_confusion_matrix_plot(
            cm,
            self.class_names,
            figure_size=(16, 7),
            show_normalized=True,
        )
        save_path = self.output_dir / "confusion_matrix.png"
        fig.savefig(save_path, dpi=ANALYSIS_DEFAULTS["FIGURE_DPI"], bbox_inches="tight")
        logger.info(ANALYSIS_INFO.SAVED_PLOT.format(name="confusion matrix", path=save_path))
        plt.close(fig)

        return results

    def analyze_summary_metrics(self, summary: Mapping[str, Any]) -> None:
        logger.info("Summary metrics")

        logger.info("Overall Metrics:")
        for key in ["AA", "mIoU", "accuracy", "mean_f1"]:
            if key in summary:
                val = summary[key]
                logger.info(f"  {key:12s}: {val:.4f} ({val * 100:.2f}%)")

        if "IoU" in summary:
            logger.info("Per-Class IoU:")
            for i, iou in enumerate(summary["IoU"]):
                class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
                logger.info(f"  {class_name:20s}: {iou:.4f}")

        if "f1" in summary:
            logger.info("Per-Class F1:")
            for i, f1 in enumerate(summary["f1"]):
                class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
                logger.info(f"  {class_name:20s}: {f1:.4f}")


def run_analysis(args_file_path: str, args_override: Mapping[str, Any] | None = None) -> None:
    """Run analysis with programmatic arguments.

    Args:
        args_file_path: Path to arguments file
        args_override: Optional dict to override specific arguments
    """
    # Load and validate arguments
    args = load_args_from_file(args_file_path, AnalysisArgDefaults.copy())

    # Apply overrides if provided
    if args_override:
        for key, value in args_override.items():
            setattr(args, key, value)

    validate_analysis_args(args)

    _run_analysis_with_args(args)


def _run_analysis_with_args(args: argparse.Namespace) -> None:
    """Internal function to run analysis with loaded arguments."""

    # Setup paths
    log_dir_path = cast(str, os.path.join(args.log_dir, args.experiment_name))
    analysis_dir_path = os.path.join(log_dir_path, args.analysis_dir)
    os.makedirs(analysis_dir_path, exist_ok=True)

    log_file_path = os.path.join(log_dir_path, "logs.log")
    setup_logging(str(args.log_level), log_file_path)

    # Log config
    log_args_config(args, logger=logger, header="Config", blank_line=True)

    out_dir_path = Path(os.path.join(log_dir_path, args.out_dir))
    predictions_dir = os.path.join(log_dir_path, "predictions")

    class_names: list[str] = args.class_names if args.class_names else [str(i) for i in range(10)]
    analyzer = PredictionAnalyzer(args, class_names=class_names)
    analyzer.set_output_dir(Path(analysis_dir_path))

    # Element metrics
    element_csv = os.path.join(predictions_dir, "element_metrics.csv")
    if os.path.exists(element_csv):
        logger.info(f"Loading element metrics from {element_csv}")

        df = pd.read_csv(element_csv)
        logger.info(f"Loaded metrics for {len(df)} elements")
        logger.info(f"Available metrics: {list(df.columns)}")

        if "accuracy" in df.columns:
            df_scored = df[df["accuracy"].notna()]
            logger.info(f"Elements with scores: {len(df_scored)}")

            if len(df_scored) > 0:
                analyzer.analyze_element_metrics(df_scored)
            else:
                logger.warning("No scored elements found")
        else:
            logger.warning("No accuracy metrics found in element_metrics.csv")
    else:
        logger.warning(f"Element metrics file not found: {element_csv}")

    # Confusion matrix
    cm_csv = os.path.join(predictions_dir, "confusion_matrix.csv")
    if os.path.exists(cm_csv):
        logger.info(f"Loading confusion matrix from {cm_csv}")

        cm = np.loadtxt(cm_csv, delimiter=",")
        logger.info(f"Confusion matrix shape: {cm.shape}")
        analyzer.visualize_confusion_matrix(cm)
    else:
        logger.warning(f"Confusion matrix file not found: {cm_csv}")

    # Summary metrics
    summary_json = os.path.join(predictions_dir, "summary_metrics.json")
    if os.path.exists(summary_json):
        with open(summary_json) as f:
            summary = json.load(f)
        analyzer.analyze_summary_metrics(summary)
    else:
        logger.warning(f"Summary metrics file not found: {summary_json}")

    logger.info(f"All analysis outputs saved to {out_dir_path}")


def main() -> None:
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Analyze prediction results")
    parser.add_argument(
        "--args_file_path",
        type=str,
        required=True,
        help="Path to a file listing arguments as key=value per line",
    )

    parsed = parser.parse_args()
    run_analysis(parsed.args_file_path)


if __name__ == "__main__":
    main()
