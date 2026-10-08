import argparse
import logging
import os
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_args_from_file, validate_plot_args
from src.constants import PLOT_DEFAULTS, PlotArgDefaults
from src.logging_utils import log_args_config, setup_logging
from src.messages import PLOT_ERRORS, PLOT_INFO
from src.utils import log_and_raise

logger = logging.getLogger(__name__)


class MetricsPlotter:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir: Path | None = None

    def set_output_dir(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_loss_curves(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> None:
        if self.output_dir is None:
            msg = PLOT_ERRORS.OUTPUT_DIR_NOT_SET
            log_and_raise(ValueError, msg, logger_instance=logger)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot training loss (raw)
        ax.plot(
            train_df["step"],
            train_df["loss"],
            alpha=0.3,
            label="Train Loss (raw)",
            linewidth=1,
        )

        # Plot smoothed training loss
        window = min(
            50,
            len(train_df) // 10,
        )
        if window > 1:
            train_smooth = train_df["loss"].rolling(window=window, center=True).mean()
            ax.plot(
                train_df["step"],
                train_smooth,
                label=f"Train Loss (smoothed, w={window})",
                linewidth=PLOT_DEFAULTS["LINE_WIDTH"],
            )

        # Plot validation loss
        if val_df is not None and "loss" in val_df.columns:
            ax.plot(
                val_df["step"],
                val_df["loss"],
                label="Val Loss",
                marker="o",
                linewidth=PLOT_DEFAULTS["LINE_WIDTH"],
                markersize=PLOT_DEFAULTS["MARKER_SIZE"],
            )

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=PLOT_DEFAULTS["ALPHA_GRID"])

        plt.tight_layout()

        save_path = self.output_dir / "loss_curves.png"
        plt.savefig(save_path, dpi=PLOT_DEFAULTS["FIGURE_DPI"], bbox_inches="tight")
        logger.info(PLOT_INFO.SAVED_PLOT.format(name="loss curves", path=save_path))
        plt.close()

    def plot_metrics_over_time(self, val_df: pd.DataFrame) -> None:
        if self.output_dir is None:
            msg = PLOT_ERRORS.OUTPUT_DIR_NOT_SET
            log_and_raise(ValueError, msg, logger_instance=logger)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes_flat = axes.flatten()

        metrics_config = [
            ("accuracy", "Accuracy"),
            ("mean_f1", "Mean F1 Score"),
            ("AA", "Average Accuracy (AA)"),
            ("mIoU", "Mean IoU"),
        ]

        for idx, (metric_key, title) in enumerate(metrics_config):
            ax = axes_flat[idx]

            if metric_key not in val_df.columns:
                logger.warning(f"Metric '{metric_key}' not found in dataframe")
                ax.set_visible(False)
                continue

            ax.plot(
                val_df["step"],
                val_df[metric_key] * 100,
                marker="o",
                linewidth=PLOT_DEFAULTS["LINE_WIDTH"],
                markersize=PLOT_DEFAULTS["MARKER_SIZE"],
            )
            ax.set_xlabel("Training Step", fontsize=11)
            ax.set_ylabel(f"{title} (%)", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=PLOT_DEFAULTS["ALPHA_GRID"])

            # Show min/max values
            min_val = val_df[metric_key].min() * 100
            max_val = val_df[metric_key].max() * 100
            ax.text(
                0.02,
                0.98,
                f"Min: {min_val:.2f}%\nMax: {max_val:.2f}%",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                fontsize=9,
            )

        plt.suptitle("Validation Metrics Over Time", fontsize=16, fontweight="bold", y=1.00)
        plt.tight_layout()

        save_path = self.output_dir / "metrics_over_time.png"
        plt.savefig(save_path, dpi=PLOT_DEFAULTS["FIGURE_DPI"], bbox_inches="tight")
        logger.info(PLOT_INFO.SAVED_PLOT.format(name="metrics over time", path=save_path))
        plt.close()

    def plot_comparison(self, val_df: pd.DataFrame) -> None:
        """Compare different metrics on the same plot."""
        if self.output_dir is None:
            msg = PLOT_ERRORS.OUTPUT_DIR_NOT_SET
            log_and_raise(ValueError, msg, logger_instance=logger)

        fig, ax = plt.subplots(figsize=(12, 6))

        metrics_config = [
            ("accuracy", "Accuracy"),
            ("mean_f1", "Mean F1"),
            ("AA", "Average Accuracy"),
            ("mIoU", "Mean IoU"),
        ]

        for metric_key, label in metrics_config:
            if metric_key not in val_df.columns:
                continue

            ax.plot(
                val_df["step"],
                val_df[metric_key] * 100,
                label=label,
                marker="o",
                linewidth=PLOT_DEFAULTS["LINE_WIDTH"],
                markersize=3,
            )

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Metric (%)", fontsize=12)
        ax.set_title("Comparison of Validation Metrics", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=PLOT_DEFAULTS["ALPHA_GRID"])

        plt.tight_layout()

        save_path = self.output_dir / "metrics_comparison.png"
        plt.savefig(save_path, dpi=PLOT_DEFAULTS["FIGURE_DPI"], bbox_inches="tight")
        logger.info(PLOT_INFO.SAVED_PLOT.format(name="metrics comparison", path=save_path))
        plt.close()

    def log_summary_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {"train": {}, "val_final": {}, "val_best": {}}

        logger.info("Summary statistics")

        # Training statistics
        final_train_loss = float(train_df["loss"].iloc[-1])
        results["train"]["final_loss"] = final_train_loss
        logger.info("Final Training Loss:")
        logger.info(f"  {final_train_loss:.6f}")

        # Validation statistics
        if val_df is not None and len(val_df) > 0:
            logger.info("Validation Metrics (final values):")
            for col in ["loss", "accuracy", "mean_f1", "AA", "mIoU"]:
                if col not in val_df.columns:
                    continue

                final_val = float(val_df[col].iloc[-1])
                results["val_final"][col] = final_val

                if col != "loss":
                    logger.info(f"  {col:12s}: {final_val * 100:6.2f}%")
                else:
                    logger.info(f"  {col:12s}: {final_val:.6f}")

            logger.info("Validation Metrics (best values):")
            for col in ["accuracy", "mean_f1", "AA", "mIoU"]:
                if col not in val_df.columns:
                    continue

                best_val = float(val_df[col].max())
                best_step = int(pd.to_numeric(val_df.loc[val_df[col].idxmax(), "step"], errors="coerce"))
                results["val_best"][col] = {"value": best_val, "step": best_step}
                logger.info(f"  {col:12s}: {best_val * 100:6.2f}% (at step {best_step})")

        return results


def run_plotting(args_file_path: str, args_override: dict[str, Any] | None = None) -> None:
    """Run plotting with programmatic arguments.

    Args:
        args_file_path: Path to arguments file
        args_override: Optional dict to override specific arguments
    """
    # Load and validate arguments
    args = load_args_from_file(args_file_path, PlotArgDefaults.copy())

    # Apply overrides if provided
    if args_override:
        for key, value in args_override.items():
            setattr(args, key, value)

    validate_plot_args(args)

    _run_plotting_with_args(args)


def _run_plotting_with_args(args: argparse.Namespace) -> None:
    """Internal function to run plotting with loaded arguments."""
    # Setup paths
    log_dir_path = cast(str, os.path.join(args.log_dir, args.experiment_name))
    out_dir_path = Path(os.path.join(log_dir_path, args.out_dir))
    os.makedirs(out_dir_path, exist_ok=True)

    log_file_path = os.path.join(log_dir_path, "logs.log")
    setup_logging(str(args.log_level), log_file_path)

    # Log config
    log_args_config(args, logger=logger, header="Config", blank_line=True)

    metrics_dir_path = os.path.join(log_dir_path, "metrics")
    if not os.path.exists(metrics_dir_path):
        msg = f"Metrics directory not found: {metrics_dir_path}. Ensure training metrics were saved during training."
        log_and_raise(ValueError, msg, logger_instance=logger)

    # Load metrics
    train_csv = os.path.join(metrics_dir_path, "train_metrics.csv")
    val_csv = os.path.join(metrics_dir_path, "val_metrics.csv")

    if not os.path.exists(train_csv):
        msg = f"Training metrics not found: {train_csv}. Ensure train_metrics.csv was generated during training."
        log_and_raise(ValueError, msg, logger_instance=logger)

    logger.info(f"Loading metrics from {metrics_dir_path}")

    train_df = pd.read_csv(train_csv)
    logger.info(f"Train metrics: {len(train_df)} steps")

    val_df: pd.DataFrame | None = None
    if os.path.exists(val_csv):
        val_df = pd.read_csv(val_csv)
        logger.info(f"Validation metrics: {len(val_df)} steps")
    else:
        logger.warning(f"Validation metrics not found: {val_csv}")

    plotter = MetricsPlotter(args)
    plotter.set_output_dir(Path(metrics_dir_path))

    logger.info("Generating plots...")

    # Loss curves (always generated)
    plotter.plot_loss_curves(train_df, val_df)

    # Validation metrics plots (only if val_df exists)
    if val_df is not None and len(val_df) > 0:
        plotter.plot_metrics_over_time(val_df)
        plotter.plot_comparison(val_df)

    plotter.log_summary_statistics(train_df, val_df)

    logger.info(f"All plots saved to {out_dir_path}")


def main() -> None:
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Plot training metrics and loss curves")
    parser.add_argument(
        "--args_file_path",
        type=str,
        required=True,
        help="Path to a file listing arguments as key=value per line",
    )

    parsed = parser.parse_args()
    run_plotting(parsed.args_file_path)


if __name__ == "__main__":
    main()
