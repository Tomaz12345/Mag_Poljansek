import json
import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import get_model_defaults_for_type, load_args_from_file
from src.constants import (
    CONFIG_PARAMETER_GROUPS,
    EXCLUDED_CONFIG_PARAMS,
    IGBP_CLASSES,
    IGBP_CLASSES_NO_SAVANNA,
    MODEL_TYPES,
    ReportDefaults,
    TrainArgDefaults,
)
from src.messages import REPORTING_ERRORS, REPORTING_WARNINGS
from src.plotting_utils import (
    create_confusion_matrix_plot,
    create_metrics_distribution_plot,
    create_per_class_iou_plot,
)
from src.show_image import (
    create_sample_figure,
    get_display_config,
    get_sorted_band_names,
    load_bands_and_gt,
    load_prediction,
)
from src.utils import classnames, compute_per_class_metrics, log_and_raise, validate_string_value

logger = logging.getLogger(__name__)


def _resolve_class_names(num_classes: int) -> list[str]:
    if num_classes == 10:
        return list(IGBP_CLASSES)
    if num_classes == 9:
        return list(IGBP_CLASSES_NO_SAVANNA)
    return [f"Class {i}" for i in range(num_classes)]


class ConfigurationCollector:
    """Collects configuration data from args files and merges with defaults."""

    def collect(self, args_file_path: str) -> dict[str, Any] | None:
        """
        Collect configuration data from the combined args file.

        Args:
            args_file_path: Path to the combined args file (common + model-specific)

        Returns:
            Dictionary containing configuration data with user vs default indicators
        """
        try:
            # Load all configuration from a single file
            general_defaults = TrainArgDefaults.copy()
            all_args = load_args_from_file(args_file_path, general_defaults, use_model_defaults=True)

            model_type = getattr(all_args, "model", None)
            if not model_type:
                log_and_raise(ValueError, "Model type missing in args file", logger_instance=logger)

            validate_string_value(model_type, MODEL_TYPES, "model type")
            model_defaults = get_model_defaults_for_type(model_type)

            # Split namespace back into general vs model-specific views
            all_vars = vars(all_args)
            general_config = self._merge_with_defaults(
                {k: v for k, v in all_vars.items() if k in general_defaults},
                general_defaults,
            )
            model_config = self._merge_with_defaults(
                {k: v for k, v in all_vars.items() if k in model_defaults},
                model_defaults,
            )

            general_grouped = self.group_config_by_purpose(general_config)

            # Filter model config: hide pretrained_path/pretrained_encoder when use_pretrained=False
            model_config = self._filter_model_pretrained(model_config)

            return {
                "general": general_config,
                "general_grouped": general_grouped,
                "model": model_config,
                "args_file_path": str(args_file_path),
            }

        except Exception as e:
            logger.error(REPORTING_ERRORS.DATA_COLLECTION_FAILED.format(section="configuration", error=str(e)))
            return None

    def group_config_by_purpose(self, config_data: dict[str, dict[str, Any]]) -> OrderedDict[str, dict[str, dict[str, Any]]]:
        """
        Group configuration parameters by purpose with filtering.

        Groups params according to CONFIG_PARAMETER_GROUPS. For groups with an
        enable_key that is False, only the enable param is shown. Default-value
        params are hidden unless in the group's always_show list or they start
        with 'enable_'.
        """
        grouped: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
        claimed_params: set[str] = set()

        for group_name, group_def in CONFIG_PARAMETER_GROUPS.items():
            enable_key = group_def.get("enable_key")
            always_show = set(group_def.get("always_show", []))

            # Determine which params belong to this group
            group_params: list[str] = []
            if "params" in group_def:
                group_params = [p for p in group_def["params"] if p in config_data]
            if "params_prefix" in group_def:
                prefix = group_def["params_prefix"]
                for p in config_data:
                    if p.startswith(prefix) and p not in claimed_params:
                        group_params.append(p)
            if "extra_params" in group_def:
                for p in group_def["extra_params"]:
                    if p in config_data and p not in group_params:
                        group_params.append(p)

            group_params = [p for p in group_params if p not in claimed_params and p not in EXCLUDED_CONFIG_PARAMS]

            if not group_params:
                continue

            group_disabled = False
            if enable_key and enable_key in config_data:
                enable_val = config_data[enable_key]["value"]
                if enable_val is False or str(enable_val).lower() == "false":
                    group_disabled = True

            # Determine irrelevant enum-based children
            irrelevant_children: set[str] = set()
            selector_key = group_def.get("selector_key")
            children_by_value: dict[str, list[str]] | None = group_def.get("children_by_value")
            if selector_key and children_by_value and selector_key in config_data:
                current_value = str(config_data[selector_key]["value"]).lower()
                relevant = set(children_by_value.get(current_value, []))
                all_enum_children: set[str] = set()
                for params_list in children_by_value.values():
                    all_enum_children.update(params_list)
                irrelevant_children = all_enum_children - relevant

            # Filter params
            visible_params: OrderedDict[str, dict[str, Any]] = OrderedDict()
            for p in sorted(group_params):
                info = config_data[p]
                if group_disabled:
                    # Only show the enable key itself
                    if p == enable_key:
                        visible_params[p] = info
                elif p in irrelevant_children:
                    # Skip params irrelevant to current enum value
                    continue
                else:
                    # Show if: non-default, or in always_show, or starts with enable_
                    if info["source"] == "user" or p in always_show or p.startswith("enable_") or p.startswith("use_"):
                        visible_params[p] = info

            if visible_params:
                grouped[group_name] = visible_params
                claimed_params.update(group_params)

        # Catch-all "Other" group for unclaimed params
        other_params: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for p in sorted(config_data):
            if p not in claimed_params and p not in EXCLUDED_CONFIG_PARAMS:
                info = config_data[p]
                if info["source"] == "user" or p.startswith("enable_"):
                    other_params[p] = info
        if other_params:
            grouped["Other"] = other_params

        return grouped

    @staticmethod
    def _filter_model_pretrained(model_config: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Hide pretrained_path and pretrained_encoder when use_pretrained is False."""
        use_pretrained_info = model_config.get("use_pretrained")
        if use_pretrained_info is None:
            return model_config
        use_pretrained_val = use_pretrained_info["value"]
        if use_pretrained_val is False or str(use_pretrained_val).lower() == "false":
            return {k: v for k, v in model_config.items() if k not in ("pretrained_path", "pretrained_encoder")}
        return model_config

    def _merge_with_defaults(self, user_args: dict[str, Any], defaults: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Merge user arguments with defaults and track source.

        Args:
            user_args: Arguments loaded from file
            defaults: Default values dictionary

        Returns:
            Dictionary with format: {param: {value, source, default}}
        """
        result = {}
        for param, default_val in defaults.items():
            # Skip class types (like DeepLabArgDefaults, UNetArgDefaults)
            if isinstance(default_val, type):
                continue

            user_val = user_args.get(param, default_val)
            source = "user" if user_val != default_val else "default"
            result[param] = {
                "value": user_val,
                "source": source,
                "default": default_val,
            }
        return result


class TrainingCollector:
    def collect(self, experiment_dir: Path) -> dict[str, Any] | None:
        """
        Collect training data from metrics files.
        """
        try:
            metrics_dir = experiment_dir / "metrics"
            if not metrics_dir.exists():
                logger.warning(REPORTING_WARNINGS.METRICS_FILE_MISSING.format(path=metrics_dir))
                return None

            data: dict[str, Any] = {}

            # Load training metrics
            train_metrics_file = metrics_dir / "train_metrics.csv"
            if train_metrics_file.exists():
                train_df = pd.read_csv(train_metrics_file)
                data["train_metrics"] = train_df

            # Load validation metrics
            val_metrics_file = metrics_dir / "val_metrics.csv"
            if val_metrics_file.exists():
                val_df = pd.read_csv(val_metrics_file)
                data["val_metrics"] = val_df

                # Get summary statistics from validation
                if not val_df.empty:
                    final_metrics = val_df.iloc[-1]
                    data["final_metrics"] = {
                        "loss": float(final_metrics.get("loss", 0)),
                        "AA": float(final_metrics.get("AA", 0)),
                        "mIoU": float(final_metrics.get("mIoU", 0)),
                        "accuracy": float(final_metrics.get("accuracy", 0)),
                        "mean_f1": float(final_metrics.get("mean_f1", 0)),
                    }
                    data["final_metrics_source"] = "validation"

                    best_miou_idx = val_df["mIoU"].idxmax()
                    best_metrics = val_df.loc[best_miou_idx]
                    data["best_metrics"] = {
                        "step": int(best_metrics["step"]),
                        "mIoU": float(best_metrics["mIoU"]),
                    }

            elif "train_metrics" in data and not data["train_metrics"].empty:
                # Fallback: use training metrics when no validation metrics exist
                train_df = data["train_metrics"]
                final_train = train_df.iloc[-1]
                data["final_metrics"] = {
                    "loss": float(final_train["loss"]),
                }
                data["final_metrics_source"] = "training"

            # Generate plots
            data["plots"] = {}

            # Loss curves plot
            if "train_metrics" in data and "val_metrics" in data:
                fig = self._create_loss_curves_plot(data["train_metrics"], data["val_metrics"])
                data["plots"]["loss_curves"] = fig
            elif "train_metrics" in data and not data["train_metrics"].empty:
                fig = self._create_train_only_loss_plot(data["train_metrics"])
                data["plots"]["loss_curves"] = fig

            # Metrics over time plot
            if "val_metrics" in data:
                fig = self._create_metrics_plot(data["val_metrics"])
                data["plots"]["metrics_over_time"] = fig

            # Look for pre-existing loss_curves.png saved by plot_metrics.py
            for subdir in ("results", "analysis", "plots", ""):
                candidate = experiment_dir / subdir / "loss_curves.png" if subdir else experiment_dir / "loss_curves.png"
                if candidate.exists():
                    data["loss_curves_file"] = str(candidate)
                    break

            # Extract training duration if available
            logs_file = experiment_dir / "logs.log"
            if logs_file.exists():
                duration = self._extract_training_duration(logs_file)
                if duration:
                    data["training_duration"] = duration

            return data

        except Exception as e:
            logger.error(REPORTING_ERRORS.DATA_COLLECTION_FAILED.format(section="training", error=str(e)))
            return None

    def _create_loss_curves_plot(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(ReportDefaults["FIGURE_WIDTH"], ReportDefaults["FIGURE_HEIGHT"]))

        ax.plot(train_df["step"], train_df["loss"], label="Training Loss", alpha=0.7)
        ax.plot(val_df["step"], val_df["loss"], label="Validation Loss", marker="o")

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_train_only_loss_plot(self, train_df: pd.DataFrame) -> plt.Figure:
        """Create loss curve plot from training data only (no validation)."""
        fig, ax = plt.subplots(figsize=(ReportDefaults["FIGURE_WIDTH"], ReportDefaults["FIGURE_HEIGHT"]))

        ax.plot(train_df["step"], train_df["loss"], label="Training Loss", alpha=0.7)

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_metrics_plot(self, val_df: pd.DataFrame) -> plt.Figure:
        """Create metrics over time plot."""
        fig, ax = plt.subplots(figsize=(ReportDefaults["FIGURE_WIDTH"], ReportDefaults["FIGURE_HEIGHT"]))

        metrics = ["AA", "mIoU", "accuracy", "mean_f1"]
        for metric in metrics:
            if metric in val_df.columns:
                ax.plot(val_df["step"], val_df[metric], label=metric, marker="o")

        ax.set_xlabel("Step")
        ax.set_ylabel("Metric Value")
        ax.set_title("Validation Metrics Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _extract_training_duration(self, logs_file: Path) -> float | None:
        """Extract training duration from logs."""
        # This is a placeholder - actual implementation would parse logs
        return None


class PredictionCollector:
    def collect(
        self,
        predictions_dir: Path,
        experiment_dir: Path | None = None,
        num_samples: int | None = None,
        sample_ids: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """
        Collect prediction data from results files.

        Args:
            predictions_dir: Path to predictions directory
            experiment_dir: Path to experiment directory (for loading raw data)
            num_samples: Number of sample predictions to include (default from ReportDefaults)
            sample_ids: Specific element IDs to include (overrides random selection)

        Returns:
            Dictionary containing prediction metrics and visualizations
        """
        try:
            if not predictions_dir.exists():
                logger.warning(REPORTING_WARNINGS.METRICS_FILE_MISSING.format(path=predictions_dir))
                return None

            data: dict[str, Any] = {}

            # Load element metrics
            element_metrics_file = predictions_dir / "element_metrics.csv"
            if element_metrics_file.exists():
                element_df = pd.read_csv(element_metrics_file)
                data["element_metrics"] = element_df

            # Load summary metrics
            summary_file = predictions_dir / "summary_metrics.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    data["summary_metrics"] = json.load(f)

            # Load confusion matrix
            confusion_file = predictions_dir / "confusion_matrix.csv"
            if confusion_file.exists():
                confusion_matrix = pd.read_csv(confusion_file, header=None).values
                data["confusion_matrix"] = confusion_matrix

                # Generate confusion matrix heatmap
                class_names = _resolve_class_names(confusion_matrix.shape[0])
                fig = create_confusion_matrix_plot(
                    confusion_matrix,
                    class_names,
                    figure_size=(ReportDefaults["FIGURE_WIDTH"], ReportDefaults["FIGURE_WIDTH"]),
                    show_normalized=False,
                )
                data["confusion_matrix_plot"] = fig

            # Generate distribution plots if element metrics available
            if "element_metrics" in data:
                fig = create_metrics_distribution_plot(
                    data["element_metrics"],
                    [
                        ("accuracy", "accuracy"),
                        ("AA", "AA"),
                        ("mIoU", "mIoU"),
                        ("mean_f1", "mean_f1"),
                    ],
                    bins=30,
                    figure_size=(ReportDefaults["FIGURE_WIDTH"], ReportDefaults["FIGURE_WIDTH"]),
                    title="Metrics Distributions",
                    n_cols=2,
                    show_mean=False,
                    show_median=False,
                )
                data["metrics_distribution_plot"] = fig

            # Compute per-class metrics from confusion matrix
            if "confusion_matrix" in data:
                class_names = _resolve_class_names(data["confusion_matrix"].shape[0])
                per_class_metrics = compute_per_class_metrics(
                    data["confusion_matrix"],
                    class_names,
                    format_values=True,
                )
                data["per_class_metrics"] = per_class_metrics

            # Generate sample prediction visualizations
            if experiment_dir is not None and "element_metrics" in data:
                n_samples = num_samples if num_samples is not None else 5
                sample_plots = self._create_sample_prediction_figures(
                    predictions_dir,
                    experiment_dir,
                    data["element_metrics"],
                    n_samples,
                    sample_ids,
                )
                if sample_plots:
                    data["sample_prediction_plots"] = sample_plots

            return data

        except Exception as e:
            logger.error(REPORTING_ERRORS.DATA_COLLECTION_FAILED.format(section="prediction", error=str(e)))
            return None

    def _create_sample_prediction_figures(
        self,
        predictions_dir: Path,
        experiment_dir: Path,
        element_df: pd.DataFrame,
        num_samples: int,
        sample_ids: list[str] | None,
    ) -> list[plt.Figure]:
        """Create sample prediction visualization figures.

        Each figure shows: RGB composite, individual bands, ground truth, and prediction
        with colorized labels and legend.

        Args:
            predictions_dir: Path to predictions directory with TIFF files
            experiment_dir: Path to experiment directory (contains checkpoints/args.pkl)
            element_df: DataFrame with element metrics
            num_samples: Number of samples to include
            sample_ids: Specific element IDs (without .tif extension)

        Returns:
            List of matplotlib Figure objects
        """
        # Load training args to determine sensor config and data paths
        args_pkl_path = experiment_dir / "checkpoints" / "args.pkl"
        if not args_pkl_path.exists():
            logger.warning(f"Cannot load args.pkl from {args_pkl_path}, skipping sample predictions")
            return []

        try:
            with open(args_pkl_path, "rb") as f:
                train_args = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load args.pkl: {e}")
            return []

        use_s2hr = getattr(train_args, "use_s2hr", True)
        use_s2mr = getattr(train_args, "use_s2mr", True)
        use_s2lr = getattr(train_args, "use_s2lr", True)
        use_s1 = getattr(train_args, "use_s1", True)
        no_savanna = getattr(train_args, "no_savanna", True)
        border_exclusion = getattr(train_args, "border_exclusion", False)
        border_exclusion_size: int | None = getattr(train_args, "border_exclusion_size", 10) if border_exclusion else None

        # Determine data directory
        # For prediction, we need the data_dir used for the prediction dataset
        data_dir = getattr(train_args, "data_dir", None)
        if data_dir is None:
            data_dir = getattr(train_args, "data_dir_val", None)
        if data_dir is None:
            data_dir = getattr(train_args, "data_dir_train", None)
        if data_dir is None:
            logger.warning("No data directory found in args.pkl, skipping sample predictions")
            return []

        # Determine which element IDs to use
        # Filter to only elements whose prediction files were saved to disk
        if "saved" in element_df.columns:
            saved_df = element_df[element_df["saved"].eq(True)]
            if saved_df.empty:
                logger.warning("No prediction files saved to disk, skipping sample predictions")
                return []
            all_element_ids = saved_df["element_id"].tolist()
        else:
            all_element_ids = element_df["element_id"].tolist()

        if sample_ids:
            # Match provided sample_ids against available element_ids
            selected_ids = []
            for sid in sample_ids:
                # Normalize: add .tif if needed
                sid_tif = sid if sid.endswith(".tif") else f"{sid}.tif"
                if sid_tif in all_element_ids:
                    selected_ids.append(sid_tif)
                else:
                    logger.warning(f"Sample ID '{sid}' not found in element_metrics.csv, skipping")
            if not selected_ids:
                logger.warning("None of the specified sample IDs were found")
                return []
        else:
            # Random selection
            n = min(num_samples, len(all_element_ids))
            rng = np.random.default_rng(42)
            indices = rng.choice(len(all_element_ids), size=n, replace=False)
            selected_ids = [all_element_ids[i] for i in sorted(indices)]

        # Build band name lists
        band_names = get_sorted_band_names(use_s2hr, use_s2mr, use_s2lr, use_s1)

        display_channels, brightness_factor = get_display_config(use_s2hr, use_s2mr, use_s2lr, use_s1)

        # Determine class names for legend
        class_names_list = classnames()
        if no_savanna:
            class_names_list = [c for c in class_names_list if c != "Savanna"]

        # Determine if predictions were scored
        has_scores = "accuracy" in element_df.columns and "mIoU" in element_df.columns

        figures = []
        for element_id in selected_ids:
            try:
                fig = self._create_single_sample_figure(
                    element_id,
                    predictions_dir,
                    data_dir,
                    use_s2hr,
                    use_s2mr,
                    use_s2lr,
                    use_s1,
                    no_savanna,
                    band_names,
                    display_channels,
                    brightness_factor,
                    element_df,
                    has_scores,
                    border_exclusion_size=border_exclusion_size,
                )
                if fig is not None:
                    figures.append(fig)
            except Exception as e:
                logger.warning(f"Failed to create sample prediction for {element_id}: {e}")
                continue

        return figures

    def _create_single_sample_figure(
        self,
        element_id: str,
        predictions_dir: Path,
        data_dir: str,
        use_s2hr: bool,
        use_s2mr: bool,
        use_s2lr: bool,
        use_s1: bool,
        no_savanna: bool,
        band_names: list[str],
        display_channels: tuple[int, ...] | int,
        brightness_factor: int,
        element_df: pd.DataFrame,
        has_scores: bool,
        border_exclusion_size: int | None = None,
    ) -> plt.Figure | None:
        """Create a single sample prediction figure.

        Layout:
        - Row 1: RGB composite + individual bands (S2 + S1)
        - Row 2: Ground Truth (colorized) + Prediction (colorized)
        """
        # Load prediction TIFF
        pred_path = predictions_dir / element_id
        if not pred_path.exists():
            logger.warning(f"Prediction file not found: {pred_path}")
            return None

        pred_labels = load_prediction(pred_path)

        # Load bands and ground truth
        image_bands, gt_labels = load_bands_and_gt(
            element_id,
            data_dir,
            use_s2hr,
            use_s2mr,
            use_s2lr,
            use_s1,
            no_savanna,
        )

        # Build title
        base_name = element_id.replace(".tif", "")
        title = base_name
        if has_scores:
            row = element_df[element_df["element_id"] == element_id]
            if not row.empty:
                acc = row.iloc[0].get("accuracy", None)
                miou = row.iloc[0].get("mIoU", None)
                parts = []
                if acc is not None:
                    parts.append(f"Accuracy: {acc:.4f}")
                if miou is not None:
                    parts.append(f"mIoU: {miou:.4f}")
                if parts:
                    title += f"  ({', '.join(parts)})"

        return create_sample_figure(
            image_bands=image_bands,
            band_names=band_names,
            display_channels=display_channels,
            brightness_factor=brightness_factor,
            gt_labels=gt_labels,
            pred_labels_list=[pred_labels],
            pred_names=["Prediction"],
            no_savanna=no_savanna,
            title=title,
            border_exclusion_size=border_exclusion_size,
        )


class AnalysisCollector:
    def collect(self, analysis_dir: Path) -> dict[str, Any] | None:
        try:
            if not analysis_dir.exists():
                logger.warning(REPORTING_WARNINGS.PLOT_FILE_MISSING.format(path=analysis_dir))
                return None

            data: dict[str, Any] = {}

            # Collect existing plot files
            plot_files = list(analysis_dir.glob("*.png"))
            data["plot_files"] = [str(p) for p in plot_files]

            # Load element metrics for correlation and descriptive stats
            element_metrics_file = analysis_dir.parent / "predictions" / "element_metrics.csv"
            if element_metrics_file.exists():
                element_df = pd.read_csv(element_metrics_file)

                # Generate metrics correlation scatter plots
                fig = self._create_metrics_correlation_plot(element_df)
                if fig is not None:
                    data["metrics_correlation_plot"] = fig

                # Compute descriptive statistics
                stats = self._compute_descriptive_stats(element_df)
                if stats:
                    data["descriptive_stats"] = stats

            # Generate per-class accuracy bar chart from confusion matrix
            confusion_file = analysis_dir.parent / "predictions" / "confusion_matrix.csv"
            if confusion_file.exists():
                confusion_matrix = pd.read_csv(confusion_file, header=None).values
                class_names = _resolve_class_names(confusion_matrix.shape[0])
                fig = create_per_class_iou_plot(
                    confusion_matrix,
                    class_names,
                    figure_size=(ReportDefaults["FIGURE_WIDTH"], ReportDefaults["FIGURE_HEIGHT"]),
                )
                if fig is not None:
                    data["per_class_accuracy_plot"] = fig

            return data

        except Exception as e:
            logger.error(REPORTING_ERRORS.DATA_COLLECTION_FAILED.format(section="analysis", error=str(e)))
            return None

    def _create_metrics_correlation_plot(self, element_df: pd.DataFrame) -> plt.Figure | None:
        try:
            fig_h = ReportDefaults["FIGURE_HEIGHT"] // 2 + 2
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(ReportDefaults["FIGURE_WIDTH"], fig_h),
            )

            pairs = [("accuracy", "mIoU"), ("AA", "mean_f1")]
            for idx, (x_metric, y_metric) in enumerate(pairs):
                ax = axes[idx]
                if x_metric in element_df.columns and y_metric in element_df.columns:
                    ax.scatter(
                        element_df[x_metric],
                        element_df[y_metric],
                        alpha=0.5,
                        s=15,
                        edgecolors="none",
                    )
                    ax.set_xlabel(x_metric)
                    ax.set_ylabel(y_metric)
                    ax.set_title(f"{x_metric} vs {y_metric}")
                    ax.grid(True, alpha=0.3)

                    # Add correlation coefficient
                    corr = element_df[x_metric].corr(element_df[y_metric])
                    ax.annotate(
                        f"r = {corr:.3f}",
                        xy=(0.05, 0.95),
                        xycoords="axes fraction",
                        fontsize=10,
                        verticalalignment="top",
                        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
                    )
                else:
                    ax.text(0.5, 0.5, "Data not available", transform=ax.transAxes, ha="center")

            plt.tight_layout()
            return fig
        except Exception:
            return None

    def _compute_descriptive_stats(self, element_df: pd.DataFrame) -> list[dict[str, Any]]:
        """Compute descriptive statistics for key metrics."""
        metrics = ["accuracy", "AA", "mIoU", "mean_f1"]
        stats = []
        for metric in metrics:
            if metric not in element_df.columns:
                continue
            col = element_df[metric]
            stats.append(
                {
                    "Metric": metric,
                    "Mean": f"{col.mean():.4f}",
                    "Std": f"{col.std():.4f}",
                    "Min": f"{col.min():.4f}",
                    "Q1": f"{col.quantile(0.25):.4f}",
                    "Median": f"{col.median():.4f}",
                    "Q3": f"{col.quantile(0.75):.4f}",
                    "Max": f"{col.max():.4f}",
                }
            )
        return stats


class TestCollector:
    def collect(self, test_results_file: str | Path | None = None) -> dict[str, Any] | None:
        try:
            data: dict[str, Any] = {}

            # Check for test files existence
            tests_dir = Path("tests")
            if tests_dir.exists():
                test_files = list(tests_dir.glob("test_*.py"))
                data["test_files"] = [{"file": f.name, "exists": True} for f in test_files]

            # Parse pytest JSON report if available
            if test_results_file and Path(test_results_file).exists():
                with open(test_results_file) as f:
                    test_data = json.load(f)
                    data["test_summary"] = {
                        "total": test_data.get("total", 0),
                        "passed": test_data.get("passed", 0),
                        "failed": test_data.get("failed", 0),
                        "skipped": test_data.get("skipped", 0),
                        "duration": test_data.get("duration", 0),
                    }

            # Check for coverage data
            coverage_dir = Path("htmlcov")
            if coverage_dir.exists():
                index_file = coverage_dir / "index.html"
                if index_file.exists():
                    # Parse coverage percentage (simplified)
                    data["coverage_available"] = True

            return data

        except Exception as e:
            logger.error(REPORTING_ERRORS.DATA_COLLECTION_FAILED.format(section="tests", error=str(e)))
            return None
