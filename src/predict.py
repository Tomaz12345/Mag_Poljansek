import argparse
import io
import logging
import os
import pickle as pkl
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from . import metrics
from .config import (
    load_args_from_file,
    validate_predict_args,
)
from .constants import DATASET_TYPES, SPECIAL_VALUES, PredictArgDefaults
from .datasets import BaseDataset, build_inference_dataset, mask_label_borders
from .logging_utils import log_args_config, setup_logging
from .model_factory import create_model
from .result_saver import PredictionSaver, select_random_subset
from .timing_utils import (
    Timer,
    get_aggregator,
    load_timing_config_from_args,
    reset_aggregator,
    set_timing_args,
)
from .utils import is_dfc2020_dataset, labels_to_dfc, log_and_raise, validate_string_value
from .utils import mycmap as dfc_cmap
from .utils import mypatches as dfc_legend

logger = logging.getLogger(__name__)


def run_prediction(
    args_file_path: str,
    args_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run prediction on a dataset.

    Args:
        args_file_path: Path to combined arguments file (common + model-specific)
        args_override: Optional dict to override specific arguments

    Returns:
        Dictionary containing prediction results and metrics
    """
    # Load all arguments (common + model-specific) from a single file
    args = load_args_from_file(
        args_file_path,
        PredictArgDefaults.copy(),
        use_model_defaults=True,
        resolve_paths=("config_file", "checkpoint_file", "out_dir"),
    )

    # Apply overrides if provided
    if args_override:
        for key, value in args_override.items():
            setattr(args, key, value)

    # Validate all arguments early with clear error messages
    validate_predict_args(args)

    # Make args.enable_timing available to timing decorators
    set_timing_args(args)
    load_timing_config_from_args(args)
    reset_aggregator()  # Reset aggregator for fresh run

    # load config
    with open(args.config_file, "rb") as f:
        train_args = pkl.load(f)

    return _run_prediction_with_args(args, train_args)


def _run_prediction_with_args(args: argparse.Namespace, train_args: argparse.Namespace) -> dict[str, Any]:
    """Internal function to run prediction with loaded arguments.

    Args:
        args: Prediction arguments
        train_args: Training arguments loaded from checkpoint

    Returns:
        Dictionary containing prediction results and metrics
    """
    # configure logging (use log_level from args or train_args)
    log_level = getattr(args, "log_level", PredictArgDefaults["log_level"])
    log_dir: str | None = getattr(train_args, "log_dir", None)
    experiment = getattr(train_args, "experiment_name", None)

    if log_dir is None:
        log_and_raise(ValueError, "log_dir must be specified in args!", logger_instance=logger)

    if experiment is None:
        log_and_raise(ValueError, "experiment_name must be specified in args!", logger_instance=logger)

    log_dir_path = os.path.join(log_dir, experiment)
    log_file_path = os.path.join(log_dir_path, "logs.log")

    setup_logging(str(log_level), log_file_path)

    log_args_config(args, logger=logger, header="Config", blank_line=False)
    log_args_config(train_args, logger=logger, header="Train config", blank_line=True)

    # create output dir
    os.makedirs(args.out_dir, exist_ok=True)

    # create preview dir
    if args.preview_dir is not None:
        os.makedirs(args.preview_dir, exist_ok=True)

    # set flags for GPU processing if available
    if torch.cuda.is_available():
        args.use_gpu = True
        if torch.cuda.device_count() > 1:
            msg = "multi-gpu prediction not implemented! try to run script as: CUDA_VISIBLE_DEVICES=0 predict.py"
            log_and_raise(NotImplementedError, msg, logger_instance=logger)
    else:
        args.use_gpu = False

    dataset_type = validate_string_value(args.dataset, DATASET_TYPES, "dataset type")

    dataset: BaseDataset
    gt_id: str

    if dataset_type == "tiff_dir" and args.score:
        log_and_raise(ValueError, "Cannot score predictions on unlabeled 'tiff_dir' dataset", logger_instance=logger)

    dataset = build_inference_dataset(
        dataset_type,
        args.data_dir,
        no_savanna=train_args.no_savanna,
        use_s2hr=train_args.use_s2hr,
        use_s2mr=train_args.use_s2mr,
        use_s2lr=train_args.use_s2lr,
        use_s1=train_args.use_s1,
        normalization_mode=getattr(train_args, "normalization_mode", "global"),
    )

    if dataset_type == "sen12ms_holdout":
        gt_id = "lc"
    elif dataset_type == "tiff_dir":
        gt_id = "pred"
    elif is_dfc2020_dataset(dataset_type):
        gt_id = "dfc"
    else:
        log_and_raise(ValueError, f"unknown dataset: {args.dataset}", logger_instance=logger)

    n_classes = dataset.n_classes
    n_inputs = dataset.n_inputs

    # Select subset of elements to predict (skip inference for the rest)
    total_elements = len(dataset)
    predict_indices = select_random_subset(total_elements, args.predict_subset_ratio, args.seed)
    subset_dataset: BaseDataset | Subset[Any] = dataset
    if args.predict_subset_ratio < 1.0:
        sorted_indices = sorted(predict_indices)
        subset_dataset = Subset(dataset, sorted_indices)
        logger.info(f"Predicting subset: {len(sorted_indices)} out of {total_elements} elements ({args.predict_subset_ratio * 100:.1f}%)")
    else:
        logger.info(f"Predicting all {total_elements} elements")

    # set up dataloader
    dataloader = DataLoader(
        subset_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    # Set up network
    with Timer("prediction/model_init", logger, component="prediction"):
        model = create_model(train_args, n_classes, n_inputs, inference=True)

        if args.use_gpu:
            model = model.cuda()

    # restore network weights
    with Timer("prediction/checkpoint_load", logger, component="prediction"):
        state = torch.load(args.checkpoint_file, weights_only=False)
        step = state["step"]
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        msg = f"loaded checkpoint from step {step}"
        logger.info(msg)

    # initialize scoring if ground-truth is available
    conf_mat: metrics.ConfMatrix | None = None
    if args.score:
        conf_mat = metrics.ConfMatrix(n_classes)

    # Initialize prediction saver
    prediction_saver = PredictionSaver(args.out_dir)

    # Cache colormap once — it's constant across all samples
    cmap = dfc_cmap() if args.preview_dir is not None else None
    legend_handles = dfc_legend() if args.preview_dir is not None else None
    display_channels = dataset.display_channels
    brightness_factor = dataset.brightness_factor
    show_border = args.score and getattr(train_args, "border_exclusion", False)

    # Reusable ConfMatrix for per-element metrics (reset state each sample)
    element_conf_mat = metrics.ConfMatrix(n_classes) if args.score else None

    def _save_tiff(path: str, arr: np.ndarray) -> None:
        Image.fromarray(arr).save(path)

    def _write_bytes(path: str, buf: bytes) -> None:
        with open(path, "wb") as f:
            f.write(buf)

    def _render_preview(
        img_np: np.ndarray,
        output_colored: np.ndarray,
        gt_colored: np.ndarray | None,
        gt_border_colored: np.ndarray | None,
        sample_id: str,
        border_size: int,
    ) -> bytes:
        """Render preview figure to PNG bytes on the calling thread, then close the figure."""
        n_panels = 2 + (1 if gt_colored is not None else 0) + (1 if gt_border_colored is not None else 0)
        fig, axes = plt.subplots(1, n_panels)
        ax_idx = 0
        axes[ax_idx].imshow(img_np)
        axes[ax_idx].set_title("input")
        axes[ax_idx].axis("off")
        ax_idx += 1
        axes[ax_idx].imshow(output_colored)
        axes[ax_idx].set_title("prediction")
        axes[ax_idx].axis("off")
        ax_idx += 1
        if gt_colored is not None:
            axes[ax_idx].imshow(gt_colored)
            axes[ax_idx].set_title("label")
            axes[ax_idx].axis("off")
            ax_idx += 1
        if gt_border_colored is not None:
            axes[ax_idx].imshow(gt_border_colored)
            axes[ax_idx].set_title(f"border excl. (r={border_size})")
            axes[ax_idx].axis("off")
        lgd = plt.legend(
            loc="lower left",
            bbox_to_anchor=(1.1, 0),
            handles=legend_handles,
            ncol=2,
            title="DFC Classes",
        )
        ttl = fig.suptitle(sample_id, y=0.75)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_extra_artists=(lgd, ttl), bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    # predict samples
    n = 0
    # Wrap entire prediction loop in timer
    with Timer("prediction/total", logger, component="prediction"), ThreadPoolExecutor() as io_pool:
        for _i_batch, batch in enumerate(tqdm(dataloader, desc="[Pred]")):
            # unpack sample
            image = batch["image"]
            if args.score:
                target = batch["label"]

            # move data to gpu if model is on gpu
            if args.use_gpu:
                image = image.cuda()
                if args.score:
                    target = target.cuda()

            with torch.inference_mode():
                prediction = model(image)
            prediction = prediction.cpu().numpy()
            prediction = np.argmax(prediction, axis=1)

            if args.score:
                target = target.cpu().numpy()

            # Convert image to numpy once per batch (avoids repeated GPU→CPU transfers)
            image_np = image.cpu().numpy() if args.preview_dir is not None else None

            # Vectorized: convert entire batch of predictions to DFC labels at once
            output_batch = labels_to_dfc(prediction, train_args.no_savanna)

            # Add entire batch to confusion matrix at once (vectorized)
            if args.score and conf_mat is not None:
                conf_mat.add_batch(target, prediction)

            # save predictions (element-wise operations still needed for I/O)
            io_futures = []
            for i in range(prediction.shape[0]):
                n += 1
                id = batch["id"][i].replace("_s2_", "_" + gt_id + "_")

                output = output_batch[i, :, :]

                # Save prediction (offloaded to thread pool)
                output_uint8 = output.astype(np.uint8)
                io_futures.append(io_pool.submit(_save_tiff, os.path.join(args.out_dir, id), output_uint8))

                # Compute per-element metrics if scoring
                element_metrics: dict[str, float] = {}
                if args.score and element_conf_mat is not None:
                    gt = labels_to_dfc(target[i, :, :], train_args.no_savanna)

                    # Reuse element_conf_mat — reset state instead of allocating new object
                    element_conf_mat.state[:] = 0
                    element_conf_mat.add(target[i, :, :], prediction[i, :, :])

                    element_metrics = {
                        "AA": element_conf_mat.get_aa(),
                        "mIoU": element_conf_mat.get_mIoU(),
                        "accuracy": element_conf_mat.get_accuracy(),
                        "mean_f1": element_conf_mat.get_mean_f1(),
                    }

                # Log element metrics
                prediction_saver.log_element(id, element_metrics, save_prediction=True)

                # save preview
                if args.preview_dir is not None and cmap is not None and image_np is not None:
                    output_colored = cmap((output - 1) / 10)[:, :, 0:3]
                    gt_colored_i = None
                    gt_border_colored_i = None
                    computed_border_size = 0
                    if args.score:
                        gt_colored_i = cmap((gt - 1) / 10)[:, :, 0:3]
                    if show_border and gt_colored_i is not None:
                        ignore_val = SPECIAL_VALUES["IGNORE_LABEL"]
                        computed_border_size = getattr(train_args, "border_exclusion_size", 10)
                        gt_zero = gt.copy()
                        gt_zero[gt != ignore_val] = gt[gt != ignore_val] - 1
                        masked_zero = mask_label_borders(gt_zero, computed_border_size, ignore_val)
                        border_mask = (masked_zero == ignore_val) & (gt_zero != ignore_val)
                        gt_border_colored_i = gt_colored_i.copy()
                        gt_border_colored_i[border_mask] = [0.0, 0.0, 0.0]

                    # Prepare display image (numpy already on CPU)
                    img = image_np[i, display_channels, :, :]
                    img = np.rollaxis(img, 0, 3)
                    img_display = np.clip(img * brightness_factor, 0, 1)

                    # Render figure on main thread (matplotlib is not thread-safe),
                    # then offload only the disk write to the thread pool.
                    png_bytes = _render_preview(
                        img_display,
                        output_colored,
                        gt_colored_i,
                        gt_border_colored_i,
                        id,
                        computed_border_size,
                    )
                    io_futures.append(io_pool.submit(_write_bytes, os.path.join(args.preview_dir, id), png_bytes))

            # Wait for all I/O futures and propagate any exceptions
            for fut in as_completed(io_futures):
                fut.result()

            batch_num = n // args.batch_size
            if batch_num % args.log_batch_freq == 0:
                if args.score and conf_mat is not None:
                    aa = conf_mat.get_aa() * 100
                    miou = conf_mat.get_mIoU() * 100
                    msg = f"Batch {batch_num} — AA: {aa:.4f}%, mIoU: {miou:.4f}%"
                else:
                    msg = f"Batch {batch_num}"
                logger.info(msg)

    # print scoring results
    results: dict[str, Any] = {
        "out_dir": args.out_dir,
        "n_predictions": n,
        "scored": args.score,
    }

    if args.score and conf_mat is not None:
        msg = f"AA\t {conf_mat.get_aa()}"
        logger.info(msg)

        msg = f"mIoU\t {conf_mat.get_mIoU()}"
        logger.info(msg)

        # Save confusion matrix and summary metrics
        all_metrics = conf_mat.to_dict()

        prediction_saver.save_confusion_matrix(conf_mat.state)
        prediction_saver.save_summary_metrics(all_metrics)
        logger.info("Confusion matrix and summary metrics saved")

        results["metrics"] = all_metrics

    # Save per-element metrics
    prediction_saver.save_element_metrics()
    logger.info("Per-element metrics saved")

    # Log timing summary if timing was enabled
    if args.enable_timing:
        aggregator = get_aggregator()
        aggregator.log_summary(logger, prefix="Prediction")

    return results


def main() -> dict[str, Any]:
    """Main entry point for command-line usage."""
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--args_file_path",
        type=str,
        required=True,
        help="path to a combined args file (key=value per line) containing both common and model-specific parameters",
    )

    # parse only the path, then load rest from file
    parsed = parser.parse_args()

    return run_prediction(parsed.args_file_path)


if __name__ == "__main__":
    main()
