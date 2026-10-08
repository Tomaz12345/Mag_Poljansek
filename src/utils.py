import logging
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib import colors

from .constants import DFC2020_SUBSETS, LOG_LEVELS, SPECIAL_VALUES
from .messages import UTILS_ERRORS

logger = logging.getLogger(__name__)

# Dict helpers


def make_validation_metrics(
    *,
    loss: float,
    AA: float,
    mIoU: float,
    accuracy: float,
    mean_f1: float,
) -> dict[str, float]:
    return {
        "loss": float(loss),
        "AA": float(AA),
        "mIoU": float(mIoU),
        "accuracy": float(accuracy),
        "mean_f1": float(mean_f1),
    }


def validation_metrics_items(metrics: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    return (
        ("loss", float(metrics.get("loss", 0.0))),
        ("AA", float(metrics.get("AA", 0.0))),
        ("mIoU", float(metrics.get("mIoU", 0.0))),
        ("accuracy", float(metrics.get("accuracy", 0.0))),
        ("mean_f1", float(metrics.get("mean_f1", 0.0))),
    )


def make_per_class_metrics(
    *,
    precision: list[float],
    recall: list[float],
    f1: list[float],
    iou: list[float],
) -> dict[str, list[float]]:
    return {
        "precision": list(precision),
        "recall": list(recall),
        "f1": list(f1),
        "iou": list(iou),
    }


def compute_per_class_metrics_from_cm(
    confusion_matrix: np.ndarray,
) -> dict[str, list[float]]:
    """Compute per-class precision, recall, F1, and IoU from a confusion matrix.

    Returns dict with keys 'precision', 'recall', 'f1', 'iou', each a list of floats.
    """
    num_classes = confusion_matrix.shape[0]

    precision_list: list[float] = []
    recall_list: list[float] = []
    f1_list: list[float] = []
    iou_list: list[float] = []

    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        precision_list.append(float(precision))
        recall_list.append(float(recall))
        f1_list.append(float(f1))
        iou_list.append(float(iou))

    return make_per_class_metrics(
        precision=precision_list,
        recall=recall_list,
        f1=f1_list,
        iou=iou_list,
    )


def compute_per_class_metrics(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str],
    *,
    format_values: bool = True,
) -> list[dict[str, Any]]:
    """Compute per-class precision, recall, F1, IoU from confusion matrix.

    Returns a list of dicts, one per class, with keys 'class', 'precision',
    'recall', 'f1', 'iou'. Values are formatted strings if format_values=True,
    otherwise raw floats.
    """
    num_classes = confusion_matrix.shape[0]
    per_class: list[dict[str, Any]] = []

    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        if format_values:
            per_class.append(
                {
                    "class": class_names[i],
                    "precision": f"{precision:.4f}",
                    "recall": f"{recall:.4f}",
                    "f1": f"{f1:.4f}",
                    "iou": f"{iou:.4f}",
                }
            )
        else:
            per_class.append(
                {
                    "class": class_names[i],
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "iou": float(iou),
                }
            )

    return per_class


def make_data_directories_info(
    *,
    data_dir_train: str | None = None,
    data_dir_val: str | None = None,
    dataset_val: str | None = None,
    no_savanna: bool = False,
    use_s2hr: bool = False,
    use_s2mr: bool = False,
    use_s2lr: bool = False,
    use_s1: bool = False,
) -> dict[str, Any]:
    return {
        "data_dir_train": data_dir_train,
        "data_dir_val": data_dir_val,
        "dataset_val": dataset_val,
        "no_savanna": bool(no_savanna),
        "use_s2hr": bool(use_s2hr),
        "use_s2mr": bool(use_s2mr),
        "use_s2lr": bool(use_s2lr),
        "use_s1": bool(use_s1),
    }


def make_model_config(
    *,
    model: Any = None,
    out_stride: Any = None,
    bilinear: Any = None,
    use_pretrained: Any = None,
    lr: Any = None,
    batch_size: Any = None,
    max_epochs: Any = None,
    num_steps: Any = None,
    momentum: Any = None,
    weight_decay: Any = None,
) -> dict[str, Any]:
    return {
        "model": model,
        "out_stride": out_stride,
        "bilinear": bilinear,
        "use_pretrained": use_pretrained,
        "lr": lr,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "num_steps": num_steps,
        "momentum": momentum,
        "weight_decay": weight_decay,
    }


def make_experiment_data(
    *,
    name: str,
    path: str,
    configuration: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    prediction: dict[str, Any] | None = None,
    data_directories: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    saved_elements: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "path": path,
        "configuration": configuration,
        "training": training,
        "prediction": prediction,
        "data_directories": data_directories if data_directories is not None else make_data_directories_info(),
        "model_config": model_config if model_config is not None else make_model_config(),
        "saved_elements": list(saved_elements) if saved_elements is not None else [],
    }


def make_validation_detail(
    *,
    valid: bool = True,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "valid": bool(valid),
        "warnings": list(warnings) if warnings is not None else [],
    }


def make_validation_report(
    *,
    is_valid: bool = True,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    details: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "is_valid": bool(is_valid),
        "warnings": list(warnings) if warnings is not None else [],
        "errors": list(errors) if errors is not None else [],
        "details": dict(details) if details is not None else {},
    }


def make_report_config(
    *,
    source: str = "",
    args_file_path: str = "",
    experiment_dir: str = "",
    stages_run: list[str] | None = None,
    formats: list[str] | None = None,
    report_num_samples: int | None = None,
    report_sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "args_file_path": args_file_path,
        "experiment_dir": experiment_dir,
        "stages_run": list(stages_run) if stages_run is not None else [],
        "formats": list(formats) if formats is not None else ["pdf"],
        "report_num_samples": report_num_samples,
        "report_sample_ids": list(report_sample_ids) if report_sample_ids is not None else None,
    }


def make_experiment_entry(
    *,
    name: str = "",
    path: str = "",
    status: str = "",
    has_checkpoints: bool = False,
    has_predictions: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "path": path,
        "status": status,
        "has_checkpoints": bool(has_checkpoints),
        "has_predictions": bool(has_predictions),
    }


def make_experiment_summary(
    *,
    status: str = "loading",
    config: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    predictions: dict[str, Any] | None = None,
    tensorboard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "config": config,
        "training": training,
        "predictions": predictions,
        "tensorboard": tensorboard,
    }


def make_training_state(
    *,
    model_state: Mapping[str, Any],
    step: int | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
    rng_state_torch: Any | None = None,
    rng_state_numpy: tuple[Any, ...] | None = None,
    rng_state_python: object | None = None,
    rng_state_cuda: list[Any] | None = None,
    rng_state_generator: object | None = None,
    sampler_state: object | None = None,
) -> dict[str, Any]:
    return {
        "model_state_dict": model_state,
        "step": step,
        "optimizer_state_dict": optimizer_state,
        "rng_state_torch": rng_state_torch,
        "rng_state_numpy": rng_state_numpy,
        "rng_state_python": rng_state_python,
        "rng_state_cuda": rng_state_cuda,
        "rng_state_generator": rng_state_generator,
        "sampler_state": sampler_state,
    }


def training_state_items(state: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    items: list[tuple[str, Any]] = [("model_state_dict", state["model_state_dict"])]
    optional_keys = [
        "step",
        "optimizer_state_dict",
        "rng_state_torch",
        "rng_state_numpy",
        "rng_state_python",
        "rng_state_cuda",
        "rng_state_generator",
        "sampler_state",
    ]
    for key in optional_keys:
        value = state.get(key)
        if value is not None:
            items.append((key, value))
    return tuple(items)


def make_training_result(
    *,
    log_dir: str,
    experiment_name: str,
    checkpoint_dir: str,
    completed: bool,
    step: int | None = None,
    final_step: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "log_dir": log_dir,
        "experiment_name": experiment_name,
        "checkpoint_dir": checkpoint_dir,
        "completed": completed,
    }
    if step is not None:
        result["step"] = step
    if final_step is not None:
        result["final_step"] = final_step
    return result


# Error context builder and error logging


class ErrorContext:
    """Builder class for constructing error context information.

    Provides a fluent interface for assembling contextual information
    about errors, with automatic formatting of complex objects like
    arrays, tensors, and paths.

    Example:
        Basic usage:
        >>> context = ErrorContext().add("shape", tensor.shape).add("dtype", tensor.dtype)
        >>> log_and_raise(ValueError, "Invalid tensor", context=context)

        With convenience methods:
        >>> context = ErrorContext().add_tensor_info("input", tensor).add_path_info("file", path)
        >>> log_and_raise(FileNotFoundError, msg, context=context)

        Multiple additions at once:
        >>> context = ErrorContext().add_many(shape=(256, 256), dtype="float32", device="cuda")
        >>> log_and_raise(ValueError, msg, context=context)
    """

    def __init__(self) -> None:
        """Initialize an empty error context."""
        self._context: dict[str, Any] = {}

    def add(self, key: str, value: Any) -> "ErrorContext":
        """Add a key-value pair to the context.

        Args:
            key: The context key/variable name
            value: The value to store.

        Returns:
            Self for method chaining
        """
        self._context[key] = value
        return self

    def add_many(self, **kwargs: Any) -> "ErrorContext":
        """Add multiple key-value pairs at once.

        Example:
            >>> ctx.add_many(batch_size=32, num_classes=10, lr=0.001)
        """
        self._context.update(kwargs)
        return self

    def add_tensor_info(self, name: str, tensor: Any) -> "ErrorContext":
        """Add a tensor's shape, dtype, device and min/max/mean to the context.

        Args:
            name: Base name for the tensor context keys
            tensor: Tensor or array object (numpy.ndarray, torch.Tensor, etc.)

        Returns:
            Self for method chaining

        Example:
            >>> ctx.add_tensor_info("input_tensor", my_tensor)
            # Adds: input_tensor_shape, input_tensor_dtype, input_tensor_device
        """
        if hasattr(tensor, "shape"):
            self.add(f"{name}_shape", tuple(tensor.shape))
        if hasattr(tensor, "dtype"):
            self.add(f"{name}_dtype", str(tensor.dtype))
        if hasattr(tensor, "device"):
            self.add(f"{name}_device", str(tensor.device))
        if isinstance(tensor, (np.ndarray, torch.Tensor)):
            try:
                tensor_np = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
                if tensor_np.size > 0:  # Only if non-empty
                    self.add(f"{name}_min", float(tensor_np.min()))
                    self.add(f"{name}_max", float(tensor_np.max()))
                    self.add(f"{name}_mean", float(tensor_np.mean()))
            except (RuntimeError, ValueError):
                pass  # Skip statistics if conversion fails
        return self

    def add_path_info(self, name: str, path: str | Path) -> "ErrorContext":
        """Add path information including existence check.

        Args:
            name: Base name for the path context keys
            path: File or directory path

        Returns:
            Self for method chaining

        Example:
            >>> ctx.add_path_info("config_file", "/path/to/config.txt")
            # Adds: config_file, config_file_exists
        """
        self.add(name, str(path))
        try:
            self.add(f"{name}_exists", Path(path).exists())
        except (OSError, ValueError):
            pass  # Skip existence check if path is invalid
        return self

    def to_dict(self) -> dict[str, Any]:
        """Get the underlying dictionary representation."""
        return self._context

    def __bool__(self) -> bool:
        """Check if context is non-empty."""
        return bool(self._context)

    def __repr__(self) -> str:
        return f"ErrorContext({self._context})"


def _format_context_value(value: Any) -> str:
    """Format a context value for logging, with intelligent handling of complex types."""
    # Handle None
    if value is None:
        return "None"

    # Handle numpy arrays
    if isinstance(value, np.ndarray):
        info_parts = [
            f"shape={tuple(value.shape)}",
            f"dtype={value.dtype}",
        ]
        if value.size > 0:
            try:
                info_parts.extend(
                    [
                        f"min={value.min():.4g}",
                        f"max={value.max():.4g}",
                        f"mean={value.mean():.4g}",
                    ]
                )
            except (TypeError, ValueError):
                pass  # Skip stats for non-numeric arrays
        return f"ndarray({', '.join(info_parts)})"

    # Handle torch tensors
    if isinstance(value, torch.Tensor):
        info_parts = [
            f"shape={tuple(value.shape)}",
            f"dtype={value.dtype}",
            f"device={value.device}",
        ]
        if value.numel() > 0:
            try:
                info_parts.extend(
                    [
                        f"min={value.min().item():.4g}",
                        f"max={value.max().item():.4g}",
                        f"mean={value.float().mean().item():.4g}",
                    ]
                )
            except (RuntimeError, ValueError):
                pass  # Skip stats if operation not supported
        return f"Tensor({', '.join(info_parts)})"

    # Handle long strings
    if isinstance(value, str):
        max_len = 100
        if len(value) > max_len:
            return f"'{value[:max_len]}...' ({len(value)} chars total)"
        return f"'{value}'"

    # Handle large lists
    if isinstance(value, list):
        max_items = 10
        if len(value) > max_items:
            sample = ", ".join(repr(x) for x in value[:max_items])
            return f"list[{len(value)} items]: [{sample}, ...]"
        return repr(value)

    # Handle large dicts
    if isinstance(value, dict):
        max_items = 10
        if len(value) > max_items:
            sample_items = list(value.items())[:max_items]
            sample = ", ".join(f"{k!r}: {v!r}" for k, v in sample_items)
            return f"dict[{len(value)} items]: {{{sample}, ...}}"
        return repr(value)

    # Handle paths
    if isinstance(value, Path):
        return f"Path('{value}')"

    # Handle tuples (common for shapes)
    if isinstance(value, tuple):
        return repr(value)

    # Default: use repr
    return repr(value)


def log_and_raise(
    error_type: type[Exception],
    message: str,
    *,
    logger_instance: logging.Logger | None = None,
    context: ErrorContext | dict[str, Any] | None = None,
    include_traceback: bool = True,
) -> NoReturn:
    """Log an error message with context and raise the specified exception.

    This function combines error logging and exception raising into a single
    operation, with enhanced context information for debugging. It automatically
    formats complex objects (arrays, tensors, paths) in a readable way.

    Args:
        error_type: The exception class to raise (e.g., ValueError, FileNotFoundError)
        message: The error message to log and include in the exception
        logger_instance: Logger instance to use (defaults to module-level logger)
        context: ErrorContext builder or dict of variable names to values for debugging.
                Large objects are automatically summarized.
        include_traceback: Whether to include current stack trace in the log output.
                          Defaults to True for better debugging.

    Raises:
        The specified exception type with the given message

    Examples:
        Simple usage without context:
        >>> log_and_raise(ValueError, "Invalid input value")

        With context builder (recommended for complex cases):
        >>> ctx = ErrorContext().add("shape", tensor.shape).add("dtype", tensor.dtype)
        >>> log_and_raise(ValueError, "Invalid tensor", context=ctx)

        With convenience methods:
        >>> ctx = ErrorContext().add_tensor_info("input", tensor)
        >>> log_and_raise(ValueError, msg, context=ctx)

        With inline dict (simple cases):
        >>> log_and_raise(ValueError, msg, context={"num_classes": num_classes})

        With custom logger:
        >>> my_logger = logging.getLogger("my_module")
        >>> log_and_raise(FileNotFoundError, msg, logger_instance=my_logger)
    """
    # Use provided logger or default to module logger
    log = logger_instance if logger_instance is not None else logger

    # Build the error log message
    log_message_parts = [message]

    # Add context information if provided
    if context:
        # Convert ErrorContext to dict if needed
        context_dict = context.to_dict() if isinstance(context, ErrorContext) else context

        if context_dict:
            log_message_parts.append("\n  Context:")
            for key, value in context_dict.items():
                formatted_value = _format_context_value(value)
                log_message_parts.append(f"    {key}: {formatted_value}")

    # Add traceback if requested
    if include_traceback:
        # Get the current stack (excluding this function)
        stack_summary = traceback.extract_stack()[:-1]
        # Format just the last few frames for conciseness
        relevant_frames = stack_summary[-5:] if len(stack_summary) > 5 else stack_summary
        tb_lines = traceback.format_list(relevant_frames)
        if tb_lines:
            log_message_parts.append("\n  Traceback (most recent calls):")
            for line in tb_lines:
                # Indent each line of the traceback
                indented = "    " + line.rstrip().replace("\n", "\n    ")
                log_message_parts.append(indented)

    # Log the complete error message
    full_log_message = "".join(log_message_parts)
    log.error(full_log_message)

    # Raise the exception with the original message (without context/traceback)
    raise error_type(message)


def validate_string_value(value: str, valid_values: tuple[str, ...], name: str) -> str:
    """Validate a string against allowed values and return the normalized match.

    Args:
        value: Input string to validate.
        valid_values: Tuple of allowed lowercase values.
        name: Human-readable parameter name for error messages.

    Returns:
        The normalized (lowercased) value if valid.

    Raises:
        ValueError: If the value is not in valid_values.
    """
    normalized = value.lower()
    if normalized in valid_values:
        return normalized
    valid = ", ".join(valid_values)
    msg = f"Invalid {name}: '{value}'. Must be one of: {valid}"
    log_and_raise(ValueError, msg, logger_instance=logger)


def validate_log_level(value: str) -> str:
    """Validate a log level string and return the normalized (uppercase) value.

    Args:
        value: Input log level string.

    Returns:
        The normalized (uppercased) log level if valid.

    Raises:
        ValueError: If the value is not a valid log level.
    """
    normalized = value.upper()
    if normalized in LOG_LEVELS:
        return normalized
    valid = ", ".join(LOG_LEVELS)
    msg = f"Invalid log level: '{value}'. Must be one of: {valid}"
    log_and_raise(ValueError, msg, logger_instance=logger)


def is_dfc2020_dataset(dataset_type: str) -> bool:
    """Check if a dataset type string refers to a DFC2020 dataset."""
    return dataset_type in ("dfc2020_val", "dfc2020_test")


def get_dfc2020_subset(dataset_type: str) -> str:
    """Get the DFC2020 subset string ('val' or 'test') from a dataset type string.

    Args:
        dataset_type: Dataset type string (e.g. 'dfc2020_val').

    Returns:
        The subset string ('val' or 'test').

    Raises:
        ValueError: If the dataset_type is not a DFC2020 dataset.
    """
    if dataset_type == "dfc2020_val":
        return DFC2020_SUBSETS[0]  # "val"
    if dataset_type == "dfc2020_test":
        return DFC2020_SUBSETS[1]  # "test"
    msg = f"Not a DFC2020 dataset: '{dataset_type}'"
    log_and_raise(ValueError, msg, logger_instance=logger)


def convert_to_np(tensor: Any) -> np.ndarray:
    """Convert pytorch tensors or numpy arrays to numpy arrays.

    Raises:
        ValueError: If input is neither a torch.Tensor nor numpy.ndarray.
    """
    if isinstance(tensor, np.ndarray):
        return cast(np.ndarray, tensor)
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    # Check for tensor-like objects with cpu() method (e.g., custom tensors)
    if hasattr(tensor, "cpu") and hasattr(tensor, "numpy"):
        return cast(np.ndarray, tensor.cpu().numpy())
    msg = f"Expected torch.Tensor or numpy.ndarray, got {type(tensor).__name__}. Provide a valid tensor or array for conversion."
    log_and_raise(ValueError, msg, context={"tensor_type": type(tensor).__name__})


def labels_to_dfc(tensor: Any, no_savanna: bool) -> np.ndarray:
    """
    INPUT:
    Classes encoded in the training scheme (0-9 if savanna is a valid label
    or 0-8 if not). Invalid labels are marked by 255 and will not be changed.

    OUTPUT:
    Classes encoded in the DFC2020 scheme (1-10, and 255 for invalid).
    """

    # transform to numpy array
    tensor = convert_to_np(tensor)

    # copy the original input
    out = np.copy(tensor)

    # shift labels if there is no savanna class (vectorized)
    if no_savanna:
        # Vectorized: shift classes 2-8 to 3-9 using boolean mask
        # Classes >= 2 and < 9 and not ignore (255) need to be shifted by 1
        shift_mask = (tensor >= 2) & (tensor < 9) & (tensor != SPECIAL_VALUES["IGNORE_LABEL"])
        out[shift_mask] = tensor[shift_mask] + 1

    # transform from zero-based labels to 1-10
    out[tensor != SPECIAL_VALUES["IGNORE_LABEL"]] += 1

    # make sure the mask is intact and return transformed labels
    if not np.all((tensor == SPECIAL_VALUES["IGNORE_LABEL"]) == (out == SPECIAL_VALUES["IGNORE_LABEL"])):
        log_and_raise(ValueError, "Label transformation corrupted the ignore mask")
    return out


def display_input_batch(
    tensor: torch.Tensor,
    display_indices: int | Sequence[int] = 0,
    brightness_factor: int = 3,
) -> torch.Tensor:
    """Extract and scale display channels from input tensor.

    Args:
        tensor: Input tensor of shape (N, C, H, W).
        display_indices: Channel index or indices to display.
        brightness_factor: Multiplier for brightness scaling (must be positive).

    Returns:
        Tensor with selected channels, scaled for display.

    Raises:
        ValueError: If tensor is not 4D, indices are invalid, or brightness_factor <= 0.
    """
    # Validate input tensor shape
    if tensor.ndim != 4:
        msg = f"Expected 4D input tensor (N, C, H, W), got {tensor.ndim}D tensor with shape {tuple(tensor.shape)}"
        ctx = ErrorContext().add_tensor_info("input_tensor", tensor)
        log_and_raise(ValueError, msg, context=ctx)

    # Validate brightness factor
    if brightness_factor <= 0:
        msg = f"brightness_factor must be a positive number, got {brightness_factor}"
        log_and_raise(ValueError, msg, context={"brightness_factor": brightness_factor})

    # Validate display_indices
    num_channels = tensor.shape[1]
    if isinstance(display_indices, int):
        if display_indices < 0 or display_indices >= num_channels:
            msg = UTILS_ERRORS.CHANNEL_INDEX_OUT_OF_RANGE.format(index=display_indices, channels=num_channels, max_index=num_channels - 1)
            ctx = ErrorContext().add_many(index=display_indices, num_channels=num_channels, tensor_shape=tensor.shape)
            log_and_raise(ValueError, msg, context=ctx)
    else:
        # display_indices is Sequence[int]
        for idx in display_indices:
            if idx < 0 or idx >= num_channels:
                msg = UTILS_ERRORS.CHANNEL_INDEX_OUT_OF_RANGE.format(index=idx, channels=num_channels, max_index=num_channels - 1)
                ctx = ErrorContext().add_many(
                    index=idx,
                    display_indices=list(display_indices),
                    num_channels=num_channels,
                    tensor_shape=tensor.shape,
                )
                log_and_raise(ValueError, msg, context=ctx)

    # extract display channels
    tensor = tensor[:, display_indices, :, :]

    # restore NCHW tensor shape if single channel image
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)

    # scale image
    tensor = torch.clamp((tensor * brightness_factor), 0, 1)

    return tensor


def display_label_batch(tensor: Any, no_savanna: bool = False) -> np.ndarray:
    # get predictions if input is one-hot encoded
    if len(tensor.shape) == 4:
        tensor = tensor.max(1)[1]

    # convert train labels to DFC2020 class scheme
    tensor = labels_to_dfc(tensor, no_savanna)

    # Vectorized colormap application for entire batch
    cmap = mycmap()
    # Normalize: (tensor - 1) / 10 for all elements at once
    normalized = (tensor - 1) / 10
    # Apply colormap to entire batch (returns shape: batch, H, W, 4)
    colored = cmap(normalized)[:, :, :, 0:3]  # Take RGB channels
    # Transpose from (batch, H, W, C) to (batch, C, H, W)
    result = np.transpose(colored, (0, 3, 1, 2))

    return result


def classnames() -> list[str]:
    return [
        "Forest",
        "Shrubland",
        "Savanna",
        "Grassland",
        "Wetlands",
        "Croplands",
        "Urban/Built-up",
        "Snow/Ice",
        "Barren",
        "Water",
    ]


def mycmap() -> colors.ListedColormap:
    cmap = colors.ListedColormap(
        [
            "#009900",
            "#c6b044",
            "#fbff13",
            "#b6ff05",
            "#27ff87",
            "#c24f44",
            "#a5a5a5",
            "#69fff8",
            "#f9ffa4",
            "#1c0dff",
        ]
    )
    return cmap


def mypatches() -> list[mpatches.Patch]:
    patches: list[mpatches.Patch] = []
    colors_list = list(cast(list[str], mycmap().colors))
    for counter, name in enumerate(classnames()):
        color: str = colors_list[counter]
        patches.append(mpatches.Patch(color=color, label=name))
    return patches
