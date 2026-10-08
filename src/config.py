import argparse
import logging
import os
from typing import Any, NamedTuple, cast

from .constants import (
    CONFIG_PARAMETER_GROUPS,
    DATASET_TYPES,
    GRADUAL_UNFREEZE_UNITS,
    LOSS_TYPES,
    MODEL_TYPES,
    TRAIN_DATASET_TYPES,
    AnalysisArgDefaults,
    PlotArgDefaults,
)
from .messages import CONFIG_ERRORS, CONFIG_WARNINGS
from .model_factory import MODEL_REGISTRY
from .utils import log_and_raise, validate_log_level, validate_string_value

logger = logging.getLogger(__name__)


class NumericConstraint(NamedTuple):
    """A single numeric validation constraint.

    Attributes:
        attr: Attribute name on the args namespace.
        kind: Constraint type: "positive", "non_negative", or "range".
        error_msg: Error message template string (from messages.py).
        format_kwarg: Keyword for str.format(). Defaults to attr if None.
        check_hasattr: Skip silently when attr doesn't exist on args.
        check_not_none: Skip silently when attr value is None.
        range_min: Minimum value for "range" constraints.
        range_max: Maximum value for "range" constraints.
    """

    attr: str
    kind: str
    error_msg: str
    format_kwarg: str | None = None
    check_hasattr: bool = False
    check_not_none: bool = False
    range_min: float = 0.0
    range_max: float = 1.0


def _apply_numeric_constraints(
    args: argparse.Namespace,
    constraints: tuple[NumericConstraint, ...],
) -> None:
    """Apply numeric constraints to args, raising on the first violation."""
    for c in constraints:
        if c.check_hasattr and not hasattr(args, c.attr):
            continue

        value = getattr(args, c.attr)

        if c.check_not_none and value is None:
            continue

        # Coerce string values to numeric (handles defaults=None cases where
        # cast_value cannot determine the target type).
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                value = float(value)
            setattr(args, c.attr, value)

        kwarg_name = c.format_kwarg if c.format_kwarg is not None else c.attr

        violated = False
        if c.kind == "positive":
            violated = value <= 0
        elif c.kind == "non_negative":
            violated = value < 0
        elif c.kind == "range":
            violated = value < c.range_min or value > c.range_max

        if violated:
            msg = c.error_msg.format(**{kwarg_name: value})
            log_and_raise(ValueError, msg, logger_instance=logger)


def get_default_value(defaults: dict[str, Any], key: str) -> Any:
    return defaults[key]


def cast_value(key: str, sval: str, defaults: dict[str, Any]) -> Any:
    """
    Cast string value to the appropriate type based on the default value.

    Args:
        key: The key/parameter name
        sval: The string value to cast
        defaults: Dictionary of default values

    Returns:
        The casted value with the appropriate type
    """
    default = get_default_value(defaults, key)
    if sval == "":
        return None if default is None else default
    if sval.lower() == "none":
        return None if default is None else (sval.lower() if isinstance(default, str) else default)
    if isinstance(default, bool):
        return sval.lower() in ("1", "true", "yes", "y", "t")
    if isinstance(default, int):
        return int(sval)
    if isinstance(default, float):
        return float(sval)
    if isinstance(default, str):
        return sval
    return sval


def _read_args_file(path: str, not_found_msg: str) -> dict[str, str]:
    if not os.path.exists(path):
        log_and_raise(FileNotFoundError, not_found_msg, logger_instance=logger)

    raw: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            raw[k.strip()] = v.strip()

    return raw


def load_args_from_file(
    path: str,
    defaults: dict[str, Any],
    *,
    use_model_defaults: bool = False,
    resolve_paths: tuple[str, ...] = (),
) -> argparse.Namespace:
    """Load arguments from a key=value file format.

    Supports three usage patterns controlled by keyword arguments:

    1. **Simple** (default): Uses only the provided *defaults*.
    2. **Model-aware** (``use_model_defaults=True``): Reads ``model=`` from the
       file, merges model-specific defaults from ``MODEL_REGISTRY`` on top of
       *defaults*, then applies file overrides.
    3. **Path-resolving** (``resolve_paths``): Like model-aware, but also
       resolves the listed keys as paths relative to
       ``<log_dir>/<experiment_name>`` when they are not absolute.

    Args:
        path: Path to the arguments file.
        defaults: Dictionary of default values for all parameters.
        use_model_defaults: If True, require ``model=`` in the file and merge
            model-specific defaults from MODEL_REGISTRY.
        resolve_paths: Keys whose string values should be resolved relative to
            ``<log_dir>/<experiment_name>``. Implies ``use_model_defaults=True``.

    Returns:
        Namespace object containing all parsed arguments.

    Raises:
        FileNotFoundError: If the args file doesn't exist.
        ValueError: If ``use_model_defaults`` is True and ``model`` is missing
            or unknown, or if ``resolve_paths`` is non-empty and base_dir
            cannot be determined.
    """
    if resolve_paths:
        use_model_defaults = True

    msg = CONFIG_ERRORS.ARGS_FILE_NOT_FOUND.format(path=path)
    raw = _read_args_file(path, msg)

    merged_defaults = defaults
    if use_model_defaults:
        if "model" not in raw:
            log_and_raise(ValueError, "'model' parameter is required in args file", logger_instance=logger)

        model_type_value = raw["model"]
        try:
            model_type = validate_string_value(model_type_value, MODEL_TYPES, "model type")
        except ValueError:
            valid_models = ", ".join(MODEL_TYPES)
            msg = CONFIG_ERRORS.UNKNOWN_MODEL_TYPE_IN_CONFIG.format(model=model_type_value, valid=valid_models)
            log_and_raise(ValueError, msg, logger_instance=logger)

        merged_defaults = {**defaults, **MODEL_REGISTRY[model_type]}

    # determine base_dir for path resolution
    base_dir_value: str | None = None
    if resolve_paths:
        if "log_dir" in raw and "experiment_name" in raw:
            base_dir_value = os.path.join(raw["log_dir"], raw["experiment_name"])

        if base_dir_value is None:
            log_and_raise(ValueError, "base_dir is required to resolve relative paths in args file", logger_instance=logger)

    final: dict[str, object] = {}
    for k in merged_defaults:
        if k in raw:
            try:
                val = cast_value(k, raw[k], merged_defaults)
            except (ValueError, TypeError):
                logger.warning(CONFIG_WARNINGS.CAST_VALUE_FAILED.format(key=k, value=raw[k], default=merged_defaults[k]))
                val = merged_defaults[k]
        else:
            val = merged_defaults[k]

        if resolve_paths and k in resolve_paths and isinstance(val, str) and not os.path.isabs(val):
            assert base_dir_value is not None
            val = os.path.join(base_dir_value, val)

        final[k] = val

    ns = argparse.Namespace(**final)
    ns._user_keys = frozenset(k for k in merged_defaults if k in raw)
    return ns


def get_model_defaults_for_type(model_type: str) -> dict[str, Any]:
    """
    Get default values for a specific model type.

    Args:
        model_type: The model type (e.g., 'deeplab', 'unet')

    Returns:
        Dictionary of default values for the model

    Raises:
        ValueError: If model type is unknown
    """
    try:
        model_type_normalized = validate_string_value(model_type, MODEL_TYPES, "model type")
    except ValueError:
        valid_models = ", ".join(MODEL_TYPES)
        msg = CONFIG_ERRORS.UNKNOWN_MODEL_TYPE_IN_CONFIG.format(model=model_type, valid=valid_models)
        log_and_raise(ValueError, msg, logger_instance=logger)

    return cast(dict[str, Any], dict(MODEL_REGISTRY[model_type_normalized]))


_TRAIN_NUMERIC_CONSTRAINTS: tuple[NumericConstraint, ...] = (
    NumericConstraint("batch_size", "positive", CONFIG_ERRORS.BATCH_SIZE_POSITIVE),
    NumericConstraint("workers", "non_negative", CONFIG_ERRORS.WORKERS_NON_NEGATIVE),
    NumericConstraint("max_epochs", "positive", "max_epochs must be positive, got {max_epochs}"),
    NumericConstraint("lr", "positive", "lr (learning rate) must be positive, got {lr}"),
    NumericConstraint("momentum", "range", "momentum must be in [0, 1], got {momentum}"),
    NumericConstraint("weight_decay", "non_negative", "weight_decay must be non-negative, got {weight_decay}"),
    NumericConstraint("val_freq", "positive", "val_freq must be positive, got {val_freq}"),
    NumericConstraint("save_freq", "non_negative", CONFIG_ERRORS.SAVE_FREQ_NON_NEGATIVE),
    NumericConstraint("log_freq", "positive", "log_freq must be positive, got {log_freq}"),
    NumericConstraint("num_steps", "positive", "num_steps must be positive, got {num_steps}", check_not_none=True),
    NumericConstraint(
        "accumulation_steps",
        "positive",
        "accumulation_steps must be positive, got {accumulation_steps}",
        check_not_none=True,
    ),
    NumericConstraint("focal_gamma", "non_negative", "focal_gamma must be non-negative, got {focal_gamma}"),
    NumericConstraint("focal_alpha", "range", "focal_alpha must be in [0, 1], got {focal_alpha}"),
    NumericConstraint(
        "gce_q",
        "range",
        "gce_q must be in (0, 1], got {gce_q}",
        range_min=1e-8,
        range_max=1.0,
    ),
    NumericConstraint("sce_alpha", "non_negative", "sce_alpha must be non-negative, got {sce_alpha}"),
    NumericConstraint("sce_beta", "non_negative", "sce_beta must be non-negative, got {sce_beta}"),
    NumericConstraint("dice_smooth", "non_negative", "dice_smooth must be non-negative, got {dice_smooth}"),
    NumericConstraint(
        "noisy_label_warmup_steps",
        "non_negative",
        "noisy_label_warmup_steps must be non-negative, got {noisy_label_warmup_steps}",
        check_hasattr=True,
    ),
    NumericConstraint(
        "noisy_label_ema_alpha",
        "range",
        "noisy_label_ema_alpha must be in [0, 1], got {noisy_label_ema_alpha}",
        check_hasattr=True,
    ),
    NumericConstraint(
        "noisy_label_min_weight",
        "range",
        "noisy_label_min_weight must be in [0, 1], got {noisy_label_min_weight}",
        check_hasattr=True,
    ),
    NumericConstraint(
        "gradual_unfreeze_every_n",
        "positive",
        "gradual_unfreeze_every_n must be positive, got {gradual_unfreeze_every_n}",
        check_hasattr=True,
    ),
)

_TRAIN_PROFILING_CONSTRAINTS: tuple[NumericConstraint, ...] = (
    NumericConstraint(
        "profile_steps",
        "positive",
        "profile_steps must be positive, got {profile_steps}",
        check_hasattr=True,
        check_not_none=True,
    ),
)


def warn_disabled_children_overrides(args: argparse.Namespace) -> None:
    """Warn when user explicitly sets child params that have no effect.

    Handles two cases:
    1. Boolean toggles (enable_key is False) — child params are irrelevant.
    2. Enum selectors (selector_key) — only params for the current value matter.
    """
    user_keys: frozenset[str] = getattr(args, "_user_keys", frozenset())
    if not user_keys:
        return

    items = vars(args)
    for _group_name, group_def in CONFIG_PARAMETER_GROUPS.items():
        # --- Boolean toggle warnings ---
        enable_key = group_def.get("enable_key")
        if enable_key:
            enable_val = items.get(enable_key)
            if enable_val is False or str(enable_val).lower() == "false":
                children: set[str] = set()
                if "params" in group_def:
                    children.update(group_def["params"])
                if "extra_params" in group_def:
                    children.update(group_def["extra_params"])
                if "params_prefix" in group_def:
                    prefix = group_def["params_prefix"]
                    children.update(k for k in items if k.startswith(prefix))
                children.discard(enable_key)

                for child in sorted(children):
                    if child in user_keys:
                        logger.warning(CONFIG_WARNINGS.DISABLED_CHILD_PARAM_SET.format(child=child, enable_key=enable_key))
            continue

        # --- Enum selector warnings ---
        selector_key = group_def.get("selector_key")
        children_by_value: dict[str, list[str]] | None = group_def.get("children_by_value")
        if selector_key and children_by_value and selector_key in items:
            current_value = str(items[selector_key]).lower()
            relevant = set(children_by_value.get(current_value, []))
            all_enum_children: set[str] = set()
            for params_list in children_by_value.values():
                all_enum_children.update(params_list)
            irrelevant = all_enum_children - relevant

            for child in sorted(irrelevant):
                if child in user_keys:
                    logger.warning(
                        CONFIG_WARNINGS.IRRELEVANT_CHILD_PARAM_SET.format(child=child, selector_key=selector_key, selector_value=items[selector_key])
                    )


class TrainArgsValidator:
    @staticmethod
    def validate(args: argparse.Namespace) -> None:
        """
        Validate training arguments with enum checks and constraints.

        Raises:
            ValueError: If any parameter is invalid
        """
        TrainArgsValidator._validate_enums(args)
        TrainArgsValidator._validate_sensor_inputs(args)
        TrainArgsValidator._validate_required_paths(args)
        TrainArgsValidator._validate_numeric_constraints(args)
        TrainArgsValidator._validate_profiling_constraints(args)
        TrainArgsValidator._warn_validation_unreachable(args)
        warn_disabled_children_overrides(args)

    @staticmethod
    def _validate_enums(args: argparse.Namespace) -> None:
        if hasattr(args, "dataset_val") and args.dataset_val is not None:
            args.dataset_val = validate_string_value(args.dataset_val, TRAIN_DATASET_TYPES, "validation dataset type")

        if hasattr(args, "log_level") and args.log_level is not None:
            args.log_level = validate_log_level(args.log_level)

        if hasattr(args, "loss_type") and args.loss_type is not None:
            args.loss_type = validate_string_value(args.loss_type, LOSS_TYPES, "loss type")

        if hasattr(args, "gradual_unfreeze_unit") and args.gradual_unfreeze_unit is not None:
            args.gradual_unfreeze_unit = validate_string_value(args.gradual_unfreeze_unit, GRADUAL_UNFREEZE_UNITS, "gradual unfreeze unit")

    @staticmethod
    def _validate_sensor_inputs(args: argparse.Namespace) -> None:
        """Validate that at least one sensor input is specified."""
        if not (args.use_s2hr or args.use_s2mr or args.use_s2lr or args.use_s1):
            msg = "At least one of use_s2hr, use_s2mr, use_s2lr, or use_s1 must be True. Cannot train with no input data!"
            log_and_raise(ValueError, msg, logger_instance=logger)

    @staticmethod
    def _validate_required_paths(args: argparse.Namespace) -> None:
        if args.data_dir_train is None:
            log_and_raise(ValueError, "data_dir_train is required for training", logger_instance=logger)

        if args.data_dir_val is None:
            log_and_raise(ValueError, "data_dir_val is required for training", logger_instance=logger)

        if args.log_dir is None:
            msg = CONFIG_ERRORS.LOG_DIR_REQUIRED
            log_and_raise(ValueError, msg, logger_instance=logger)

    @staticmethod
    def _validate_numeric_constraints(args: argparse.Namespace) -> None:
        _apply_numeric_constraints(args, _TRAIN_NUMERIC_CONSTRAINTS)

    @staticmethod
    def _validate_profiling_constraints(args: argparse.Namespace) -> None:
        _apply_numeric_constraints(args, _TRAIN_PROFILING_CONSTRAINTS)

    @staticmethod
    def _warn_validation_unreachable(args: argparse.Namespace) -> None:
        """Warn if validation will never run due to num_steps < val_freq."""
        num_steps = getattr(args, "num_steps", None)
        val_freq = getattr(args, "val_freq", None)
        if num_steps is not None and val_freq is not None and num_steps < val_freq:
            logger.warning(CONFIG_WARNINGS.NUM_STEPS_LESS_THAN_VAL_FREQ.format(num_steps=num_steps, val_freq=val_freq))


_PREDICT_NUMERIC_CONSTRAINTS: tuple[NumericConstraint, ...] = (
    NumericConstraint("batch_size", "positive", CONFIG_ERRORS.BATCH_SIZE_POSITIVE),
    NumericConstraint("workers", "non_negative", CONFIG_ERRORS.WORKERS_NON_NEGATIVE),
)


class PredictArgsValidator:
    @staticmethod
    def validate(args: argparse.Namespace) -> None:
        """
        Validate prediction arguments with enum checks and constraints.

        Raises:
            ValueError: If any parameter is invalid
            FileNotFoundError: If required files don't exist
        """
        PredictArgsValidator._validate_enums(args)
        PredictArgsValidator._validate_required_files(args)
        PredictArgsValidator._validate_numeric_constraints(args)
        warn_disabled_children_overrides(args)

    @staticmethod
    def _validate_enums(args: argparse.Namespace) -> None:
        if hasattr(args, "dataset") and args.dataset is not None:
            args.dataset = validate_string_value(args.dataset, DATASET_TYPES, "dataset type")

        if hasattr(args, "log_level") and args.log_level is not None:
            args.log_level = validate_log_level(args.log_level)

    @staticmethod
    def _validate_required_files(args: argparse.Namespace) -> None:
        if not os.path.exists(args.config_file):
            log_and_raise(FileNotFoundError, f"config_file not found: {args.config_file}", logger_instance=logger)

        if not os.path.exists(args.checkpoint_file):
            log_and_raise(FileNotFoundError, f"checkpoint_file not found: {args.checkpoint_file}", logger_instance=logger)

    @staticmethod
    def _validate_numeric_constraints(args: argparse.Namespace) -> None:
        _apply_numeric_constraints(args, _PREDICT_NUMERIC_CONSTRAINTS)


class AnalysisArgsValidator:
    @staticmethod
    def validate(args: argparse.Namespace) -> None:
        """
        Validate analysis arguments.

        Raises:
            ValueError: If any parameter is invalid
        """
        AnalysisArgsValidator._validate_enums(args)
        AnalysisArgsValidator._validate_required_paths(args)

    @staticmethod
    def _validate_enums(args: argparse.Namespace) -> None:
        if hasattr(args, "log_level") and args.log_level is not None:
            args.log_level = validate_log_level(args.log_level)

    @staticmethod
    def _validate_required_paths(args: argparse.Namespace) -> None:
        if getattr(args, "out_dir", AnalysisArgDefaults["out_dir"]) is None:
            msg = "out_dir must be specified in args file. This is where analysis outputs will be saved."
            log_and_raise(ValueError, msg, logger_instance=logger)

        if getattr(args, "analysis_dir", AnalysisArgDefaults["analysis_dir"]) is None:
            msg = "analysis_dir must be specified in args file. This is the subdirectory name for analysis outputs."
            log_and_raise(ValueError, msg, logger_instance=logger)

        if getattr(args, "log_dir", AnalysisArgDefaults["log_dir"]) is None:
            msg = CONFIG_ERRORS.LOG_DIR_REQUIRED
            log_and_raise(ValueError, msg, logger_instance=logger)

        log_dir: str = args.log_dir
        experiment_name: str = args.experiment_name
        log_dir_path = os.path.join(log_dir, experiment_name)
        if not os.path.exists(log_dir_path):
            msg = f"log_dir_path doesn't exist: {log_dir_path}. Ensure the experiment has been run and the path is correct."
            log_and_raise(ValueError, msg, logger_instance=logger)


class PlotArgsValidator:
    @staticmethod
    def validate(args: argparse.Namespace) -> None:
        """
        Validate plot arguments.

        Raises:
            ValueError: If any parameter is invalid
        """
        PlotArgsValidator._validate_enums(args)
        PlotArgsValidator._validate_required_paths(args)

    @staticmethod
    def _validate_enums(args: argparse.Namespace) -> None:
        if hasattr(args, "log_level") and args.log_level is not None:
            args.log_level = validate_log_level(args.log_level)

    @staticmethod
    def _validate_required_paths(args: argparse.Namespace) -> None:
        if getattr(args, "out_dir", PlotArgDefaults["out_dir"]) is None:
            msg = "out_dir must be specified in args file. This is where plot outputs will be saved."
            log_and_raise(ValueError, msg, logger_instance=logger)

        if getattr(args, "log_dir", PlotArgDefaults["log_dir"]) is None:
            msg = CONFIG_ERRORS.LOG_DIR_REQUIRED
            log_and_raise(ValueError, msg, logger_instance=logger)

        log_dir: str = args.log_dir
        experiment_name: str = args.experiment_name
        log_dir_path = os.path.join(log_dir, experiment_name)
        if not os.path.exists(log_dir_path):
            msg = f"log_dir_path doesn't exist: {log_dir_path}. Ensure the experiment has been run and the path is correct."
            log_and_raise(ValueError, msg, logger_instance=logger)


def validate_train_args(args: argparse.Namespace) -> None:
    """Validate training arguments. Alias for TrainArgsValidator.validate()."""
    TrainArgsValidator.validate(args)


def validate_predict_args(args: argparse.Namespace) -> None:
    """Validate prediction arguments. Alias for PredictArgsValidator.validate()."""
    PredictArgsValidator.validate(args)


def validate_analysis_args(args: argparse.Namespace) -> None:
    """Validate analysis arguments. Alias for AnalysisArgsValidator.validate()."""
    AnalysisArgsValidator.validate(args)


def validate_plot_args(args: argparse.Namespace) -> None:
    """Validate plot arguments. Alias for PlotArgsValidator.validate()."""
    PlotArgsValidator.validate(args)
