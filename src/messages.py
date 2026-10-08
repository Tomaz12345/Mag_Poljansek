class CONFIG_ERRORS:
    # File not found errors
    ARGS_FILE_NOT_FOUND: str = "Args file not found: {path}"

    # Required field errors
    LOG_DIR_REQUIRED: str = "log_dir is required for training"

    # Validation errors
    BATCH_SIZE_POSITIVE: str = "batch_size must be positive, got {batch_size}"
    WORKERS_NON_NEGATIVE: str = "workers must be non-negative, got {workers}"
    SAVE_FREQ_NON_NEGATIVE: str = "save_freq must be non-negative (0 disables intermediate checkpoints), got {save_freq}"

    # Model config file errors
    UNKNOWN_MODEL_TYPE_IN_CONFIG: str = "Unknown model type '{model}' in model config file. Must be one of: {valid}"


class CONFIG_WARNINGS:
    CAST_VALUE_FAILED: str = "could not cast value for '{key}' ('{value}'), using default {default}"
    NUM_STEPS_LESS_THAN_VAL_FREQ: str = "num_steps ({num_steps}) < val_freq ({val_freq}): validation will never run during training"
    DISABLED_CHILD_PARAM_SET: str = "'{child}' is set but has no effect because '{enable_key}' is False"
    IRRELEVANT_CHILD_PARAM_SET: str = "'{child}' is set but has no effect because '{selector_key}' is '{selector_value}'"


class DATASET_ERRORS:
    # File not found errors
    DATASET_PATH_NOT_EXIST: str = "Dataset path does not exist: {path}"


class MODEL_ERRORS:
    PARAMETER_VALUE_NOT_SUPPORTED: str = "Parameter value '{value}' not supported in parameter '{parameter}'"


class UTILS_ERRORS:
    CHANNEL_INDEX_OUT_OF_RANGE: str = "Channel index {index} out of range for tensor with {channels} channels. Valid indices are 0 to {max_index}."


class RESULT_SAVER_ERRORS:
    INVALID_SAVE_DIR: str = "save_dir must be a non-empty string, got {value!r}. Provide a valid directory path."


class MODEL_INIT_ERRORS:
    NUM_CLASSES_POSITIVE: str = "num_classes must be a positive integer, got {num_classes}. Typical values: 10 for DFC2020, 17 for IGBP."
    NUM_CHANNELS_POSITIVE: str = (
        "n_channels (input channels) must be a positive integer, got {n_channels}. "
        "Example: 13 for all Sentinel-2 bands, 4 for S2 high-resolution only."
    )


# Analysis script messages (src/analyze_predictions.py)


class ANALYSIS_INFO:
    SAVED_PLOT: str = "Saved {name} to {path}"


# Plot script messages (src/plot_metrics.py)


class PLOT_INFO:
    SAVED_PLOT: str = "Saved {name} to {path}"


class PLOT_ERRORS:
    OUTPUT_DIR_NOT_SET: str = "Output directory not set. Call set_output_dir() before generating plots."


# TensorBoard logger messages (src/tensorboard/tensorboard_utils.py)


class TENSORBOARD_LOGGER_WARNINGS:
    HISTOGRAM_FAILED: str = "Failed to log histogram for {name}: {error}"


class REPORTING_INFO:
    COLLECTING_DATA: str = "Collecting data for section: {section}"
    SECTION_SKIPPED: str = "Skipping section '{section}': {reason}"


class REPORTING_ERRORS:
    DATA_COLLECTION_FAILED: str = "Failed to collect data for section '{section}': {error}"


class REPORTING_WARNINGS:
    METRICS_FILE_MISSING: str = "Metrics file not found: {path}. Skipping metrics section."
    PLOT_FILE_MISSING: str = "Plot file not found: {path}. Skipping plot."
