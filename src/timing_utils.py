import logging
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from functools import wraps
from typing import Any

import numpy as np
import torch

from .constants import TIMING_FORMAT, TrainArgDefaults

logger = logging.getLogger(__name__)

# Global reference to args for accessing enable_timing setting (legacy support)
_global_args: Any | None = None


# Hierarchical timing configuration


class TimingConfig:
    """Hierarchical timing configuration with local overrides.

    Allows global timing control with component-specific overrides.
    Priority: local setting > global setting > False

    Components:
        - training: Training loop operations
        - prediction: Prediction/inference operations
        - datasets: Data loading operations (default: False for legacy)
        - reporting: Report generation

    Example:
        # Enable timing globally
        TimingConfig.set_global(True)

        # But disable for datasets (verbose per-sample logging)
        TimingConfig.set_local("datasets", False)

        # Check if enabled for a component
        if TimingConfig.is_enabled("training"):
            ...
    """

    _global_enabled: bool = False
    _local_settings: dict[str, bool] = {}

    @classmethod
    def set_global(cls, enabled: bool) -> None:
        cls._global_enabled = enabled

    @classmethod
    def set_local(cls, component: str, enabled: bool) -> None:
        """Set local timing for a specific component.

        Args:
            component: Component name (e.g., 'training', 'datasets')
            enabled: Whether timing is enabled for this component
        """
        cls._local_settings[component] = enabled

    @classmethod
    def is_enabled(cls, component: str | None = None) -> bool:
        """Check if timing is enabled for a component.

        Priority: local setting > global setting > False

        Args:
            component: Component name (optional)

        Returns:
            True if timing is enabled, False otherwise
        """
        # Check component-specific setting first
        if component and component in cls._local_settings:
            return cls._local_settings[component]

        # Fall back to global setting
        return cls._global_enabled

    @classmethod
    def reset(cls) -> None:
        """Reset all settings (useful for testing)."""
        cls._global_enabled = False
        cls._local_settings.clear()


class TimingStats:
    """Statistics for a timed operation.

    Tracks count, total time, min, max for computing aggregates.

    Example:
        stats = TimingStats()
        stats.add(0.5)
        stats.add(0.3)
        print(stats)  # count=2, total=0.8000s, avg=0.4000s, min=0.3000s, max=0.5000s
    """

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.min = float("inf")
        self.max = 0.0

    @property
    def avg(self) -> float:
        """Calculate average time."""
        return self.total / self.count if self.count > 0 else 0.0

    def add(self, elapsed: float) -> None:
        """Add a timing measurement.

        Args:
            elapsed: Elapsed time in seconds
        """
        self.count += 1
        self.total += elapsed
        self.min = min(self.min, elapsed)
        self.max = max(self.max, elapsed)

    def __str__(self) -> str:
        """Format statistics as string."""
        if self.count == 0:
            return "no measurements"
        return f"count={self.count}, total={self.total:.4f}s, avg={self.avg:.4f}s, min={self.min:.4f}s, max={self.max:.4f}s"

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary.

        Returns:
            Dict with count, total, avg, min, max
        """
        return {
            "count": float(self.count),
            "total": self.total,
            "avg": self.avg,
            "min": self.min if self.count > 0 else 0.0,
            "max": self.max,
        }


class TimingAggregator:
    """Collects timing statistics for repeated operations.

    Use this to track aggregate statistics (min/max/avg/total) across
    multiple invocations of the same operation.

    Example:
        aggregator = TimingAggregator()

        for batch in dataloader:
            start = time.perf_counter()
            result = model(batch)
            aggregator.record("forward_pass", time.perf_counter() - start)

        aggregator.log_summary(logger)
        aggregator.to_tensorboard(writer, step)
    """

    def __init__(self) -> None:
        """Initialize empty aggregator."""
        self._stats: dict[str, TimingStats] = {}

    def record(self, name: str, elapsed: float) -> None:
        """Record a timing measurement.

        Args:
            name: Operation name
            elapsed: Elapsed time in seconds
        """
        if name not in self._stats:
            self._stats[name] = TimingStats()
        self._stats[name].add(elapsed)

    def log_summary(
        self,
        logger_instance: logging.Logger,
        prefix: str = "",
    ) -> None:
        """Log summary of all collected statistics.

        Args:
            logger_instance: Logger to use
            prefix: Optional prefix for log messages
        """
        if not self._stats:
            return

        header = f"[TIMING SUMMARY] {prefix}" if prefix else "[TIMING SUMMARY]"
        logger_instance.info(header)
        for name, stats in sorted(self._stats.items()):
            logger_instance.info(f"  {name}: {stats}")

    def to_tensorboard(
        self,
        writer: Any,
        step: int,
        prefix: str = "timing_summary",
    ) -> None:
        """Write summary statistics to TensorBoard.

        Args:
            writer: TensorBoard SummaryWriter
            step: Global step for indexing
            prefix: Prefix for metric names
        """
        if writer is None:
            return

        for name, stats in self._stats.items():
            # Sanitize name for TensorBoard (replace / with _)
            safe_name = name.replace("/", "_")
            writer.add_scalar(f"{prefix}/{safe_name}_avg", stats.avg, step)
            writer.add_scalar(f"{prefix}/{safe_name}_total", stats.total, step)
            writer.add_scalar(f"{prefix}/{safe_name}_count", stats.count, step)

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Convert all statistics to dictionary.

        Returns:
            Dict mapping operation names to stat dicts
        """
        return {name: stats.to_dict() for name, stats in self._stats.items()}

    def reset(self) -> None:
        """Clear all statistics."""
        self._stats.clear()

    def __len__(self) -> int:
        """Get number of tracked operations."""
        return len(self._stats)

    def __contains__(self, name: str) -> bool:
        """Check if operation is tracked."""
        return name in self._stats


# Global aggregator instance
_global_aggregator = TimingAggregator()


def get_aggregator() -> TimingAggregator:
    return _global_aggregator


def reset_aggregator() -> None:
    _global_aggregator.reset()


# Legacy support


def set_timing_args(args: Any) -> None:
    """Set global args reference for timing configuration (legacy).

    This allows the timeit decorator to check args.enable_timing at runtime
    without requiring it to be passed to every decorator call.

    Note: Prefer using load_timing_config_from_args() for new code.

    Example:
        args = load_args_from_file(...)
        set_timing_args(args)
    """
    global _global_args
    _global_args = args

    # Also update TimingConfig for consistency
    if hasattr(args, "enable_timing"):
        TimingConfig.set_global(bool(args.enable_timing))


def load_timing_config_from_args(args: Any) -> None:
    """Load timing configuration from args namespace.

    Configures both global and component-specific timing settings
    from an args namespace (typically loaded from args_file.txt).

    Expected attributes:
    - enable_timing: bool (global)
    - enable_timing_training: bool (optional)
    - enable_timing_prediction: bool (optional)
    - enable_timing_datasets: bool (optional)
    - enable_timing_reporting: bool (optional)

    Example:
        args = load_args_from_file("args_file.txt", defaults)
        load_timing_config_from_args(args)
    """
    # Also set legacy args reference
    global _global_args
    _global_args = args

    # Set global timing
    TimingConfig.set_global(getattr(args, "enable_timing", TrainArgDefaults["enable_timing"]))

    # Set component-specific settings (if present in args)
    components = [
        "training",
        "prediction",
        "datasets",
        "reporting",
    ]

    for comp in components:
        attr_name = f"enable_timing_{comp}"
        if hasattr(args, attr_name):
            value = getattr(args, attr_name)
            if value is not None:
                TimingConfig.set_local(comp, bool(value))


def format_arg_value(arg: Any) -> str:
    """Format function arguments for logging.

    Handles common types with appropriate detail levels:
    - torch.Tensor: shape, dtype, device
    - np.ndarray: shape, dtype
    - DataLoader: batch_size, dataset length
    - Large collections: length only
    - Primitives: actual value
    - Others: type name
    """
    # Handle torch.Tensor
    if isinstance(arg, torch.Tensor):
        return f"Tensor(shape={tuple(arg.shape)}, dtype={arg.dtype}, device={arg.device})"

    # Handle numpy arrays
    if isinstance(arg, np.ndarray):
        return f"ndarray(shape={arg.shape}, dtype={arg.dtype})"

    # Handle DataLoader
    if hasattr(arg, "__class__") and arg.__class__.__name__ == "DataLoader":
        batch_size = getattr(arg, "batch_size", "?")
        dataset_len = len(arg.dataset) if hasattr(arg, "dataset") else "?"
        return f"DataLoader(batch_size={batch_size}, dataset_len={dataset_len})"

    # Handle large lists/tuples
    if isinstance(arg, (list, tuple)):
        if len(arg) > TIMING_FORMAT["MAX_DISPLAY_COLLECTION_SIZE"]:
            return f"{type(arg).__name__}[{len(arg)} items]"
        # For small lists/tuples, show actual values
        items = ", ".join(format_arg_value(item) for item in arg)
        return f"{type(arg).__name__}[{items}]"

    # Handle large dictionaries
    if isinstance(arg, dict):
        if len(arg) > TIMING_FORMAT["MAX_DISPLAY_COLLECTION_SIZE"]:
            return f"dict[{len(arg)} keys]"
        # For small dicts, show keys
        keys = ", ".join(str(k) for k in arg)
        return f"dict[{keys}]"

    if isinstance(arg, (str, int, float, bool, type(None))):
        if isinstance(arg, str) and len(arg) > TIMING_FORMAT["MAX_DISPLAY_STRING_LENGTH"]:
            return f'"{arg[: TIMING_FORMAT["MAX_DISPLAY_STRING_LENGTH"] - 3]}..."'
        return repr(arg)

    # Handle Path objects
    if hasattr(arg, "__fspath__"):
        return f"Path({arg})"

    # Default: show type name
    return f"{type(arg).__name__}(...)"


# Timer context manager


class _TimerImpl:
    """Internal timer implementation - only instantiated when timing is enabled."""

    def __init__(
        self,
        name: str,
        logger_instance: logging.Logger,
        writer: Any | None = None,
        step: int | None = None,
        aggregate: bool = False,
    ) -> None:
        self.name = name
        self.logger = logger_instance
        self.writer = writer
        self.step = step
        self.aggregate = aggregate
        self.start_time: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> "_TimerImpl":
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *_args: Any) -> None:
        """Stop timing and log results."""
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            self.logger.info(f"[TIMING] {self.name}: {self.elapsed:.4f}s")

            # Log to TensorBoard if writer and step are provided
            if self.writer is not None and self.step is not None:
                self.writer.add_scalar(f"timing/{self.name}", self.elapsed, self.step)

            # Record in aggregator if requested
            if self.aggregate:
                _global_aggregator.record(self.name, self.elapsed)


def Timer(
    name: str,
    logger_instance: logging.Logger,
    writer: Any | None = None,
    step: int | None = None,
    enable_timing: bool | None = None,
    component: str | None = None,
    aggregate: bool = False,
) -> AbstractContextManager[Any]:
    """Context manager for timing code blocks with automatic logging.

    Measures elapsed time and logs to both Python logger and optionally TensorBoard.
    Returns a no-op context manager when timing is disabled for zero overhead.

    Args:
        name: Identifier for this timing measurement
        logger_instance: Python logger instance for output
        writer: Optional TensorBoard SummaryWriter for metric tracking
        step: Optional global step for TensorBoard indexing
        enable_timing: Override for enable/disable (highest priority)
        component: Component name for hierarchical config lookup
        aggregate: Whether to collect aggregate statistics

    Returns:
        Timer context manager if enabled, otherwise nullcontext (zero overhead)

    Example:
        # Basic usage
        with Timer("data_loading", logger, writer, global_step, component="training"):
            data = load_data()
        # Logs: [TIMING] data_loading: 1.2345s

        # With aggregation for repeated operations
        for batch in dataloader:
            with Timer("forward_pass", logger, writer, step,
                       component="training", aggregate=True):
                output = model(batch)
    """
    # Determine if timing is enabled (priority: direct > local > global)
    timing_enabled = enable_timing if enable_timing is not None else TimingConfig.is_enabled(component)

    if timing_enabled:
        return _TimerImpl(name, logger_instance, writer, step, aggregate)
    else:
        return nullcontext()


def timeit(
    func_name: str | None = None,
    log_args: bool = True,
    enable_timing: bool | Callable[[], bool] | None = None,
    component: str | None = None,
    aggregate: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to time function execution and optionally log arguments.

    Measures function execution time and logs formatted arguments. Timing is
    controlled by the hierarchical configuration system.

    Args:
        func_name: Custom name for timing logs (default: function.__name__)
        log_args: Whether to log formatted function arguments (default: True)
        enable_timing: Override for enable/disable. Can be:
                      - bool: static value (highest priority)
                      - Callable[[], bool]: function to call at runtime
                      - None: use hierarchical config (default)
        component: Component name for hierarchical config lookup
        aggregate: Whether to collect aggregate statistics

    Returns:
        Decorated function with timing instrumentation

    Example:
        # Basic usage with component
        @timeit("load_satellite_data", component="datasets")
        def load_s2(path: str, use_hr: bool) -> np.ndarray:
            ...

        # With aggregation
        @timeit("process_batch", component="training", aggregate=True)
        def process(batch: torch.Tensor) -> torch.Tensor:
            ...

        # Force enable (ignores config)
        @timeit("critical_op", enable_timing=True)
        def critical_operation():
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        name = func_name if func_name is not None else func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine if timing is enabled at runtime
            if callable(enable_timing):
                timing_enabled = enable_timing()
            elif enable_timing is not None:
                timing_enabled = bool(enable_timing)
            else:
                # Use hierarchical config
                timing_enabled = TimingConfig.is_enabled(component)

            # Fast path: timing disabled
            if not timing_enabled:
                return func(*args, **kwargs)

            # Log arguments if requested
            if log_args:
                arg_strs = [format_arg_value(a) for a in args]
                kwarg_strs = [f"{k}={format_arg_value(v)}" for k, v in kwargs.items()]
                all_args = ", ".join(arg_strs + kwarg_strs)
                logger.debug(f"[TIMING] Calling {name}({all_args})")

            # Time execution
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            logger.info(f"[TIMING] {name}: {elapsed:.4f}s")

            # Record in aggregator if requested
            if aggregate:
                _global_aggregator.record(name, elapsed)

            return result

        return wrapper

    return decorator
