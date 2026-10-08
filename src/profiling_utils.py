import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from .constants import TrainArgDefaults

logger = logging.getLogger(__name__)


class TrainingProfiler:
    """Profiler for PyTorch training loops.

    Wraps torch.profiler.profile to provide easy integration with training loops.
    Handles Chrome trace export, TensorBoard integration, and key averages tables.

    Attributes:
        output_dir: Directory to save profiling outputs
        profile_steps: Number of training steps to profile
        enable_profiling: Whether profiling is enabled
        skip_first: Number of initial steps to skip (avoids initialization overhead)
    """

    def __init__(
        self,
        output_dir: str,
        profile_steps: int = 5,
        enable_profiling: bool = False,
        skip_first: int | None = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,  # Disabled by default - can cause issues
        with_flops: bool = True,
        with_modules: bool = True,
    ) -> None:
        """Initialize the training profiler.

        Args:
            output_dir: Directory to save profiling outputs
            profile_steps: Number of training steps to profile
            enable_profiling: Whether profiling is enabled
            skip_first: Number of initial steps to skip before profiling starts
            record_shapes: Record tensor shapes for detailed analysis
            profile_memory: Profile CUDA memory usage
            with_stack: Record source code stack traces (can cause issues, disabled by default)
            with_flops: Estimate FLOPs for operators
            with_modules: Record module hierarchy
        """
        self.output_dir = output_dir
        self.profile_steps = profile_steps
        self.enable_profiling = enable_profiling
        self.skip_first = skip_first if skip_first is not None else 10
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory and torch.cuda.is_available()
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules

        self._profiler: torch.profiler.profile | None = None
        self._step_count = 0
        self._profiling_started = False
        self._profiling_finished = False
        self._tensorboard_dir: str | None = None

        if self.enable_profiling:
            os.makedirs(self.output_dir, exist_ok=True)
            msg = f"Profiling enabled: will profile {self.profile_steps} steps starting at step {self.skip_first}"
            logger.info(msg)

            if self.profile_memory and not torch.cuda.is_available():
                logger.warning("CUDA profiling requested but CUDA is not available. Profiling CPU only.")
                self.profile_memory = False

            if self.profile_steps < 5:
                msg = f"profile_steps={self.profile_steps} is very low. Consider using at least 5 steps for meaningful results."
                logger.warning(msg)

    def _create_profiler(self) -> torch.profiler.profile:
        """Create and configure the PyTorch profiler."""
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self._tensorboard_dir = os.path.join(self.output_dir, "tensorboard")
        os.makedirs(self._tensorboard_dir, exist_ok=True)

        # Use schedule-based profiler with tensorboard_trace_handler for proper TensorBoard support
        # The handler writes traces in the format TensorBoard expects
        schedule = torch.profiler.schedule(
            wait=1,  # Skip 1 step
            warmup=1,  # Warmup for 1 step
            active=self.profile_steps,  # Profile for profile_steps
            repeat=1,  # Only one cycle
        )

        profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self._tensorboard_dir),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            with_modules=self.with_modules,
        )

        return profiler

    @contextmanager
    def profile_step(self, step: int) -> Iterator[None]:
        """Context manager for profiling a single training step.

        Yields:
            None

        Example:
            with profiler.profile_step(step):
                prediction = model(image)
                loss.backward()
                optimizer.step()
        """
        if not self.enable_profiling or self._profiling_finished:
            yield
            return

        # Skip initial steps to avoid profiling initialization overhead
        if step < self.skip_first:
            yield
            return

        # Total steps needed for schedule: wait + warmup + active
        total_schedule_steps = 1 + 1 + self.profile_steps  # wait=1, warmup=1, active=profile_steps

        # Check if we've already completed the profiling schedule
        if self._step_count >= total_schedule_steps:
            if self._profiling_started and not self._profiling_finished:
                self.finish()
            yield
            return

        # Start profiler on first profiled step
        if not self._profiling_started:
            self._profiler = self._create_profiler()
            self._profiler.__enter__()
            self._profiling_started = True
            msg = f"[PROFILING] Started profiling at step {step}"
            logger.info(msg)

        try:
            yield
        finally:
            if self._profiler is not None and not self._profiling_finished:
                try:
                    self._profiler.step()
                except Exception as e:
                    # Schedule completed, finish profiling
                    logger.debug(f"Profiler schedule completed: {e}")
                    self.finish()
                    return

                self._step_count += 1

                msg = f"[PROFILING] Profiled step {step}"
                logger.debug(msg)

                # Check if we've completed the schedule
                if self._step_count >= total_schedule_steps:
                    self.finish()

    def finish(self) -> None:
        """Finish profiling and save results.

        Exports Chrome trace, key averages table, and other profiling data.
        Safe to call multiple times (will only execute once).
        """
        if not self.enable_profiling or self._profiling_finished:
            return

        if self._profiler is None:
            return

        self._profiling_finished = True

        # Exit profiler context
        try:
            self._profiler.__exit__(None, None, None)
        except Exception as e:
            logger.debug(f"Profiler context exit: {e}")

        msg = f"[PROFILING] Profiling complete after {self._step_count} steps"
        logger.info(msg)

        # Export Chrome trace (separate from TensorBoard)
        try:
            trace_path = os.path.join(self.output_dir, "trace.json")
            self._profiler.export_chrome_trace(trace_path)
            msg = f"[PROFILING] Chrome trace saved to: {trace_path}"
            logger.info(msg)
        except Exception as e:
            logger.warning(f"Failed to export Chrome trace: {e}")

        # Export key averages table
        try:
            table_path = os.path.join(self.output_dir, "key_averages.txt")
            sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
            table = self._profiler.key_averages().table(
                sort_by=sort_key,
                row_limit=50,
            )
            with open(table_path, "w") as f:
                f.write(table)
            msg = f"[PROFILING] Key averages table saved to: {table_path}"
            logger.info(msg)

            # Log summary to console
            summary_table = self._profiler.key_averages().table(
                sort_by=sort_key,
                row_limit=10,
            )
            msg = f"[PROFILING] Summary:\n{summary_table}"
            logger.info(msg)
        except Exception as e:
            logger.warning(f"Failed to export key averages table: {e}")

        # Export stacks if available
        if self.with_stack:
            try:
                stacks_path = os.path.join(self.output_dir, "stacks.txt")
                self._profiler.export_stacks(stacks_path, metric="self_cuda_time_total")
                msg = f"[PROFILING] Stacks saved to: {stacks_path}"
                logger.info(msg)
            except Exception as e:
                logger.debug(f"Failed to export stacks (this is normal for CPU-only): {e}")

        # TensorBoard traces are written by tensorboard_trace_handler automatically
        if self._tensorboard_dir:
            msg = f"[PROFILING] TensorBoard profiling data saved to: {self._tensorboard_dir}"
            logger.info(msg)

        # Log instructions for viewing results
        logger.info(
            f"[PROFILING] To view in TensorBoard, run:\n"
            f"  tensorboard --logdir={self._tensorboard_dir} --bind_all\n"
            f"  Then forward port 6006 to your local machine.\n"
            f"  Look for the 'PyTorch Profiler' tab in TensorBoard."
        )


def create_profiler_from_args(args: Any) -> TrainingProfiler:
    """Create a TrainingProfiler from parsed arguments.

    Args:
        args: Namespace containing profiling configuration:
            - enable_profiling: bool
            - profile_steps: int
            - profile_output_dir: str | None
            - log_dir: str
            - experiment_name: str

    Returns:
        Configured TrainingProfiler instance
    """
    enable_profiling = getattr(args, "enable_profiling", TrainArgDefaults["enable_profiling"])
    profile_steps = getattr(args, "profile_steps", TrainArgDefaults["profile_steps"])

    # Determine output directory
    profile_output_dir = getattr(args, "profile_output_dir", TrainArgDefaults["profile_output_dir"])
    if profile_output_dir is None:
        log_dir = getattr(args, "log_dir", TrainArgDefaults["log_dir"]) or "."
        experiment_name = getattr(args, "experiment_name", TrainArgDefaults["experiment_name"])
        profile_output_dir = os.path.join(log_dir, experiment_name, "profiling")

    return TrainingProfiler(
        output_dir=profile_output_dir,
        profile_steps=profile_steps,
        enable_profiling=enable_profiling,
    )
