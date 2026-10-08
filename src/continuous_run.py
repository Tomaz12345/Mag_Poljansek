import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from src.analyze_predictions import run_analysis as _run_analysis_impl

# Import programmatic interfaces from other modules
from src.constants import MODEL_TYPES, REPORT_FORMATS, REPORT_SOURCES, REPORT_STAGES, ReportArgDefaults
from src.plot_metrics import run_plotting as _run_plotting_impl
from src.predict import run_prediction as _run_prediction_impl
from src.train import run_training as _run_training_impl
from src.utils import log_and_raise, make_report_config

logger = logging.getLogger(__name__)

# Default test configuration paths (relative to project root) — one combined file per model
DEFAULT_EXAMPLE_ARGS_FILES = {
    "unet": "args_files/args_reproducible_test_unet.txt",
    "deeplabv3+": "args_files/args_reproducible_test_deeplab.txt",
    "swin": "args_files/args_reproducible_test_swin.txt",
    "copernicus_fm": "args_files/args_reproducible_test_copernicus_fm.txt",
    "unetformer": "args_files/args_reproducible_test_unetformer.txt",
    "croma": "args_files/args_reproducible_test_croma.txt",
    "fleximo": "args_files/args_reproducible_test_fleximo.txt",
    "dinov2": "args_files/args_reproducible_test_dinov2.txt",
    "oneformer": "args_files/args_reproducible_test_oneformer.txt",
    "deo_fm": "args_files/args_reproducible_test_deo_fm.txt",
    "vit": "args_files/args_reproducible_test_vit.txt",
}
if len(DEFAULT_EXAMPLE_ARGS_FILES) != len(MODEL_TYPES):
    log_and_raise(
        ValueError,
        f"DEFAULT_EXAMPLE_ARGS_FILES has {len(DEFAULT_EXAMPLE_ARGS_FILES)} entries but MODEL_TYPES has {len(MODEL_TYPES)}. Keep them in sync.",
        logger_instance=logger,
    )

# Get project root directory
ROOT = Path(__file__).resolve().parent.parent


def validate_outputs(log_dir: str, experiment_name: str) -> tuple[bool, list[str]]:
    """
    Validate that expected outputs were created.

    Returns:
        tuple: (success: bool, missing_items: list[str])
    """
    logger.info("Validating training outputs...")

    log_path = Path(log_dir) / experiment_name
    missing_items = []

    # Check for log directory
    if not log_path.exists():
        missing_items.append(f"Log directory: {log_path}")

    # Check for checkpoints directory
    checkpoints_path = log_path / "checkpoints"
    if not checkpoints_path.exists():
        missing_items.append(f"Checkpoints directory: {checkpoints_path}")
    else:
        # Check for args.pkl
        if not (checkpoints_path / "args.pkl").exists():
            missing_items.append("args.pkl file")

        # Check for final checkpoint
        if not (checkpoints_path / "final.pth").exists():
            missing_items.append("final.pth checkpoint")

    logs_file = log_path / "logs.log"
    if not logs_file.exists():
        missing_items.append("logs.log file")

    if missing_items:
        logger.error(f"Missing expected outputs: {missing_items}")
        return False, missing_items

    logger.info("All expected outputs found")
    return True, []


def run_tensorboard_check(log_dir: str, experiment_name: str) -> tuple[bool, str]:
    """
    Verify that TensorBoard files were created.

    Returns:
        tuple: (success: bool, error_message: str)
    """
    logger.info("Checking TensorBoard files")

    log_path = Path(log_dir) / experiment_name

    # Check for TensorBoard event files
    tb_files = list(log_path.glob("events.out.tfevents.*"))

    if not tb_files:
        error_msg = "No TensorBoard event files found"
        logger.error(error_msg)
        return False, error_msg

    logger.info(f"Found {len(tb_files)} TensorBoard event file(s)")
    logger.info("TensorBoard check passed")
    return True, ""


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


def cleanup_experiment_folder(log_dir: str, experiment_name: str) -> bool:
    """Delete the experiment's log folder.

    Args:
        log_dir: Base log directory
        experiment_name: Name of the experiment

    Returns:
        True if cleanup was successful, False otherwise
    """
    experiment_path = Path(log_dir) / experiment_name
    if not experiment_path.exists():
        logger.info(f"Experiment folder does not exist, nothing to clean up: {experiment_path}")
        return True

    try:
        logger.info(f"Cleaning up experiment folder: {experiment_path}")
        shutil.rmtree(experiment_path)
        logger.info("Deleted experiment folder: %s", experiment_path)
        return True
    except Exception as e:
        logger.error(f"Failed to delete experiment folder {experiment_path}: {e}")
        return False


def run_training(args_file_path: str) -> tuple[bool, str]:
    """Run the training stage.

    Args:
        args_file_path: Path to combined arguments file (common + model-specific)

    Returns:
        tuple: (success: bool, error_message: str)
    """
    logger.info("Running training")

    try:
        _run_training_impl(args_file_path)
        logger.info("Training finished")
        return True, ""
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def run_prediction(args_file_path: str) -> tuple[bool, str]:
    """Run the prediction stage.

    Args:
        args_file_path: Path to combined arguments file (common + model-specific)

    Returns:
        tuple: (success: bool, error_message: str)
    """
    logger.info("Running prediction")

    try:
        _run_prediction_impl(args_file_path)
        logger.info("Prediction finished")
        return True, ""
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def run_plotting(args_file_path: str) -> tuple[bool, str]:
    """Run the metrics plotting stage.

    Args:
        args_file_path: Path to common arguments file

    Returns:
        tuple: (success: bool, error_message: str)
    """
    logger.info("Running plotting")

    try:
        _run_plotting_impl(args_file_path)
        logger.info("Plotting finished")
        return True, ""
    except Exception as e:
        error_msg = f"Plotting failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def run_analysis(args_file_path: str) -> tuple[bool, str]:
    """Run the prediction analysis stage.

    Args:
        args_file_path: Path to common arguments file

    Returns:
        tuple: (success: bool, error_message: str)
    """
    logger.info("Running analysis")

    try:
        _run_analysis_impl(args_file_path)
        logger.info("Analysis finished")
        return True, ""
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def main() -> int:
    """Main continuous run pipeline.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Continuous run pipeline for DFC2020 baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--args_file_path",
        type=str,
        default=None,
        help="Path to combined arguments file (key=value format, contains both common and model-specific params). Required unless --example is used.",
    )

    # Example/test mode
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run with example/test configuration using data_small dataset. Enables all stages (training, prediction, analysis).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_TYPES),
        default="unet",
        help="Model to use in --example mode (default: unet)",
    )

    # Cleanup
    parser.add_argument(
        "--cleanup-after-run",
        action="store_true",
        help="Delete experiment folder after successful run (useful for CI/CD)",
    )

    # Stage control flags
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training stage (use existing checkpoint)",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip prediction stage",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip metrics plotting stage",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip prediction analysis stage",
    )

    # Reporting options
    parser.add_argument(
        "--report-format",
        type=str,
        nargs="+",
        choices=list(REPORT_FORMATS.values()),
        default=[ReportArgDefaults["report_format"]],
        help=f"Report output format(s) (default: {ReportArgDefaults['report_format']})",
    )
    parser.add_argument(
        "--report-num-samples",
        type=int,
        default=ReportArgDefaults["report_num_samples"],
        help=f"Number of sample predictions to include in report (default: {ReportArgDefaults['report_num_samples']})",
    )
    parser.add_argument(
        "--report-sample-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific sample IDs to include in report (e.g., ROIs1868_summer_lc_89_p738)",
    )

    args = parser.parse_args()

    # Handle --example mode
    if args.example:
        # Select combined args file based on --model
        model_type = args.model.lower()
        if model_type not in DEFAULT_EXAMPLE_ARGS_FILES:
            logger.error(f"Unknown model: {args.model}")
            return 1

        args.args_file_path = str(ROOT / DEFAULT_EXAMPLE_ARGS_FILES[model_type])

        # Enable all optional stages in example mode
        logger.info("Running in --example mode with test configuration")
        logger.info(f"  Model: {model_type}")
        logger.info(f"  Args file: {args.args_file_path}")

    # Validate required arguments
    if args.args_file_path is None:
        logger.error("--args_file_path is required (or use --example for test mode)")
        return 1

    if not Path(args.args_file_path).exists():
        logger.error(f"Args file not found: {args.args_file_path}")
        return 1

    # Print header
    logger.info("Continuous run pipeline")

    # Track pipeline state
    start_time = time.time()
    stages_run = []

    from src.config import load_args_from_file
    from src.constants import TrainArgDefaults

    # Merge defaults
    defaults = TrainArgDefaults.copy()

    train_args = load_args_from_file(args.args_file_path, defaults)
    experiment_name = train_args.experiment_name
    log_dir = train_args.log_dir

    try:
        # Stage 1: training
        if not args.skip_train:
            logger.info("Step 1: Running training...")
            success, error_msg = run_training(args.args_file_path)

            if not success:
                logger.error(f"Training failed: {error_msg}")
                return 1

            logger.info("Training stage complete")
            stages_run.append(REPORT_STAGES["TRAINING"])

            # Validate training outputs
            logger.info("Validating training outputs...")
            outputs_valid, missing_items = validate_outputs(log_dir, experiment_name)

            if not outputs_valid:
                logger.error("Output validation failed")
                return 1
            logger.info("Training outputs validated")

            # Check TensorBoard files
            logger.info("Checking TensorBoard files...")
            success, error_msg = run_tensorboard_check(log_dir, experiment_name)

            if not success:
                logger.warning(f"TensorBoard check failed: {error_msg}")
                logger.warning("Continuing with pipeline...")
            else:
                logger.info("TensorBoard files validated")
        else:
            logger.info("Skipping training stage (using existing checkpoint)")

            # Verify checkpoint exists
            checkpoint_dir = os.path.join(log_dir, experiment_name, "checkpoints")
            if not os.path.exists(checkpoint_dir):
                logger.error(f"Checkpoint directory does not exist: {checkpoint_dir}")
                logger.error("Must either run training or have existing checkpoint")
                return 1

        # Stage 2: prediction
        if not args.skip_predict:
            logger.info("Step 2: Running inference/prediction...")
            success, error_msg = run_prediction(args.args_file_path)

            if not success:
                logger.error(f"Inference failed: {error_msg}")
                return 1

            logger.info("Prediction stage complete")
            stages_run.append(REPORT_STAGES["PREDICTION"])
        else:
            logger.info("Skipping prediction stage")

        # Stage 3: plot metrics
        if not args.skip_plot:
            logger.info("Step 3: Plotting metrics...")
            success, error_msg = run_plotting(args.args_file_path)

            if not success:
                logger.warning(f"Plotting failed (non-fatal): {error_msg}")
                logger.warning("Continuing with remaining steps...")
            else:
                logger.info("Metrics plotted")
        else:
            logger.info("Skipping plot metrics stage")

        # Stage 4: analyze predictions
        if not args.skip_analysis:
            logger.info("Step 4: Running prediction analysis...")
            success, error_msg = run_analysis(args.args_file_path)

            if not success:
                logger.warning(f"Analysis failed (non-fatal): {error_msg}")
                logger.warning("Continuing with remaining steps...")
            else:
                logger.info("Analysis stage complete")
                stages_run.append(REPORT_STAGES["ANALYSIS"])
        else:
            logger.info("Skipping analysis stage")

        # Report generation
        if args.report_format and len(stages_run) > 0:
            logger.info("Generating experiment report...")

            try:
                from src.reporting import generate_report

                report_config = make_report_config(
                    source=REPORT_SOURCES["CONTINUOUS_RUN"],
                    args_file_path=args.args_file_path,
                    experiment_dir=os.path.join(log_dir, experiment_name),
                    stages_run=stages_run,
                    formats=[REPORT_FORMATS["ALL"]] if REPORT_FORMATS["ALL"] in args.report_format else args.report_format,
                    report_num_samples=args.report_num_samples,
                    report_sample_ids=args.report_sample_ids,
                )

                report_paths = generate_report(report_config)

                logger.info("Reports generated:")
                for report_path in report_paths:
                    logger.info(f"  - {report_path}")

            except Exception as e:
                logger.warning(f"Report generation failed (non-fatal): {e}")
                logger.warning("Continuing with completion...")

        # Summary
        elapsed_time = time.time() - start_time

        logger.info("All validation checks passed")
        logger.info("Continuous run finished")
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Log directory: {log_dir}")
        logger.info(f"Stages run: {', '.join(stages_run)}")
        logger.info(f"Total time: {format_time(elapsed_time)}")

        # Cleanup (optional)
        if args.cleanup_after_run:
            logger.info("Cleaning up experiment folder...")
            if cleanup_experiment_folder(log_dir, experiment_name):
                logger.info("Cleanup complete")
            else:
                logger.warning("Cleanup failed, but run was successful")

        return 0

    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
