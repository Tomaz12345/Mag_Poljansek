import logging
from pathlib import Path
from typing import Any

from src.constants import REPORT_FORMATS, REPORT_SOURCES, REPORT_STAGES, VALID_FORMATS, VALID_SOURCES, VALID_STAGES
from src.messages import REPORTING_INFO
from src.timing_utils import Timer, timeit
from src.utils import log_and_raise

from .content_collectors import (
    AnalysisCollector,
    ConfigurationCollector,
    PredictionCollector,
    TestCollector,
    TrainingCollector,
)
from .renderers import HTMLRenderer, PDFRenderer

logger = logging.getLogger(__name__)


def _normalize_choice(value: str, valid_values: set[str], label: str) -> str:
    normalized = str(value).lower()
    if normalized in valid_values:
        return normalized
    valid = ", ".join(sorted(valid_values))
    msg = f"Invalid {label} '{value}'. Must be one of: {valid}"
    log_and_raise(ValueError, msg, logger_instance=logger, context={"value": value, "valid_values": valid})


@timeit("reporting/generate_report", log_args=False, component="reporting")
def generate_report(config: dict[str, Any]) -> list[str]:
    """
    Generate experiment report(s) in specified format(s).

    Args:
        config: Report configuration dictionary with keys:
            - source: str - Report source ('continuous_run', 'train', 'predict', 'tests')
            - args_file_path: str - Path to combined args file (common + model-specific)
            - experiment_dir: str - Path to experiment directory
            - stages_run: list[str] - List of pipeline stages that were run
            - formats: list[str] - Output formats ['pdf', 'html'] or ['all']

    Returns:
        List of paths to generated report files

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If required files are missing
    """
    logger.info("Generating experiment report...")

    # Validate configuration
    _validate_config(config)

    # Collect data from all relevant sources
    with Timer("reporting/collect_data", logger, component="reporting"):
        report_data = _collect_report_data(config)

    if not report_data:
        msg = "No data available for report generation"
        log_and_raise(ValueError, msg, logger_instance=logger)

    # Determine output formats
    formats = config.get("formats", [REPORT_FORMATS["PDF"]])
    if REPORT_FORMATS["ALL"] in formats:
        formats = [REPORT_FORMATS["PDF"], REPORT_FORMATS["HTML"]]

    # Create reports directory
    experiment_dir = Path(config["experiment_dir"])
    reports_dir = experiment_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Determine experiment name from directory
    experiment_name = experiment_dir.name

    # Render reports in each format
    generated_reports = []

    for fmt in formats:
        try:
            logger.info(f"Rendering report to {fmt}...")

            renderer: HTMLRenderer | PDFRenderer
            if fmt == REPORT_FORMATS["HTML"]:
                renderer = HTMLRenderer()
            elif fmt == REPORT_FORMATS["PDF"]:
                renderer = PDFRenderer()
            else:
                logger.warning(f"Unknown format: {fmt}, skipping")
                continue

            with Timer(f"reporting/render_{fmt}", logger, component="reporting"):
                report_path = renderer.render(report_data, reports_dir, config["source"], experiment_name)
            generated_reports.append(str(report_path))

            logger.info(f"Report generated successfully: {report_path}")

        except Exception as e:
            logger.error(f"Failed to render report to {fmt}: {str(e)}")
            # Continue with other formats even if one fails
            continue

    if generated_reports:
        logger.info(f"Reports saved to: {reports_dir}")

    return generated_reports


def _set_config_value(config: dict[str, Any], key: str, value: Any) -> None:
    if isinstance(config, dict):
        config[key] = value
    else:
        setattr(config, key, value)


def _validate_config(config: dict[str, Any]) -> None:
    """
    Validate report configuration.

    Raises:
        ValueError: If configuration is invalid
    """
    # Check required fields
    if "source" not in config:
        msg = "Report configuration must include 'source' field"
        log_and_raise(ValueError, msg, logger_instance=logger)

    source = _normalize_choice(config["source"], VALID_SOURCES, "report source")
    _set_config_value(config, "source", source)

    # For non-test sources, require args file
    if config["source"] != REPORT_SOURCES["TESTS"] and "args_file_path" not in config:
        msg = "args_file_path is required for report generation"
        log_and_raise(ValueError, msg, logger_instance=logger)

    # Check experiment directory
    if "experiment_dir" not in config:
        msg = "Report configuration must include 'experiment_dir' field"
        log_and_raise(ValueError, msg, logger_instance=logger)

    experiment_dir = Path(config["experiment_dir"])
    if not experiment_dir.exists():
        msg = f"Experiment directory not found: {experiment_dir}"
        log_and_raise(FileNotFoundError, msg, logger_instance=logger)

    # Validate formats
    formats = config.get("formats", [REPORT_FORMATS["PDF"]])
    normalized_formats = [_normalize_choice(fmt, VALID_FORMATS, "report format") for fmt in formats]

    _set_config_value(config, "formats", normalized_formats)

    # Validate stages if provided
    if "stages_run" in config:
        normalized_stages = [_normalize_choice(stage, VALID_STAGES, "pipeline stage") for stage in config["stages_run"]]

        _set_config_value(config, "stages_run", normalized_stages)


def _collect_report_data(config: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    experiment_dir = Path(config["experiment_dir"])

    # Always collect configuration (except for tests-only reports)
    if config["source"] != REPORT_SOURCES["TESTS"]:
        logger.info(REPORTING_INFO.COLLECTING_DATA.format(section="configuration"))
        config_collector = ConfigurationCollector()
        config_data = config_collector.collect(config["args_file_path"])
        if config_data:
            data["configuration"] = config_data
        else:
            logger.warning("Only partial data available for section 'configuration': Could not load configuration")

    # Collect data for each stage
    stages_run = config.get("stages_run", [])

    # Training stage
    if REPORT_STAGES["TRAINING"] in stages_run or config["source"] == REPORT_SOURCES["TRAIN"]:
        logger.info(REPORTING_INFO.COLLECTING_DATA.format(section="training"))
        training_collector = TrainingCollector()
        training_data = training_collector.collect(experiment_dir)
        if training_data:
            data["training"] = training_data
        else:
            logger.info(REPORTING_INFO.SECTION_SKIPPED.format(section="training", reason="No training data found"))

    # Prediction stage
    if REPORT_STAGES["PREDICTION"] in stages_run or config["source"] == REPORT_SOURCES["PREDICT"]:
        logger.info(REPORTING_INFO.COLLECTING_DATA.format(section="prediction"))
        prediction_collector = PredictionCollector()
        predictions_dir = experiment_dir / "predictions"
        prediction_data = prediction_collector.collect(
            predictions_dir,
            experiment_dir=experiment_dir,
            num_samples=config.get("report_num_samples"),
            sample_ids=config.get("report_sample_ids"),
        )
        if prediction_data:
            data["prediction"] = prediction_data
        else:
            logger.info(REPORTING_INFO.SECTION_SKIPPED.format(section="prediction", reason="No prediction data found"))

    # Analysis stage
    if REPORT_STAGES["ANALYSIS"] in stages_run:
        logger.info(REPORTING_INFO.COLLECTING_DATA.format(section="analysis"))
        analysis_collector = AnalysisCollector()
        analysis_dir = experiment_dir / "analysis"
        analysis_data = analysis_collector.collect(analysis_dir)
        if analysis_data:
            data["analysis"] = analysis_data
        else:
            logger.info(REPORTING_INFO.SECTION_SKIPPED.format(section="analysis", reason="No analysis data found"))

    # Tests stage
    if config["source"] == REPORT_SOURCES["TESTS"] or REPORT_STAGES["TESTS"] in stages_run:
        logger.info(REPORTING_INFO.COLLECTING_DATA.format(section="tests"))
        test_collector = TestCollector()
        test_results_file = config.get("test_results_file")
        test_data = test_collector.collect(test_results_file)
        if test_data:
            data["tests"] = test_data
        else:
            logger.info(REPORTING_INFO.SECTION_SKIPPED.format(section="tests", reason="No test data found"))

    return data
