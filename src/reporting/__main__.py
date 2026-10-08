import argparse
import logging
import sys
from pathlib import Path

from src.config import load_args_from_file
from src.constants import REPORT_FORMATS, REPORT_SOURCES, REPORT_STAGES, ReportArgDefaults
from src.utils import make_report_config

from . import generate_report

logger = logging.getLogger(__name__)


def main() -> int:
    """Run report generation as a standalone command."""
    parser = argparse.ArgumentParser(
        description="Generate experiment report from a completed pipeline run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report (default: all formats)
  python -m src.reporting --args_file_path args_file.txt

  # Configure format and samples in args_file.txt:
  #   report_format=pdf,html
  #   report_num_samples=10
  #   report_sample_ids=ROIs1868_summer_lc_89_p738,ROIs1868_summer_lc_89_p999
""",
    )
    parser.add_argument(
        "--args_file_path",
        type=str,
        required=True,
        help="Path to args file (key=value format). Reads experiment settings and report options.",
    )
    parsed = parser.parse_args()

    if not Path(parsed.args_file_path).exists():
        print(f"Error: Args file not found: {parsed.args_file_path}", file=sys.stderr)
        return 1

    args = load_args_from_file(parsed.args_file_path, ReportArgDefaults.copy())

    logging.basicConfig(
        level=getattr(logging, getattr(args, "log_level", "INFO"), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    log_dir = getattr(args, "log_dir", ReportArgDefaults["log_dir"])
    experiment_name = str(getattr(args, "experiment_name", ReportArgDefaults["experiment_name"]))

    if not log_dir:
        logger.error("log_dir is not set in the args file")
        return 1

    experiment_dir = Path(log_dir) / experiment_name
    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return 1

    # Parse report_format: support comma-separated values and "all"
    raw_format: str = str(getattr(args, "report_format", REPORT_FORMATS["PDF"]))
    formats: list[str] = [f.strip() for f in raw_format.split(",") if f.strip()]
    if not formats:
        formats = [REPORT_FORMATS["PDF"]]

    # Parse report_sample_ids: comma-separated string or None
    raw_sample_ids = getattr(args, "report_sample_ids", ReportArgDefaults["report_sample_ids"])
    sample_ids: list[str] | None = None
    if raw_sample_ids is not None:
        sample_ids = [s.strip() for s in str(raw_sample_ids).split(",") if s.strip()] or None

    num_samples: int = int(getattr(args, "report_num_samples", ReportArgDefaults["report_num_samples"]))

    report_config = make_report_config(
        source=REPORT_SOURCES["CONTINUOUS_RUN"],
        args_file_path=parsed.args_file_path,
        experiment_dir=str(experiment_dir),
        stages_run=[
            REPORT_STAGES["TRAINING"],
            REPORT_STAGES["PREDICTION"],
            REPORT_STAGES["ANALYSIS"],
        ],
        formats=formats,
        report_num_samples=num_samples,
        report_sample_ids=sample_ids,
    )

    try:
        report_paths = generate_report(report_config)
        if report_paths:
            print("Reports generated successfully:")
            for path in report_paths:
                print(f"  - {path}")
        else:
            logger.warning("No reports were generated")
        return 0
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
