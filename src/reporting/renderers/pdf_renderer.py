import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.constants import PLOT_DEFAULTS

logger = logging.getLogger(__name__)


class PDFRenderer:
    def __init__(self, dpi: int = PLOT_DEFAULTS["FIGURE_DPI"]) -> None:
        self.dpi = dpi
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self) -> None:
        """Set up custom paragraph styles."""
        # Title style
        self.styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=self.styles["Heading1"],
                fontSize=24,
                textColor=colors.HexColor("#2c3e50"),
                spaceAfter=30,
                alignment=1,  # Center
            )
        )

        # Section heading style
        self.styles.add(
            ParagraphStyle(
                name="SectionHeading",
                parent=self.styles["Heading2"],
                fontSize=16,
                textColor=colors.HexColor("#34495e"),
                spaceAfter=12,
                spaceBefore=12,
            )
        )

        # Subsection heading style
        self.styles.add(
            ParagraphStyle(
                name="SubsectionHeading",
                parent=self.styles["Heading3"],
                fontSize=14,
                textColor=colors.HexColor("#34495e"),
                spaceAfter=10,
                spaceBefore=10,
            )
        )

    def render(
        self,
        report_data: dict[str, Any],
        output_dir: Path,
        source: str,
        experiment_name: str,
    ) -> Path:
        """
        Render report to PDF file.

        Args:
            report_data: Collected report data
            output_dir: Directory where to save the report
            source: Report source (e.g., 'continuous_run', 'train')
            experiment_name: Name of the experiment

        Returns:
            Path to the generated PDF file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create figures directory for temporary image files
        figures_dir = output_dir / "pdf_temp_figures"
        figures_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{source}_report_{experiment_name}_{timestamp}.pdf"
        output_path = output_dir / filename

        try:
            # Build PDF content
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=0.75 * inch,
                leftMargin=0.75 * inch,
                topMargin=0.75 * inch,
                bottomMargin=0.75 * inch,
            )

            story = []

            # Title and metadata
            title_text = f"{source.replace('_', ' ').title()} Report: {experiment_name}"
            story.append(Paragraph(title_text, self.styles["CustomTitle"]))
            story.append(Spacer(1, 12))

            metadata_text = f"""
            <b>Date:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br/>
            <b>Source:</b> {source}
            """
            story.append(Paragraph(metadata_text, self.styles["Normal"]))
            story.append(Spacer(1, 20))

            # Build content sections
            chapter_num = 1

            if "configuration" in report_data:
                self._add_configuration_section(story, report_data["configuration"], chapter_num)
                chapter_num += 1

            if "training" in report_data:
                story.append(PageBreak())
                self._add_training_section(story, report_data["training"], chapter_num, figures_dir)
                chapter_num += 1

            if "prediction" in report_data:
                story.append(PageBreak())
                self._add_prediction_section(story, report_data["prediction"], chapter_num, figures_dir)
                chapter_num += 1

            if "analysis" in report_data:
                story.append(PageBreak())
                self._add_analysis_section(story, report_data["analysis"], chapter_num, figures_dir)
                chapter_num += 1

            if "tests" in report_data:
                story.append(PageBreak())
                self._add_tests_section(story, report_data["tests"], chapter_num)
                chapter_num += 1

            # Build PDF
            doc.build(story)
            logger.info(f"PDF report generated: {output_path}")

            # Clean up temporary figures
            import shutil

            shutil.rmtree(figures_dir, ignore_errors=True)

        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            raise

        return output_path

    def _add_configuration_section(self, story: list[Any], config_data: dict[str, Any], chapter_num: int) -> None:
        story.append(Paragraph(f"{chapter_num}. Configuration", self.styles["SectionHeading"]))
        story.append(Spacer(1, 12))

        # General configuration (grouped)
        story.append(Paragraph(f"{chapter_num}.1 General Configuration", self.styles["SubsectionHeading"]))

        general_grouped = config_data.get("general_grouped")
        if general_grouped:
            group_header_color = colors.HexColor("#4A6FA5")
            table_data = [["Parameter", "Value"]]
            # Track which rows are group headers for styling
            group_header_rows: list[int] = []

            for group_name, params in general_grouped.items():
                # Add group header row
                row_idx = len(table_data)
                group_header_rows.append(row_idx)
                table_data.append(
                    [
                        Paragraph(f"<b>{group_name}</b>", self.styles["Normal"]),
                        "",
                    ]
                )

                # Add parameter rows
                for param, info in params.items():
                    value_str = str(info["value"])
                    value_str_ = f"<b>{value_str}</b>" if info["source"] == "user" else f"<i>{value_str}</i>"
                    table_data.append([param, Paragraph(value_str_, self.styles["Normal"])])

            table = Table(table_data, colWidths=[3 * inch, 4 * inch])
            style_commands = [
                # Header row
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                # Default body style
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
            for row_idx in group_header_rows:
                style_commands.extend(
                    [
                        ("SPAN", (0, row_idx), (-1, row_idx)),
                        ("BACKGROUND", (0, row_idx), (-1, row_idx), group_header_color),
                        ("TEXTCOLOR", (0, row_idx), (-1, row_idx), colors.whitesmoke),
                        ("FONTNAME", (0, row_idx), (-1, row_idx), "Helvetica-Bold"),
                        ("FONTSIZE", (0, row_idx), (-1, row_idx), 11),
                    ]
                )
            table.setStyle(TableStyle(style_commands))
            story.append(table)
            story.append(Spacer(1, 8))
            story.append(
                Paragraph(
                    "<i>Note: Bold = user-specified, Italic = default. Only non-default values shown (disabled groups collapsed).</i>",
                    self.styles["Normal"],
                )
            )
        elif "general" in config_data and config_data["general"]:
            # Fallback: flat table if grouped data not available
            table_data = [["Parameter", "Value"]]
            for param, info in sorted(config_data["general"].items()):
                value_str = str(info["value"])
                value_str_ = f"<b>{value_str}</b>" if info["source"] == "user" else f"<i>{value_str}</i>"
                table_data.append([param, Paragraph(value_str_, self.styles["Normal"])])

            table = Table(table_data, colWidths=[3 * inch, 4 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 8))
            story.append(
                Paragraph(
                    "<i>Note: Bold = user-specified, Italic = default</i>",
                    self.styles["Normal"],
                )
            )

        story.append(Spacer(1, 20))

        # Model configuration
        if "model" in config_data and config_data["model"]:
            story.append(Paragraph(f"{chapter_num}.2 Model Configuration", self.styles["SubsectionHeading"]))

            table_data = [["Parameter", "Value"]]
            for param, info in sorted(config_data["model"].items()):
                value_str = str(info["value"])
                value_str_ = f"<b>{value_str}</b>" if info["source"] == "user" else f"<i>{value_str}</i>"
                table_data.append([param, Paragraph(value_str_, self.styles["Normal"])])

            table = Table(table_data, colWidths=[3 * inch, 4 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(table)

        story.append(Spacer(1, 20))

    def _add_training_section(self, story: list[Any], training_data: dict[str, Any], chapter_num: int, figures_dir: Path) -> None:
        story.append(Paragraph(f"{chapter_num}. Training", self.styles["SectionHeading"]))
        story.append(Spacer(1, 12))

        # Training summary
        story.append(Paragraph(f"{chapter_num}.1 Training Summary", self.styles["SubsectionHeading"]))

        if "final_metrics" in training_data:
            source = training_data.get("final_metrics_source", "validation")
            if source == "training":
                note = "<i>Note: Only training metrics available (no validation was performed)</i>"
                story.append(Paragraph(note, self.styles["Normal"]))
            metrics = training_data["final_metrics"]
            summary_lines = []
            for metric_name, value in metrics.items():
                label = metric_name.replace("_", " ").title()
                summary_lines.append(f"<b>Final {label}:</b> {value}")
            if "best_metrics" in training_data:
                best = training_data["best_metrics"]
                summary_lines.append(f"<b>Best mIoU:</b> {best['mIoU']} (step {best['step']})")
            summary_text = "<br/>".join(summary_lines)

            story.append(Paragraph(summary_text, self.styles["Normal"]))

        story.append(Spacer(1, 20))

        # Loss curves
        loss_curves_shown = False
        if "loss_curves_file" in training_data:
            lc_path = training_data["loss_curves_file"]
            if Path(lc_path).exists():
                story.append(Paragraph(f"{chapter_num}.2 Loss Curves", self.styles["SubsectionHeading"]))
                try:
                    img = Image(lc_path, width=5 * inch, height=3 * inch)
                    story.append(img)
                    story.append(Spacer(1, 20))
                    loss_curves_shown = True
                except Exception as e:
                    logger.warning(f"Could not embed loss_curves.png: {e}")

        has_loss_plot = "plots" in training_data and "loss_curves" in training_data["plots"]
        if not loss_curves_shown and has_loss_plot:
            story.append(Paragraph(f"{chapter_num}.2 Loss Curves", self.styles["SubsectionHeading"]))
            fig = training_data["plots"]["loss_curves"]
            img_path = figures_dir / "loss_curves.png"
            fig.savefig(img_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

            img = Image(str(img_path), width=5 * inch, height=3 * inch)
            story.append(img)
            story.append(Spacer(1, 20))
        elif has_loss_plot:
            plt.close(training_data["plots"]["loss_curves"])

        # Metrics over time
        if "plots" in training_data and "metrics_over_time" in training_data["plots"]:
            story.append(Paragraph(f"{chapter_num}.3 Metrics Over Time", self.styles["SubsectionHeading"]))
            fig = training_data["plots"]["metrics_over_time"]
            img_path = figures_dir / "metrics_over_time.png"
            fig.savefig(img_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

            img = Image(str(img_path), width=5 * inch, height=3 * inch)
            story.append(img)

    def _add_prediction_section(
        self,
        story: list[Any],
        prediction_data: dict[str, Any],
        chapter_num: int,
        figures_dir: Path,
    ) -> None:
        story.append(Paragraph(f"{chapter_num}. Prediction", self.styles["SectionHeading"]))
        story.append(Spacer(1, 12))

        # Confusion matrix
        if "confusion_matrix_plot" in prediction_data:
            story.append(Paragraph(f"{chapter_num}.1 Confusion Matrix", self.styles["SubsectionHeading"]))
            fig = prediction_data["confusion_matrix_plot"]
            img_path = figures_dir / "confusion_matrix.png"
            fig.savefig(img_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

            img = Image(str(img_path), width=4.5 * inch, height=4.5 * inch)
            story.append(img)
            story.append(Spacer(1, 20))

        # Per-class metrics
        if "per_class_metrics" in prediction_data:
            story.append(Paragraph(f"{chapter_num}.2 Per-Class Metrics", self.styles["SubsectionHeading"]))

            metrics = prediction_data["per_class_metrics"]
            if metrics:
                headers = list(metrics[0].keys())
                table_data = [headers]
                for row in metrics:
                    table_data.append([str(row[h]) for h in headers])

                table = Table(table_data)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 10),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            ("FONTSIZE", (0, 1), (-1, -1), 8),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 20))

        # Metrics distribution
        if "metrics_distribution_plot" in prediction_data:
            story.append(Paragraph(f"{chapter_num}.3 Metrics Distribution", self.styles["SubsectionHeading"]))
            fig = prediction_data["metrics_distribution_plot"]
            img_path = figures_dir / "metrics_distribution.png"
            fig.savefig(img_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

            img = Image(str(img_path), width=4.5 * inch, height=4.5 * inch)
            story.append(img)
            story.append(Spacer(1, 20))

        # Sample predictions
        has_samples = "sample_prediction_plots" in prediction_data and prediction_data["sample_prediction_plots"]
        if has_samples:
            story.append(
                Paragraph(
                    f"{chapter_num}.4 Sample Predictions",
                    self.styles["SubsectionHeading"],
                )
            )
            desc = "<i>Random sample predictions showing input bands, ground truth, and model prediction</i>"
            story.append(Paragraph(desc, self.styles["Normal"]))
            story.append(Spacer(1, 8))
            for i, fig in enumerate(prediction_data["sample_prediction_plots"]):
                img_path = figures_dir / f"sample_prediction_{i}.png"
                fig.savefig(img_path, dpi=self.dpi, bbox_inches="tight")
                plt.close(fig)

                img = Image(str(img_path), width=7 * inch, height=3.5 * inch)
                story.append(img)
                story.append(Spacer(1, 12))

    def _add_analysis_section(
        self,
        story: list[Any],
        analysis_data: dict[str, Any],
        chapter_num: int,
        figures_dir: Path | None = None,
    ) -> None:
        story.append(Paragraph(f"{chapter_num}. Analysis", self.styles["SectionHeading"]))
        story.append(Spacer(1, 12))

        subsection = 1

        if "per_class_accuracy_plot" in analysis_data:
            story.append(
                Paragraph(
                    f"{chapter_num}.{subsection} Per-Class IoU",
                    self.styles["SubsectionHeading"],
                )
            )
            fig = analysis_data["per_class_accuracy_plot"]
            if figures_dir is None:
                import tempfile

                figures_dir = Path(tempfile.mkdtemp())
            img_path = figures_dir / "per_class_accuracy.png"
            fig.savefig(img_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

            img = Image(str(img_path), width=5 * inch, height=3 * inch)
            story.append(img)
            story.append(Spacer(1, 20))
            subsection += 1

        # Metrics correlation scatter plots
        if "metrics_correlation_plot" in analysis_data:
            story.append(
                Paragraph(
                    f"{chapter_num}.{subsection} Metrics Correlation",
                    self.styles["SubsectionHeading"],
                )
            )
            fig = analysis_data["metrics_correlation_plot"]
            if figures_dir is None:
                import tempfile

                figures_dir = Path(tempfile.mkdtemp())
            img_path = figures_dir / "metrics_correlation.png"
            fig.savefig(img_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

            img = Image(str(img_path), width=5 * inch, height=3 * inch)
            story.append(img)
            story.append(Spacer(1, 20))
            subsection += 1

        # Descriptive statistics table
        if "descriptive_stats" in analysis_data and analysis_data["descriptive_stats"]:
            story.append(
                Paragraph(
                    f"{chapter_num}.{subsection} Descriptive Statistics",
                    self.styles["SubsectionHeading"],
                )
            )

            stats = analysis_data["descriptive_stats"]
            headers = list(stats[0].keys())
            table_data = [headers]
            for row in stats:
                table_data.append([str(row[h]) for h in headers])

            table = Table(table_data)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 20))
            subsection += 1

        # Analysis plots
        if "plot_files" in analysis_data and analysis_data["plot_files"]:
            story.append(
                Paragraph(
                    f"{chapter_num}.{subsection} Analysis Plots",
                    self.styles["SubsectionHeading"],
                )
            )
            for plot_file in analysis_data["plot_files"]:
                try:
                    img = Image(str(plot_file), width=5 * inch, height=3 * inch)
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception as e:
                    logger.warning(f"Could not embed analysis plot {plot_file}: {e}")

    def _add_tests_section(self, story: list[Any], test_data: dict[str, Any], chapter_num: int) -> None:
        story.append(Paragraph(f"{chapter_num}. Tests", self.styles["SectionHeading"]))
        story.append(Spacer(1, 12))

        if "test_summary" in test_data:
            summary = test_data["test_summary"]
            summary_text = f"""
            <b>Total Tests:</b> {summary.get("total", 0)}<br/>
            <b>Passed:</b> {summary.get("passed", 0)}<br/>
            <b>Failed:</b> {summary.get("failed", 0)}<br/>
            <b>Skipped:</b> {summary.get("skipped", 0)}
            """
            story.append(Paragraph(summary_text, self.styles["Normal"]))

        story.append(Spacer(1, 20))
