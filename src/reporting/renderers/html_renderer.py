import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from jinja2 import Template

from src.constants import PLOT_DEFAULTS

from ..formatters import (
    dict_to_html_table,
    figure_to_html_img,
    format_duration,
    format_number,
    image_file_to_html_img,
    list_of_dicts_to_html_table,
)

logger = logging.getLogger(__name__)


# HTML template with embedded CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2E86AB;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 10px;
        }
        h2 {
            color: #2E86AB;
            margin-top: 30px;
            border-left: 4px solid #2E86AB;
            padding-left: 15px;
        }
        h3 {
            color: #555;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th {
            background-color: #F5F5F5;
            color: #333;
            font-weight: bold;
            text-align: left;
            padding: 12px;
            border: 1px solid #ddd;
        }
        td {
            padding: 10px;
            border: 1px solid #ddd;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f0f0f0;
        }
        .metadata {
            background-color: #f0f8ff;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .metadata p {
            margin: 5px 0;
        }
        .toc {
            background-color: #fff9e6;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }
        .toc li {
            padding: 5px 0;
        }
        .toc a {
            color: #2E86AB;
            text-decoration: none;
        }
        .toc a:hover {
            text-decoration: underline;
        }
        .section-placeholder {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #2E86AB;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2E86AB;
        }
        .user-specified {
            font-weight: bold;
        }
        .default-value {
            font-style: italic;
            color: #666;
        }
        .group-header td {
            background: #4A6FA5;
            color: white;
            font-weight: bold;
            font-size: 1.1em;
            padding: 6px 8px;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>

        <div class="metadata">
            <p><strong>Date:</strong> {{ date }}</p>
            <p><strong>Source:</strong> {{ source }}</p>
            {% if experiment_name %}
            <p><strong>Experiment:</strong> {{ experiment_name }}</p>
            {% endif %}
        </div>

        {% if toc_items %}
        <div class="toc">
            <h2>Table of Contents</h2>
            <ul>
            {% for item in toc_items %}
                <li><a href="#{{ item.anchor }}">{{ item.title }}</a></li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}

        {{ content | safe }}

        <hr style="margin-top: 50px;">
        <p style="text-align: center; color: #999; font-size: 12px;">
            Generated by DFC2020 Baseline Reporting Module
        </p>
    </div>
</body>
</html>
"""


class HTMLRenderer:
    def __init__(self, dpi: int = PLOT_DEFAULTS["FIGURE_DPI"]) -> None:
        self.dpi = dpi
        self.template = Template(HTML_TEMPLATE)

    def render(
        self,
        report_data: dict[str, Any],
        output_dir: Path,
        source: str,
        experiment_name: str,
    ) -> Path:
        """
        Render report to HTML file.

        Args:
            report_data: Collected report data
            output_dir: Directory where to save the report
            source: Report source (e.g., 'continuous_run', 'train')
            experiment_name: Name of the experiment

        Returns:
            Path to the generated HTML file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build TOC
        toc_items = []
        chapter_num = 1

        if "configuration" in report_data:
            toc_items.append({"anchor": f"chapter{chapter_num}", "title": f"{chapter_num}. Configuration"})
            chapter_num += 1
        if "training" in report_data:
            toc_items.append({"anchor": f"chapter{chapter_num}", "title": f"{chapter_num}. Training"})
            chapter_num += 1
        if "prediction" in report_data:
            toc_items.append({"anchor": f"chapter{chapter_num}", "title": f"{chapter_num}. Prediction"})
            chapter_num += 1
        if "analysis" in report_data:
            toc_items.append({"anchor": f"chapter{chapter_num}", "title": f"{chapter_num}. Analysis"})
            chapter_num += 1
        if "tests" in report_data:
            toc_items.append({"anchor": f"chapter{chapter_num}", "title": f"{chapter_num}. Tests"})
            chapter_num += 1

        # Build content sections
        content_parts = []
        chapter_num = 1

        if "configuration" in report_data:
            content_parts.append(self._render_configuration(report_data["configuration"], chapter_num))
            chapter_num += 1

        if "training" in report_data:
            content_parts.append(self._render_training(report_data["training"], chapter_num))
            chapter_num += 1

        if "prediction" in report_data:
            content_parts.append(self._render_prediction(report_data["prediction"], chapter_num))
            chapter_num += 1

        if "analysis" in report_data:
            content_parts.append(self._render_analysis(report_data["analysis"], chapter_num))
            chapter_num += 1

        if "tests" in report_data:
            content_parts.append(self._render_tests(report_data["tests"], chapter_num))
            chapter_num += 1

        # Render final HTML
        html_content = self.template.render(
            title=f"{source.replace('_', ' ').title()} Report",
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source=source,
            experiment_name=experiment_name,
            toc_items=toc_items,
            content="\n".join(content_parts),
        )

        # Write to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{source}_report_{experiment_name}_{timestamp}.html"
        output_path = output_dir / filename

        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report generated: {output_path}")
        return output_path

    def _render_configuration(self, config_data: dict[str, Any], chapter_num: int) -> str:
        """Render configuration section."""
        html = f'<h2 id="chapter{chapter_num}">{chapter_num}. Configuration</h2>\n'

        # General configuration (grouped)
        html += f"<h3>{chapter_num}.1 General Configuration</h3>\n"
        general_grouped = config_data.get("general_grouped")
        if general_grouped:
            html += '<table class="config-table">\n'
            html += "  <thead>\n"
            html += "    <tr><th>Parameter</th><th>Value</th></tr>\n"
            html += "  </thead>\n  <tbody>\n"

            for group_name, params in general_grouped.items():
                # Group header row
                html += f'    <tr class="group-header"><td colspan="2">{group_name}</td></tr>\n'
                # Parameter rows
                for param, info in params.items():
                    value = info["value"]
                    source = info["source"]
                    css_class = "user-specified" if source == "user" else "default-value"
                    html += f'    <tr><td>{param}</td><td><span class="{css_class}">{value}</span></td></tr>\n'

            html += "  </tbody>\n</table>\n"
            html += "<p><em>Bold = user-specified, Italic = default. Only non-default values shown (disabled groups collapsed).</em></p>\n"
        elif "general" in config_data:
            table_data = {}
            for param, info in config_data["general"].items():
                value = info["value"]
                source = info["source"]
                css_class = "user-specified" if source == "user" else "default-value"
                table_data[param] = f'<span class="{css_class}">{value}</span>'

            html += dict_to_html_table(table_data)
            html += "<p><em>Bold = user-specified, Italic = default</em></p>\n"

        # Model configuration
        html += f"<h3>{chapter_num}.2 Model Configuration</h3>\n"
        if "model" in config_data:
            table_data = {}
            for param, info in config_data["model"].items():
                value = info["value"]
                source = info["source"]
                css_class = "user-specified" if source == "user" else "default-value"
                table_data[param] = f'<span class="{css_class}">{value}</span>'

            html += dict_to_html_table(table_data)

        return html

    def _render_training(self, training_data: dict[str, Any], chapter_num: int) -> str:
        html = f'<h2 id="chapter{chapter_num}">{chapter_num}. Training</h2>\n'

        # Training summary
        html += f"<h3>{chapter_num}.1 Training Summary</h3>\n"

        if "final_metrics" in training_data:
            source = training_data.get("final_metrics_source", "validation")
            if source == "training":
                note = "Note: Only training metrics available (no validation was performed)"
                html += f"<p><em>{note}</em></p>\n"
            html += '<div class="metrics-grid">\n'
            metrics = training_data["final_metrics"]
            for metric_name, value in metrics.items():
                html += f"""
                <div class="metric-card">
                    <div class="metric-label">{metric_name}</div>
                    <div class="metric-value">{format_number(value)}</div>
                </div>
                """
            html += "</div>\n"

        if "best_metrics" in training_data:
            best = training_data["best_metrics"]
            miou_val = format_number(best["mIoU"])
            html += f"<p><strong>Best mIoU:</strong> {miou_val} (step {best['step']})</p>\n"

        if "training_duration" in training_data:
            duration = format_duration(training_data["training_duration"])
            html += f"<p><strong>Duration:</strong> {duration}</p>\n"

        # Loss curves
        loss_curves_shown = False
        if "loss_curves_file" in training_data:
            lc_path = training_data["loss_curves_file"]
            if Path(lc_path).exists():
                html += f"<h3>{chapter_num}.2 Loss Curves</h3>\n"
                html += image_file_to_html_img(lc_path)
                html += "\n"
                loss_curves_shown = True

        has_loss_plot = "plots" in training_data and "loss_curves" in training_data["plots"]
        if not loss_curves_shown and has_loss_plot:
            html += f"<h3>{chapter_num}.2 Loss Curves</h3>\n"
            fig = training_data["plots"]["loss_curves"]
            html += figure_to_html_img(fig, dpi=self.dpi)
            html += "\n"
            plt.close(fig)
        elif has_loss_plot:
            plt.close(training_data["plots"]["loss_curves"])

        # Metrics over time
        if "plots" in training_data and "metrics_over_time" in training_data["plots"]:
            html += f"<h3>{chapter_num}.3 Metrics Over Time</h3>\n"
            fig = training_data["plots"]["metrics_over_time"]
            html += figure_to_html_img(fig, dpi=self.dpi)
            html += "\n"
            plt.close(fig)

        return html

    def _render_prediction(self, prediction_data: dict[str, Any], chapter_num: int) -> str:
        html = f'<h2 id="chapter{chapter_num}">{chapter_num}. Prediction</h2>\n'

        # Prediction summary
        html += f"<h3>{chapter_num}.1 Prediction Summary</h3>\n"

        if "summary_metrics" in prediction_data:
            html += '<div class="metrics-grid">\n'
            summary = prediction_data["summary_metrics"]
            for metric, values in summary.items():
                if isinstance(values, dict) and "mean" in values:
                    html += f"""
                    <div class="metric-card">
                        <div class="metric-label">{metric}</div>
                        <div class="metric-value">{format_number(values["mean"])}</div>
                    </div>
                    """
            html += "</div>\n"

        # Confusion matrix
        if "confusion_matrix_plot" in prediction_data:
            html += f"<h3>{chapter_num}.2 Confusion Matrix</h3>\n"
            fig = prediction_data["confusion_matrix_plot"]
            html += figure_to_html_img(fig, dpi=self.dpi)
            html += "\n"
            plt.close(fig)

        # Per-class metrics
        if "per_class_metrics" in prediction_data:
            html += f"<h3>{chapter_num}.3 Per-Class Metrics</h3>\n"
            html += list_of_dicts_to_html_table(prediction_data["per_class_metrics"])
            html += "\n"

        # Metrics distribution
        if "metrics_distribution_plot" in prediction_data:
            html += f"<h3>{chapter_num}.4 Metrics Distribution</h3>\n"
            fig = prediction_data["metrics_distribution_plot"]
            html += figure_to_html_img(fig, dpi=self.dpi)
            html += "\n"
            plt.close(fig)

        # Sample predictions
        has_samples = "sample_prediction_plots" in prediction_data and prediction_data["sample_prediction_plots"]
        if has_samples:
            html += f"<h3>{chapter_num}.5 Sample Predictions</h3>\n"
            html += "<p><em>Random sample predictions showing input bands, ground truth, and model prediction</em></p>\n"
            for fig in prediction_data["sample_prediction_plots"]:
                html += figure_to_html_img(fig, dpi=self.dpi)
                html += "\n"
                plt.close(fig)

        return html

    def _render_analysis(self, analysis_data: dict[str, Any], chapter_num: int) -> str:
        html = f'<h2 id="chapter{chapter_num}">{chapter_num}. Analysis</h2>\n'

        subsection = 1

        if "per_class_accuracy_plot" in analysis_data:
            html += f"<h3>{chapter_num}.{subsection} Per-Class IoU</h3>\n"
            fig = analysis_data["per_class_accuracy_plot"]
            html += figure_to_html_img(fig, dpi=self.dpi)
            html += "\n"
            plt.close(fig)
            subsection += 1

        # Metrics correlation scatter plots
        if "metrics_correlation_plot" in analysis_data:
            html += f"<h3>{chapter_num}.{subsection} Metrics Correlation</h3>\n"
            fig = analysis_data["metrics_correlation_plot"]
            html += figure_to_html_img(fig, dpi=self.dpi)
            html += "\n"
            plt.close(fig)
            subsection += 1

        # Descriptive statistics table
        if "descriptive_stats" in analysis_data and analysis_data["descriptive_stats"]:
            html += f"<h3>{chapter_num}.{subsection} Descriptive Statistics</h3>\n"
            html += list_of_dicts_to_html_table(analysis_data["descriptive_stats"])
            html += "\n"
            subsection += 1

        # Plot files
        if "plot_files" in analysis_data and analysis_data["plot_files"]:
            html += f"<h3>{chapter_num}.{subsection} Analysis Plots</h3>\n"
            for plot_file in analysis_data["plot_files"]:
                html += image_file_to_html_img(plot_file)
                html += "\n"

        return html

    def _render_tests(self, test_data: dict[str, Any], chapter_num: int) -> str:
        html = f'<h2 id="chapter{chapter_num}">{chapter_num}. Tests</h2>\n'

        # Test summary
        if "test_summary" in test_data:
            html += f"<h3>{chapter_num}.1 Test Summary</h3>\n"
            html += '<div class="metrics-grid">\n'
            summary = test_data["test_summary"]
            for key, value in summary.items():
                if key == "duration":
                    value = format_duration(value)
                html += f"""
                <div class="metric-card">
                    <div class="metric-label">{key.replace("_", " ").title()}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """
            html += "</div>\n"

        # Test files
        if "test_files" in test_data:
            html += f"<h3>{chapter_num}.2 Test Files</h3>\n"
            table_data = [{"File": f["file"], "Exists": "✓" if f["exists"] else "✗"} for f in test_data["test_files"]]
            html += list_of_dicts_to_html_table(table_data)
            html += "\n"

        # Coverage
        if "coverage_available" in test_data and test_data["coverage_available"]:
            html += f"<h3>{chapter_num}.3 Coverage</h3>\n"
            html += "<p>Coverage report available in <code>htmlcov/index.html</code></p>\n"

        return html
