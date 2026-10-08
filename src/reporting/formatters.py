import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from src.constants import PLOT_DEFAULTS


def figure_to_base64(fig: Figure, format: str = "png", dpi: int = PLOT_DEFAULTS["FIGURE_DPI"]) -> str:
    """
    Convert a matplotlib figure to a base64-encoded string.

    Args:
        fig: Matplotlib figure to convert
        format: Image format (png, jpg, etc.)
        dpi: Resolution in dots per inch

    Returns:
        Base64-encoded string of the figure image
    """
    buffer = BytesIO()
    fig.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    return image_base64


def figure_to_html_img(fig: Figure, format: str = "png", dpi: int = PLOT_DEFAULTS["FIGURE_DPI"]) -> str:
    """
    Convert a matplotlib figure to an HTML img tag with embedded base64 data.

    Args:
        fig: Matplotlib figure to convert
        format: Image format (png, jpg, etc.)
        dpi: Resolution in dots per inch

    Returns:
        HTML img tag with embedded image data
    """
    img_base64 = figure_to_base64(fig, format=format, dpi=dpi)
    return f'<img src="data:image/{format};base64,{img_base64}" alt="Figure" />'


def image_file_to_base64(image_path: str | Path) -> str:
    """
    Convert an image file to a base64-encoded string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64-encoded string of the image
    """
    image_path = Path(image_path)
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    return image_base64


def image_file_to_html_img(image_path: str | Path) -> str:
    """
    Convert an image file to an HTML img tag with embedded base64 data.
    """
    image_path = Path(image_path)
    format = image_path.suffix.lstrip(".")
    img_base64 = image_file_to_base64(image_path)
    return f'<img src="data:image/{format};base64,{img_base64}" alt="{image_path.name}" />'


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string (e.g., "2h 15m 32s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_number(value: float, precision: int = 4) -> str:
    """
    Format a number with specified precision.

    Args:
        value: Number to format
        precision: Number of decimal places

    Returns:
        Formatted number string
    """
    return f"{value:.{precision}f}"


def dict_to_html_table(
    data: dict[str, Any],
    headers: tuple[str, str] = ("Parameter", "Value"),
    add_styling: bool = True,
) -> str:
    """
    Convert a dictionary to an HTML table.

    Args:
        data: Dictionary to convert
        headers: Tuple of (key_header, value_header)
        add_styling: Whether to add CSS classes for styling

    Returns:
        HTML table string
    """
    table_class = ' class="config-table"' if add_styling else ""
    html = f"<table{table_class}>\n"
    html += f"  <thead>\n    <tr><th>{headers[0]}</th><th>{headers[1]}</th></tr>\n  </thead>\n"
    html += "  <tbody>\n"

    for key, value in data.items():
        html += f"    <tr><td>{key}</td><td>{value}</td></tr>\n"

    html += "  </tbody>\n</table>"
    return html


def list_of_dicts_to_html_table(data: list[dict[str, Any]], add_styling: bool = True) -> str:
    """
    Convert a list of dictionaries to an HTML table.

    Args:
        data: List of dictionaries with same keys
        add_styling: Whether to add CSS classes for styling

    Returns:
        HTML table string
    """
    if not data:
        return "<p>No data available</p>"

    table_class = ' class="data-table"' if add_styling else ""
    headers = list(data[0].keys())

    html = f"<table{table_class}>\n"
    html += "  <thead>\n    <tr>"
    for header in headers:
        html += f"<th>{header}</th>"
    html += "</tr>\n  </thead>\n"
    html += "  <tbody>\n"

    for row in data:
        html += "    <tr>"
        for header in headers:
            value = row.get(header, "")
            html += f"<td>{value}</td>"
        html += "</tr>\n"

    html += "  </tbody>\n</table>"
    return html
