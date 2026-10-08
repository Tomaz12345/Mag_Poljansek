from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import cast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.errors
from matplotlib import colors

from src.constants import SENTINEL_BANDS, SPECIAL_VALUES, VIZ_CONFIG
from src.datasets import load_lc, load_s1, load_s2, mask_label_borders
from src.utils import classnames, mycmap

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/tmp/dfc_images")


def get_sorted_band_names(
    use_s2hr: bool,
    use_s2mr: bool,
    use_s2lr: bool,
    use_s1: bool,
) -> list[str]:
    """Get band names matching the sorted order that load_s2 returns.

    load_s2 sorts all selected S2 band numbers, so names must match that order.
    S1 bands (VV, VH) are always appended after S2.
    """
    s2_bands: list[int] = []
    if use_s2hr:
        s2_bands += list(SENTINEL_BANDS["S2_HR"])
    if use_s2mr:
        s2_bands += list(SENTINEL_BANDS["S2_MR"])
    if use_s2lr:
        s2_bands += list(SENTINEL_BANDS["S2_LR"])
    s2_bands = sorted(s2_bands)

    names = [f"S2 B{b}" for b in s2_bands]
    if use_s1:
        names.extend(["S1 VV", "S1 VH"])
    return names


def get_display_config(
    use_s2hr: bool,
    use_s2mr: bool,
    use_s2lr: bool,
    use_s1: bool,
) -> tuple[tuple[int, ...] | int, int]:
    """Get RGB display channels and brightness factor for visualization."""
    if use_s2hr and use_s2lr:
        return VIZ_CONFIG["RGB_S2HR_S2LR"], VIZ_CONFIG["BRIGHTNESS_S2HR"]
    elif use_s2hr:
        return VIZ_CONFIG["RGB_S2HR_ONLY"], VIZ_CONFIG["BRIGHTNESS_S2_ONLY"]
    elif use_s1 and not (use_s2hr or use_s2mr or use_s2lr):
        return VIZ_CONFIG["RGB_S1_ONLY"], VIZ_CONFIG["BRIGHTNESS_S1_ONLY"]
    else:
        return VIZ_CONFIG["RGB_DEFAULT"], 3


def create_sample_figure(
    image_bands: np.ndarray | None,
    band_names: list[str],
    display_channels: tuple[int, ...] | int,
    brightness_factor: int,
    gt_labels: np.ndarray | None,
    pred_labels_list: list[np.ndarray],
    pred_names: list[str],
    no_savanna: bool,
    title: str,
    border_exclusion_size: int | None = None,
) -> plt.Figure:
    """Create a sample visualization figure with bands, ground truth, and predictions.

    Layout:
    - Row 1: RGB composite + individual bands (S2 + S1)
    - Row 2: Ground Truth (colorized) + [GT with border exclusion] + Prediction(s)

    Args:
        image_bands: Array of shape [C, H, W] with all loaded bands, or None.
        band_names: Human-readable names for each band channel.
        display_channels: Channel indices for RGB composite (tuple of 3) or single int.
        brightness_factor: Brightness multiplier for RGB composite.
        gt_labels: Ground truth labels (1-indexed DFC, 255=ignore), or None.
        pred_labels_list: List of prediction label arrays (1-indexed DFC, 255=ignore).
        pred_names: Display names for each prediction (same length as pred_labels_list).
        no_savanna: Whether savanna class is excluded.
        title: Figure title.
        border_exclusion_size: If set, show an extra subplot with border-excluded GT
            (excluded pixels shown in red).

    Returns:
        Matplotlib Figure.
    """
    cmap = mycmap()
    class_names_list = classnames()
    if no_savanna:
        class_names_list = [c for c in class_names_list if c != "Savanna"]
    num_classes = len(class_names_list)

    n_colors = cmap.N  # always 10
    # When no_savanna, class i in the 9-class list maps to original cmap index:
    #   i < 2  → index i  (Forest, Shrubland unchanged)
    #   i >= 2 → index i+1  (skip Savanna at index 2)
    original_indices = [i if i < 2 else i + 1 for i in range(num_classes)] if no_savanna else list(range(num_classes))
    legend_patches = [
        mpatches.Patch(
            color=cast(
                "tuple[float, float, float]",
                tuple(cmap(original_indices[i] / n_colors)[:3]),
            ),
            label=class_names_list[i],
        )
        for i in range(num_classes)
    ]

    show_border = border_exclusion_size is not None and gt_labels is not None

    if show_border:
        legend_patches.append(mpatches.Patch(color=(0.0, 0.0, 0.0), label="Excluded (border)"))

    # Determine subplot layout
    n_bands = len(band_names) if image_bands is not None else 0
    has_rgb = isinstance(display_channels, tuple) and n_bands >= max(display_channels) + 1

    n_cols_row1 = (1 if has_rgb else 0) + n_bands
    n_cols_row2 = (1 if gt_labels is not None else 0) + (1 if show_border else 0) + len(pred_labels_list)
    n_cols = max(n_cols_row1, n_cols_row2, 1)

    fig_width = min(3 * n_cols, 24)
    fig, axes = plt.subplots(2, n_cols, figsize=(fig_width, 6))

    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(title, fontsize=11, fontweight="bold")

    # Row 1: RGB composite + individual bands
    col_idx = 0
    if has_rgb and image_bands is not None:
        ax = axes[0, col_idx]
        dc = cast("tuple[int, ...]", display_channels)
        rgb = image_bands[np.array(list(dc)), :, :]  # [3, H, W]
        rgb = np.transpose(rgb, (1, 2, 0))  # [H, W, 3]
        rgb = np.clip(rgb * brightness_factor, 0, 1)
        ax.imshow(rgb)
        ax.set_title("RGB Composite", fontsize=9)
        ax.axis("off")
        col_idx += 1

    if image_bands is not None:
        for band_idx, band_name in enumerate(band_names):
            if col_idx >= n_cols:
                break
            ax = axes[0, col_idx]
            band_data = image_bands[band_idx]
            ax.imshow(band_data, cmap="gray")
            ax.set_title(band_name, fontsize=9)
            ax.axis("off")
            col_idx += 1

    # Hide remaining axes in row 1
    for ci in range(col_idx, n_cols):
        axes[0, ci].axis("off")

    # Row 2: Ground Truth + [Border Exclusion] + Prediction(s)
    col_idx = 0

    if gt_labels is not None:
        ax = axes[1, col_idx]
        gt_colored = _colorize_labels(gt_labels, cmap, num_classes, no_savanna)
        ax.imshow(gt_colored)
        ax.set_title("Ground Truth", fontsize=9)
        ax.axis("off")
        col_idx += 1

    if show_border:
        assert gt_labels is not None  # for type narrowing
        assert border_exclusion_size is not None
        ax = axes[1, col_idx]
        # Convert 1-indexed GT to 0-indexed for mask_label_borders, then back
        ignore_val = SPECIAL_VALUES["IGNORE_LABEL"]
        gt_zero = gt_labels.copy()
        gt_zero[gt_labels != ignore_val] = gt_labels[gt_labels != ignore_val] - 1
        masked_zero = mask_label_borders(gt_zero, border_exclusion_size, ignore_val)
        # Pixels that became ignore due to border exclusion
        border_mask = (masked_zero == ignore_val) & (gt_zero != ignore_val)
        # Colorize original labels, then paint excluded pixels black
        gt_colored_excl = _colorize_labels(gt_labels, cmap, num_classes, no_savanna)
        gt_colored_excl[border_mask] = [0.0, 0.0, 0.0]
        ax.imshow(gt_colored_excl)
        ax.set_title(f"GT + Border Excl. (r={border_exclusion_size})", fontsize=9)
        ax.axis("off")
        col_idx += 1

    for pred_labels, pred_name in zip(pred_labels_list, pred_names, strict=True):
        if col_idx >= n_cols:
            break
        ax = axes[1, col_idx]
        pred_colored = _colorize_labels(pred_labels, cmap, num_classes, no_savanna)
        ax.imshow(pred_colored)
        ax.set_title(pred_name, fontsize=9)
        ax.axis("off")
        col_idx += 1

    # Hide remaining axes in row 2
    for ci in range(col_idx, n_cols):
        axes[1, ci].axis("off")

    # Add legend below the figure
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=min(len(legend_patches), 6),
        fontsize=7,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig


def _colorize_labels(
    labels: np.ndarray,
    cmap: colors.ListedColormap,
    num_classes: int,
    no_savanna: bool = False,
) -> np.ndarray:
    """Convert 1-indexed label array to RGB image. 255 = ignore → white.

    When no_savanna=True, labels 3–9 are shifted (Savanna removed), so they
    must map to the original cmap color slots 3–9 (skipping slot 2 = Savanna).
    """
    n_colors = cmap.N  # always 10
    # When no_savanna: labels 1–2 → cmap indices 0–1; labels 3–9 → cmap indices 3–9
    color_idx = np.where((labels >= 3) & (labels != 255), labels, labels - 1).astype(float) if no_savanna else (labels - 1).astype(float)
    color_idx[labels == 255] = 0  # placeholder; overwritten below
    colored = cmap(color_idx / n_colors)[:, :, :3]
    colored[labels == 255] = 1.0  # white
    return colored


def resolve_data_paths(
    element_id: str,
    data_dir: str,
) -> dict[str, str]:
    """Resolve raw data file paths (s2, s1, lc, dfc) from element_id.

    Args:
        element_id: Element filename without extension
            (e.g., 'ROIs1158_spring_lc_146_p202' or 'ROIs0000_test_dfc_0_p42').
        data_dir: Root data directory.

    Returns:
        Dictionary with 's2', 's1', 'lc', 'dfc' file paths.
    """
    basename = element_id.replace(".tif", "")

    s2_name = basename.replace("_lc_", "_s2_").replace("_dfc_", "_s2_")
    s1_name = basename.replace("_lc_", "_s1_").replace("_dfc_", "_s1_")

    parts = s2_name.rsplit("_", 2)  # ['ROIs1158_spring_s2', '146', 'p202']
    if len(parts) >= 3:
        prefix_parts = parts[0].rsplit("_", 1)  # ['ROIs1158_spring', 's2']
        if len(prefix_parts) == 2:
            season_folder = prefix_parts[0]
            sensor = prefix_parts[1]
            num = parts[1]
            subfolder = f"{sensor}_{num}"

            s2_path = str(Path(data_dir) / season_folder / subfolder / f"{s2_name}.tif")
            s1_subfolder = subfolder.replace("s2_", "s1_")
            s1_path = str(Path(data_dir) / season_folder / s1_subfolder / f"{s1_name}.tif")

            lc_subfolder = subfolder.replace("s2_", "lc_")
            lc_name = basename.replace("_dfc_", "_lc_")
            lc_path = str(Path(data_dir) / season_folder / lc_subfolder / f"{lc_name}.tif")

            dfc_subfolder = subfolder.replace("s2_", "dfc_")
            dfc_name = basename.replace("_lc_", "_dfc_")
            dfc_path = str(Path(data_dir) / season_folder / dfc_subfolder / f"{dfc_name}.tif")

            return {"s2": s2_path, "s1": s1_path, "lc": lc_path, "dfc": dfc_path}

    logger.warning(f"Could not parse element_id pattern for {element_id}")
    return {"s2": "", "s1": "", "lc": "", "dfc": ""}


def load_bands_and_gt(
    element_id: str,
    data_dir: str,
    use_s2hr: bool,
    use_s2mr: bool,
    use_s2lr: bool,
    use_s1: bool,
    no_savanna: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load image bands and ground truth labels for an element.

    Returns:
        Tuple of (image_bands [C, H, W] or None, gt_labels [H, W] 1-indexed or None).
    """
    paths = resolve_data_paths(element_id, data_dir)
    use_s2 = use_s2hr or use_s2mr or use_s2lr

    # Load S2 bands
    image_bands: np.ndarray | None = None
    if use_s2 and paths["s2"] and Path(paths["s2"]).exists():
        try:
            image_bands = load_s2(paths["s2"], use_s2hr, use_s2mr, use_s2lr)
        except Exception as e:
            logger.warning(f"Failed to load S2 data for {element_id}: {e}")

    # Load S1 bands
    if use_s1 and paths["s1"] and Path(paths["s1"]).exists():
        try:
            s1_data = load_s1(paths["s1"])
            image_bands = np.concatenate([image_bands, s1_data], axis=0) if image_bands is not None else s1_data
        except Exception as e:
            logger.warning(f"Failed to load S1 data for {element_id}: {e}")

    # Load ground truth
    # For DFC2020 elements (_dfc_ in ID), load from the dfc file with igbp=False.
    # For SEN12MS elements (_lc_ in ID), load from the lc file with igbp=True.
    gt_labels = None
    is_dfc = "_dfc_" in element_id
    gt_path = paths["dfc"] if is_dfc else paths["lc"]
    if gt_path and Path(gt_path).exists():
        try:
            gt_labels_zero = load_lc(gt_path, no_savanna=no_savanna, igbp=not is_dfc)
            # Convert zero-based back to 1-indexed DFC for colorization
            gt_labels = gt_labels_zero.copy()
            gt_labels[gt_labels_zero != 255] = gt_labels_zero[gt_labels_zero != 255] + 1
        except Exception as e:
            logger.warning(f"Failed to load ground truth for {element_id}: {e}")

    return image_bands, gt_labels


def load_prediction(path: str | Path) -> np.ndarray:
    """Load prediction labels from a TIFF file.

    Returns:
        Label array [H, W], 1-indexed (uint8).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(str(path)) as src:
            return cast(np.ndarray, src.read(1))


def _next_output_path(element_id: str) -> Path:
    """Find next available /tmp/dfc_images/<element_id>_{i}.png path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    basename = element_id.replace(".tif", "")
    i = 0
    while True:
        path = OUTPUT_DIR / f"{basename}_{i}.png"
        if not path.exists():
            return path
        i += 1


def show_and_save(
    element_id: str,
    data_dir: str,
    use_s2hr: bool = True,
    use_s2mr: bool = True,
    use_s2lr: bool = True,
    use_s1: bool = True,
    no_savanna: bool = False,
    prediction_paths: list[str | Path] | None = None,
    border_exclusion_size: int | None = None,
) -> Path:
    """Load data, create visualization figure, and save to /tmp/dfc_images/.

    Args:
        element_id: Element filename (with or without .tif extension).
        data_dir: Root data directory.
        use_s2hr: Use S2 high-resolution bands.
        use_s2mr: Use S2 medium-resolution bands.
        use_s2lr: Use S2 low-resolution bands.
        use_s1: Use S1 bands.
        no_savanna: Exclude savanna class.
        prediction_paths: Optional list of prediction TIFF paths.
        border_exclusion_size: If set, show border exclusion overlay on GT.

    Returns:
        Path to the saved PNG file.
    """
    image_bands, gt_labels = load_bands_and_gt(
        element_id,
        data_dir,
        use_s2hr,
        use_s2mr,
        use_s2lr,
        use_s1,
        no_savanna,
    )

    band_names = get_sorted_band_names(use_s2hr, use_s2mr, use_s2lr, use_s1)
    display_channels, brightness_factor = get_display_config(use_s2hr, use_s2mr, use_s2lr, use_s1)

    # Load predictions
    pred_labels_list: list[np.ndarray] = []
    pred_names: list[str] = []
    if prediction_paths:
        for pred_path in prediction_paths:
            pred_path = Path(pred_path)
            if pred_path.exists():
                pred_labels_list.append(load_prediction(pred_path))
                pred_names.append(pred_path.parent.name or pred_path.stem)
            else:
                logger.warning(f"Prediction file not found: {pred_path}")

    title = element_id.replace(".tif", "")
    fig = create_sample_figure(
        image_bands=image_bands,
        band_names=band_names,
        display_channels=display_channels,
        brightness_factor=brightness_factor,
        gt_labels=gt_labels,
        pred_labels_list=pred_labels_list,
        pred_names=pred_names,
        no_savanna=no_savanna,
        title=title,
        border_exclusion_size=border_exclusion_size,
    )

    output_path = _next_output_path(element_id)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")
    return output_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize sample bands, ground truth, and predictions.",
    )
    parser.add_argument("--element_id", required=True, help="Element ID (e.g., ROIs1158_spring_lc_146_p202)")
    parser.add_argument("--data_dir", required=True, help="Root data directory")
    parser.add_argument("--prediction_paths", nargs="*", default=None, help="Prediction TIFF file paths")
    parser.add_argument("--use_s2hr", type=bool, default=True, help="Use S2 HR bands (default: True)")
    parser.add_argument("--use_s2mr", type=bool, default=True, help="Use S2 MR bands (default: True)")
    parser.add_argument("--use_s2lr", type=bool, default=True, help="Use S2 LR bands (default: True)")
    parser.add_argument("--use_s1", type=bool, default=True, help="Use S1 bands (default: True)")
    parser.add_argument("--no_savanna", action="store_true", help="Exclude savanna class")
    parser.add_argument(
        "--border_exclusion_size",
        type=int,
        default=None,
        help="Show border exclusion overlay with this radius (default: None = off)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_path = show_and_save(
        element_id=args.element_id,
        data_dir=args.data_dir,
        use_s2hr=args.use_s2hr,
        use_s2mr=args.use_s2mr,
        use_s2lr=args.use_s2lr,
        use_s1=args.use_s1,
        no_savanna=args.no_savanna,
        prediction_paths=args.prediction_paths,
        border_exclusion_size=args.border_exclusion_size,
    )
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
