import glob
import json
import logging
import os
from typing import Any, cast

import numpy as np
import pandas as pd
import rasterio
import torch.utils.data as data
from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure
from tqdm import tqdm

from .constants import (
    CLASS_MAPPING,
    DATASET_PATHS,
    NORM_CONFIG,
    S2_BANDS_HR,
    S2_BANDS_LR,
    S2_BANDS_MR,
    SEN12MS_SUBSETS,
    VIZ_CONFIG,
    get_sen12ms_dataset_size,
)
from .messages import DATASET_ERRORS
from .timing_utils import timeit
from .utils import get_dfc2020_subset, is_dfc2020_dataset, log_and_raise, validate_string_value

logger = logging.getLogger(__name__)


class BaseDataset(data.Dataset[dict[str, Any]]):
    # Subclasses must define samples list
    samples: list[dict[str, str]]

    def __init__(
        self,
        use_s2hr: bool = False,
        use_s2mr: bool = False,
        use_s2lr: bool = False,
        use_s1: bool = False,
        no_savanna: bool = False,
        norm_stats_path: str = DATASET_PATHS["NORMALIZATION_STATS_FILE"],
        normalization_mode: str = "global",
    ) -> None:
        """Initialize common dataset parameters.

        Args:
            use_s2hr: Use Sentinel-2 high-resolution bands
            use_s2mr: Use Sentinel-2 medium-resolution bands
            use_s2lr: Use Sentinel-2 low-resolution bands
            use_s1: Use Sentinel-1 bands
            no_savanna: Exclude savanna class
            norm_stats_path: Path to normalization statistics JSON
            normalization_mode: Normalization strategy (global, per_scene, histogram_eq, none)
        """
        # Validate at least one input is specified
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            log_and_raise(ValueError, "No input specified, set at least one of use_[s2hr, s2mr, s2lr, s1] to True!", logger_instance=logger)

        # Store sensor flags
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s1 = use_s1
        self.no_savanna = no_savanna

        self.normalization_mode = normalization_mode
        self.norm_stats: dict[str, np.ndarray] | None

        # Load normalization stats (only needed for GLOBAL mode)
        if normalization_mode == "global":
            self.norm_stats = load_normalization_stats(norm_stats_path, "SEN12MS-train")
        else:
            self.norm_stats = None

        # Calculate number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr)

        # Calculate display channels for previewing
        self.display_channels, self.brightness_factor = get_display_channels(use_s2hr, use_s2mr, use_s2lr)

        # Calculate number of classes
        if no_savanna:
            self.n_classes = max(DFC2020_CLASSES) - 1
        else:
            self.n_classes = max(DFC2020_CLASSES)


DFC2020_CLASSES = list(CLASS_MAPPING["IGBP_TO_DFC2020"])


def load_normalization_stats(json_path: str, dataset_name: str = "SEN12MS-train") -> dict[str, np.ndarray] | None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_full_path = os.path.join(script_dir, json_path)
    with open(json_full_path) as f:
        all_stats = json.load(f)

    for entry in all_stats:
        if entry["name"] == dataset_name:
            stats = {
                "mean": np.array(entry["global_mean_per_channel"], dtype=np.float32),
                "std": np.array(entry["global_std_per_channel"], dtype=np.float32),
                "p2": np.array(entry["global_2_percentile_per_channel"], dtype=np.float32),
                "p98": np.array(entry["global_98_percentile_per_channel"], dtype=np.float32),
            }
            return cast(dict[str, np.ndarray], stats)

    log_and_raise(ValueError, f"Dataset '{dataset_name}' not found in {json_path}", logger_instance=logger)


@timeit("datasets/normalize_sample", log_args=True, component="datasets", enable_timing=False)
def normalize_sample(
    img: np.ndarray,
    s2_band_indices: list[int],
    use_s1: bool,
    stats: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Normalize image with S2 (clip + min-max) and S1 (z-score) using global stats.

    Args:
        img: Image array of shape [C, H, W]
        s2_band_indices: 0-based indices of S2 bands in the stats arrays
        use_s1: Whether S1 channels are present
        stats: Dict with 'mean', 'std', 'p2', 'p98' arrays

    Returns:
        Normalized image array
    """
    img = img.copy()
    n_s2_channels = len(s2_band_indices)

    # S2 normalization: percentile clip + min-max
    if n_s2_channels > 0:
        s2 = img[:n_s2_channels]
        # Get stats for the specific S2 bands used
        p2 = stats["p2"][s2_band_indices][:, None, None]
        p98 = stats["p98"][s2_band_indices][:, None, None]

        s2 = np.clip(s2, p2, p98)
        s2 = (s2 - p2) / (p98 - p2 + NORM_CONFIG["EPSILON"])

        img[:n_s2_channels] = s2

    # S1 normalization: z-score (S1 stats are always indices 13, 14)
    if use_s1:
        s1 = img[n_s2_channels:]
        s1_indices = [13, 14]
        mean = stats["mean"][s1_indices][:, None, None]
        std = stats["std"][s1_indices][:, None, None]

        s1 = (s1 - mean) / (std + NORM_CONFIG["EPSILON"])

        img[n_s2_channels:] = s1

    return img


def normalize_per_scene(img: np.ndarray, n_s2_channels: int) -> np.ndarray:
    """
    Normalize each channel to [0, 1] based on scene-level min/max.

    This normalizes each channel independently using its own statistics,
    which can be useful when global statistics don't represent the scene well.

    Args:
        img: Image array of shape [C, H, W]
        n_s2_channels: Number of S2 channels (for reference, but normalize all)

    Returns:
        Normalized image array with values in [0, 1]
    """
    img = img.copy()

    for c in range(img.shape[0]):
        channel = img[c]
        c_min = channel.min()
        c_max = channel.max()

        if c_max - c_min > NORM_CONFIG["EPSILON"]:
            img[c] = (channel - c_min) / (c_max - c_min)
        else:
            # Constant channel - set to 0
            img[c] = 0.0

    return img


def normalize_histogram_equalization(img: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """
    Apply histogram equalization to each channel independently.

    This enhances contrast by redistributing pixel values to use the full
    dynamic range more evenly.

    Args:
        img: Image array of shape [C, H, W]
        n_bins: Number of bins for histogram (default: 256)

    Returns:
        Histogram-equalized image array with values in [0, 1]
    """
    img = img.copy()

    for c in range(img.shape[0]):
        channel = img[c]

        # Compute histogram
        c_min, c_max = channel.min(), channel.max()

        if c_max - c_min < NORM_CONFIG["EPSILON"]:
            # Constant channel - set to 0
            img[c] = 0.0
            continue

        # Normalize to [0, 1] first
        channel_norm = (channel - c_min) / (c_max - c_min)

        # Compute histogram
        hist, bin_edges = np.histogram(channel_norm.flatten(), bins=n_bins, range=(0, 1))

        # Compute CDF
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]  # Normalize to [0, 1]

        # Map pixel values through CDF
        bin_indices = np.clip((channel_norm * (n_bins - 1)).astype(np.int32), 0, n_bins - 1)
        img[c] = cdf[bin_indices]

    return img


def apply_normalization(
    img: np.ndarray,
    mode: str,
    s2_band_indices: list[int],
    use_s1: bool,
    stats: dict[str, np.ndarray] | None,
) -> np.ndarray:
    """
    Apply normalization based on the specified mode.

    Args:
        img: Image array of shape [C, H, W]
        mode: Normalization mode (global, per_scene, histogram_eq, none)
        s2_band_indices: 0-based indices of S2 bands (for global mode)
        use_s1: Whether S1 channels are present (for global mode)
        stats: Global normalization stats (required for global mode)

    Returns:
        Normalized image array
    """
    if mode == "none":
        return img

    if mode == "global":
        if stats is None:
            log_and_raise(ValueError, "Global normalization requires stats to be provided", logger_instance=logger)
        return cast(np.ndarray, normalize_sample(img, s2_band_indices, use_s1, stats))

    if mode == "per_scene":
        n_s2_channels = len(s2_band_indices)
        return normalize_per_scene(img, n_s2_channels)

    if mode == "histogram_eq":
        return normalize_histogram_equalization(img)

    # Should never reach here, but handle gracefully
    return img


def get_s2_band_indices(use_s2hr: bool, use_s2mr: bool, use_s2lr: bool) -> list[int]:
    """Get the 0-based indices of selected S2 bands in the stats array."""
    bands_selected: list[int] = []
    if use_s2hr:
        bands_selected = bands_selected + list(S2_BANDS_HR)
    if use_s2mr:
        bands_selected = bands_selected + list(S2_BANDS_MR)
    if use_s2lr:
        bands_selected = bands_selected + list(S2_BANDS_LR)
    bands_selected = sorted(bands_selected)
    # Convert to 0-based indices (rasterio bands are 1-indexed)
    return [b - 1 for b in bands_selected]


@timeit("datasets/load_s2", log_args=True, component="datasets", enable_timing=False)
def load_s2(path: str, use_hr: bool, use_mr: bool, use_lr: bool) -> np.ndarray:
    """Load Sentinel-2 data from a TIFF file.

    Args:
        path: Path to the S2 TIFF file.
        use_hr: Use high-resolution bands (10m).
        use_mr: Use medium-resolution bands (20m).
        use_lr: Use low-resolution bands (60m).

    Returns:
        S2 image array of shape [C, H, W] as float32, clipped and scaled.
    """
    bands_selected: list[int] = []
    if use_hr:
        bands_selected = bands_selected + list(S2_BANDS_HR)
    if use_mr:
        bands_selected = bands_selected + list(S2_BANDS_MR)
    if use_lr:
        bands_selected = bands_selected + list(S2_BANDS_LR)
    bands_selected = sorted(bands_selected)
    with rasterio.open(path) as data:
        s2 = data.read(bands_selected)
    s2 = cast(np.ndarray, s2.astype(np.float32, copy=False))
    s2 = np.clip(s2, 0, 10000)
    s2 /= 10000
    return s2.astype(np.float32, copy=False)


@timeit("datasets/load_s1", log_args=True, component="datasets", enable_timing=False)
def load_s1(path: str) -> np.ndarray:
    """Load Sentinel-1 data from a TIFF file.

    Args:
        path: Path to the S1 TIFF file.

    Returns:
        S1 image array of shape [C, H, W] as float32, clipped and normalized.
    """
    with rasterio.open(path) as data:
        s1 = data.read()
    s1 = cast(np.ndarray, s1.astype(np.float32, copy=False))
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    s1 /= 25
    s1 += 1
    return cast(np.ndarray, s1.astype(np.float32, copy=False))


def load_lc(path: str, no_savanna: bool = False, igbp: bool = True) -> np.ndarray:
    """Load land cover labels from a TIFF file.

    Args:
        path: Path to the label TIFF file.
        no_savanna: Exclude savanna class (shifts class indices).
        igbp: Convert from IGBP to DFC2020 class scheme.

    Returns:
        Label array of shape [H, W] with zero-based class indices (255 = ignore).
    """
    # load labels
    with rasterio.open(path) as data:
        lc = data.read(1)

    lc_arr = np.take(DFC2020_CLASSES, lc) if igbp else lc.astype(np.int64, copy=False)
    lc_arr = cast(np.ndarray, lc_arr)

    # adjust class scheme to ignore class savanna
    if no_savanna:
        lc_arr[lc_arr == 3] = 0
        mask = lc_arr > 3
        lc_arr[mask] = lc_arr[mask] - 1

    # convert to zero-based labels and set ignore mask
    lc_arr = lc_arr - 1
    lc_arr[lc_arr == -1] = 255
    return lc_arr


def mask_label_borders(
    label: np.ndarray,
    border_size: int,
    ignore_value: int = 255,
) -> np.ndarray:
    """Mask pixels near class boundaries by setting them to ignore_value.

    Detects boundaries where adjacent pixels belong to different classes,
    then dilates those boundaries by ``border_size`` pixels using a disk-like
    structuring element.  Only valid (non-ignore) pixels are considered when
    detecting boundaries, and only valid pixels are masked.

    Args:
        label: Label array of shape [H, W] with integer class indices.
        border_size: Dilation radius in pixels for the exclusion zone.
        ignore_value: Value used to mark ignored pixels (default 255).

    Returns:
        Copy of *label* with border pixels set to *ignore_value*.
    """
    valid = label != ignore_value

    # Detect boundary pixels: any valid pixel whose 4-connected neighbor
    # is a different valid class.
    boundary = np.zeros_like(label, dtype=bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        shifted = np.roll(label, (-dy, -dx), axis=(0, 1))
        shifted_valid = np.roll(valid, (-dy, -dx), axis=(0, 1))
        boundary |= (label != shifted) & valid & shifted_valid

    # Clear edge artifacts introduced by np.roll wrapping.
    if boundary.shape[0] > 0:
        boundary[0, :] = False
        boundary[-1, :] = False
    if boundary.shape[1] > 0:
        boundary[:, 0] = False
        boundary[:, -1] = False

    # Dilate boundaries using a disk-like structuring element.
    # generate_binary_structure(2, 1) gives a cross; iterating it
    # ``border_size`` times approximates a diamond/disk of that radius.
    struct = iterate_structure(generate_binary_structure(2, 1), border_size)
    dilated = binary_dilation(boundary, structure=struct)

    result = label.copy()
    result[dilated & valid] = ignore_value
    return result


def get_ninputs(use_s1: bool, use_s2hr: bool, use_s2mr: bool, use_s2lr: bool) -> int:
    """Calculate the total number of input channels based on sensor flags.

    Args:
        use_s1: Use Sentinel-1 bands.
        use_s2hr: Use Sentinel-2 high-resolution bands.
        use_s2mr: Use Sentinel-2 medium-resolution bands.
        use_s2lr: Use Sentinel-2 low-resolution bands.

    Returns:
        Total number of input channels.
    """
    n_inputs = 0
    if use_s2hr:
        n_inputs += len(S2_BANDS_HR)
    if use_s2mr:
        n_inputs += len(S2_BANDS_MR)
    if use_s2lr:
        n_inputs += len(S2_BANDS_LR)
    if use_s1:
        n_inputs += 2
    return n_inputs


def load_sample(
    sample: dict[str, str],
    use_s1: bool,
    use_s2hr: bool,
    use_s2mr: bool,
    use_s2lr: bool,
    no_savanna: bool = False,
    igbp: bool = True,
    unlabeled: bool = False,
    norm_stats: dict[str, np.ndarray] | None = None,
    normalization_mode: str = "global",
) -> dict[str, Any]:
    """Load and assemble a complete sample from S1/S2/label TIFF files.

    Args:
        sample: Dict with 's1', 's2', 'lc' (optional), and 'id' file paths.
        use_s1: Use Sentinel-1 bands.
        use_s2hr: Use Sentinel-2 high-resolution bands.
        use_s2mr: Use Sentinel-2 medium-resolution bands.
        use_s2lr: Use Sentinel-2 low-resolution bands.
        no_savanna: Exclude savanna class.
        igbp: Convert labels from IGBP to DFC2020 class scheme.
        unlabeled: If True, skip loading labels.
        norm_stats: Global normalization statistics.
        normalization_mode: Normalization strategy.

    Returns:
        Dict with 'image' array, 'id' string, and optionally 'label' array.
    """
    use_s2 = use_s2hr or use_s2mr or use_s2lr

    # load s2 data
    if use_s2:
        img: np.ndarray = load_s2(sample["s2"], use_s2hr, use_s2mr, use_s2lr)

    # load s1 data
    if use_s1:
        img = np.concatenate((img, load_s1(sample["s1"])), axis=0) if use_s2 else load_s1(sample["s1"])

    # Load label if available
    lc: np.ndarray | None = None
    if not unlabeled:
        lc = load_lc(sample["lc"], no_savanna=no_savanna, igbp=igbp)

    # apply normalization based on mode
    s2_band_indices = get_s2_band_indices(use_s2hr, use_s2mr, use_s2lr)
    img = apply_normalization(img, normalization_mode, s2_band_indices, use_s1, norm_stats)

    # Build result dictionary
    result: dict[str, Any] = {"image": img, "id": sample["id"]}

    # Add label if present
    if lc is not None:
        result["label"] = lc

    return result


def get_display_channels(
    use_s2hr: bool,
    use_s2mr: bool,
    use_s2lr: bool,
) -> tuple[tuple[int, ...] | int, int]:
    """Select channels and brightness factor for RGB preview images.

    Args:
        use_s2hr: Use Sentinel-2 high-resolution bands.
        use_s2mr: Use Sentinel-2 medium-resolution bands.
        use_s2lr: Use Sentinel-2 low-resolution bands.

    Returns:
        Tuple of (display_channels, brightness_factor).
    """
    if use_s2hr and use_s2lr:
        display_channels: tuple[int, ...] | int = VIZ_CONFIG["RGB_S2HR_S2LR"]
        brightness_factor = VIZ_CONFIG["BRIGHTNESS_S2HR"]
    elif use_s2hr:
        display_channels = VIZ_CONFIG["RGB_S2HR_ONLY"]
        brightness_factor = VIZ_CONFIG["BRIGHTNESS_S2HR"]
    elif not (use_s2hr or use_s2mr or use_s2lr):
        display_channels = VIZ_CONFIG["RGB_S1_ONLY"]
        brightness_factor = VIZ_CONFIG["BRIGHTNESS_S1_ONLY"]
    else:
        display_channels = VIZ_CONFIG["RGB_DEFAULT"]
        brightness_factor = VIZ_CONFIG["BRIGHTNESS_S2_ONLY"]
    return (display_channels, brightness_factor)


class SEN12MS(BaseDataset):
    """PyTorch dataset class for the SEN12MS dataset.

    Expects dataset directory structure::

        <root>/
            SEN12MS_holdOutScenes.txt
            ROIsxxxx_y/
                lc_n/
                s1_n/
                s2_n/

    ``SEN12MS_holdOutScenes.txt`` contains the subdirs for the official
    train/val split and can be obtained from
    https://github.com/MSchmitt1984/SEN12MS/blob/master/splits.
    """

    @timeit("datasets/init_SEN12MS", log_args=True, component="datasets", enable_timing=True)
    def __init__(
        self,
        path: str,
        subset: str = "train",
        no_savanna: bool = False,
        use_s2hr: bool = False,
        use_s2mr: bool = False,
        use_s2lr: bool = False,
        use_s1: bool = False,
        norm_stats_path: str = DATASET_PATHS["NORMALIZATION_STATS_FILE"],
        subset_indices: list[int] | None = None,
        normalization_mode: str = "global",
        border_exclusion: bool = False,
        border_exclusion_size: int = 10,
    ) -> None:
        """Initialize the dataset.

        Args:
            path: Path to dataset root directory
            subset: Dataset subset ('train' or 'holdout')
            no_savanna: Whether to exclude savanna class
            use_s2hr: Use Sentinel-2 high-resolution bands
            use_s2mr: Use Sentinel-2 medium-resolution bands
            use_s2lr: Use Sentinel-2 low-resolution bands
            use_s1: Use Sentinel-1 bands
            norm_stats_path: Path to normalization statistics JSON
            subset_indices: Optional list of indices to use as subset.
                           If provided, only these samples will be used.
            normalization_mode: Normalization strategy (global, per_scene, histogram_eq, none)
            border_exclusion: Mask pixels near class boundaries as ignore.
            border_exclusion_size: Dilation radius (pixels) for border exclusion.
        """
        self.border_exclusion = border_exclusion
        self.border_exclusion_size = border_exclusion_size

        # Initialize base class with common parameters
        super().__init__(
            use_s2hr=use_s2hr,
            use_s2mr=use_s2mr,
            use_s2lr=use_s2lr,
            use_s1=use_s1,
            no_savanna=no_savanna,
            norm_stats_path=norm_stats_path,
            normalization_mode=normalization_mode,
        )

        subset_str = validate_string_value(subset, SEN12MS_SUBSETS, "SEN12MS subset")

        # make sure parent dir exists
        if not os.path.exists(path):
            msg = DATASET_ERRORS.DATASET_PATH_NOT_EXIST.format(path=path)
            log_and_raise(FileNotFoundError, msg, logger_instance=logger)

        # find and index samples
        self.samples = []
        pbar = tqdm(total=get_sen12ms_dataset_size(subset_str))
        pbar.set_description("[Load]")

        val_list = list(pd.read_csv(os.path.join(path, "SEN12MS_holdOutScenes.txt"), header=None)[0])
        val_list = [x.replace("s1_", "s2_") for x in val_list]
        # Convert to set for O(1) membership lookup instead of O(n) list lookup
        val_set = set(val_list)

        # compile a list of paths to all samples
        if subset_str == "train":
            # Combine filtering operations and use set for fast exclusion
            train_list = [
                os.path.join(seasonfolder, x)
                for seasonfolder in (
                    "ROIs1970_fall",
                    "ROIs1158_spring",
                    "ROIs2017_winter",
                    "ROIs1868_summer",
                )
                for x in os.listdir(os.path.join(path, seasonfolder))
                if "s2_" in x
            ]
            # Use set difference for O(1) per-element lookup
            sample_dirs = [x for x in train_list if x not in val_set]
        else:
            sample_dirs = val_list

        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/{DATASET_PATHS['TIFF_GLOB']}"), recursive=True)
            # INFO there is one "broken" file in the sen12ms dataset with nan
            #      values in the s1 data. we simply ignore this specific sample
            #      at this point. id: ROIs1868_summer_xx_146_p202
            if folder == "ROIs1868_summer/s2_146":
                broken_file = os.path.join(path, folder, "ROIs1868_summer_s2_146_p202.tif")
                if broken_file in s2_locations:
                    s2_locations.remove(broken_file)
                else:
                    pass
                pbar.write("ignored one sample because of nan values in " + "the s1 data")

            for s2_loc in s2_locations:
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
                lc_loc = s2_loc.replace("_s2_", "_lc_").replace("s2_", "lc_")

                pbar.update()
                self.samples.append({"lc": lc_loc, "s1": s1_loc, "s2": s2_loc, "id": os.path.basename(s2_loc)})

        pbar.close()

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i["id"])

        # Apply subset filtering if provided
        if subset_indices is not None:
            self.samples = [self.samples[i] for i in subset_indices]
            logger.info(f"Applied subset filtering: {len(self.samples)} samples selected")

        logger.info(f"loaded {len(self.samples)} samples for SEN12MS subset '{subset_str}'")

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single example from the dataset."""
        sample = load_sample(
            self.samples[index],
            self.use_s1,
            self.use_s2hr,
            self.use_s2mr,
            self.use_s2lr,
            no_savanna=self.no_savanna,
            igbp=True,
            norm_stats=self.norm_stats,
            normalization_mode=self.normalization_mode,
        )
        if self.border_exclusion and "label" in sample:
            sample["label"] = mask_label_borders(
                sample["label"],
                self.border_exclusion_size,
            )
        return sample

    def __len__(self) -> int:
        """Get number of samples in the dataset"""
        return len(self.samples)


class DFC2020(BaseDataset):
    """PyTorch dataset class for the DFC2020 dataset.

    Note: DFC2020 is for validation/test only.
    """

    @timeit("datasets/init_DFC2020", log_args=True, component="datasets", enable_timing=True)
    def __init__(
        self,
        path: str,
        subset: str = "val",
        no_savanna: bool = False,
        use_s2hr: bool = False,
        use_s2mr: bool = False,
        use_s2lr: bool = False,
        use_s1: bool = False,
        norm_stats_path: str = DATASET_PATHS["NORMALIZATION_STATS_FILE"],
        normalization_mode: str = "global",
    ) -> None:
        # Initialize base class with common parameters
        super().__init__(
            use_s2hr=use_s2hr,
            use_s2mr=use_s2mr,
            use_s2lr=use_s2lr,
            use_s1=use_s1,
            no_savanna=no_savanna,
            norm_stats_path=norm_stats_path,
            normalization_mode=normalization_mode,
        )

        # make sure parent dir exists
        if not os.path.exists(path):
            msg = DATASET_ERRORS.DATASET_PATH_NOT_EXIST.format(path=path)
            log_and_raise(FileNotFoundError, msg, logger_instance=logger)

        # build list of sample paths
        path = os.path.join(path, "ROIs0000_validation", "s2_validation") if subset == "val" else os.path.join(path, "ROIs0000_test", "s2_0")
        s2_locations = glob.glob(os.path.join(path, DATASET_PATHS["TIFF_GLOB"]), recursive=True)
        self.samples = []
        for s2_loc in tqdm(s2_locations, desc="[Load]"):
            s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
            lc_loc = s2_loc.replace("_dfc_", "_lc_").replace("s2_", "dfc_")
            self.samples.append({"lc": lc_loc, "s1": s1_loc, "s2": s2_loc, "id": os.path.basename(s2_loc)})

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i["id"])

        logger.info(f"loaded {len(self.samples)} samples for DFC2020 subset '{subset}'")

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single example from the dataset"""
        return load_sample(
            self.samples[index],
            self.use_s1,
            self.use_s2hr,
            self.use_s2mr,
            self.use_s2lr,
            no_savanna=self.no_savanna,
            igbp=False,
            norm_stats=self.norm_stats,
            normalization_mode=self.normalization_mode,
        )

    def __len__(self) -> int:
        """Get number of samples in the dataset"""
        return len(self.samples)


class TIFFDir(BaseDataset):
    """PyTorch dataset class for SEN12MS-like data.

    Expects dataset directory structure::

        <root>/
            s1/
            s2/
    """

    @timeit("datasets/init_TIFFDir", log_args=True, component="datasets", enable_timing=True)
    def __init__(
        self,
        path: str,
        no_savanna: bool = False,
        use_s2hr: bool = False,
        use_s2mr: bool = False,
        use_s2lr: bool = False,
        use_s1: bool = False,
        norm_stats_path: str = DATASET_PATHS["NORMALIZATION_STATS_FILE"],
        normalization_mode: str = "global",
    ) -> None:
        # Initialize base class with common parameters
        super().__init__(
            use_s2hr=use_s2hr,
            use_s2mr=use_s2mr,
            use_s2lr=use_s2lr,
            use_s1=use_s1,
            no_savanna=no_savanna,
            norm_stats_path=norm_stats_path,
            normalization_mode=normalization_mode,
        )

        # make sure parent dir exists
        if not os.path.exists(path):
            msg = DATASET_ERRORS.DATASET_PATH_NOT_EXIST.format(path=path)
            log_and_raise(FileNotFoundError, msg, logger_instance=logger)

        # compile a list of paths to all samples
        self.samples = []
        s2_locations = glob.glob(os.path.join(path, f"s2/{DATASET_PATHS['TIFF_GLOB']}"), recursive=True)
        for s2_loc in tqdm(s2_locations, desc="[Load]"):
            s1_loc = s2_loc.replace("s2/", "s1/").replace("_s2_", "_s1_")
            self.samples.append({"s1": s1_loc, "s2": s2_loc, "id": os.path.basename(s2_loc)})

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i["id"])

        logger.info(f"loaded {len(self.samples)} samples from TIFF directory at '{path}'")

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single example from the dataset"""
        return load_sample(
            self.samples[index],
            self.use_s1,
            self.use_s2hr,
            self.use_s2mr,
            self.use_s2lr,
            unlabeled=True,
            norm_stats=self.norm_stats,
            normalization_mode=self.normalization_mode,
        )

    def __len__(self) -> int:
        """Get number of samples in the dataset"""
        return len(self.samples)


def build_inference_dataset(
    dataset_type: str,
    data_dir: str,
    *,
    no_savanna: bool,
    use_s2hr: bool,
    use_s2mr: bool,
    use_s2lr: bool,
    use_s1: bool,
    normalization_mode: str = "global",
) -> SEN12MS | DFC2020 | TIFFDir:
    if dataset_type == "sen12ms_holdout":
        return SEN12MS(
            data_dir,
            subset="holdout",
            no_savanna=no_savanna,
            use_s2hr=use_s2hr,
            use_s2mr=use_s2mr,
            use_s2lr=use_s2lr,
            use_s1=use_s1,
            normalization_mode=normalization_mode,
        )

    if dataset_type == "tiff_dir":
        return TIFFDir(
            data_dir,
            no_savanna=no_savanna,
            use_s2hr=use_s2hr,
            use_s2mr=use_s2mr,
            use_s2lr=use_s2lr,
            use_s1=use_s1,
            normalization_mode=normalization_mode,
        )

    if is_dfc2020_dataset(dataset_type):
        dfc_subset = get_dfc2020_subset(dataset_type)
        return DFC2020(
            data_dir,
            subset=dfc_subset,
            no_savanna=no_savanna,
            use_s2hr=use_s2hr,
            use_s2mr=use_s2mr,
            use_s2lr=use_s2lr,
            use_s1=use_s1,
            normalization_mode=normalization_mode,
        )

    log_and_raise(ValueError, f"unknown dataset: {dataset_type}", logger_instance=logger)
