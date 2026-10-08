from collections import OrderedDict
from typing import Any

SENTINEL_BANDS: dict[str, Any] = {"S2_HR": (2, 3, 4, 8), "S2_MR": (5, 6, 7, 9, 12, 13), "S2_LR": (1, 10, 11), "S1_STATS_INDICES": (13, 14)}

CLASS_MAPPING: dict[str, Any] = {"IGBP_TO_DFC2020": (0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10)}

IGBP_CLASS_NAMES: dict[str, Any] = {
    "NO_SAVANNA": ("Forest", "Shrubland", "Grassland", "Wetlands", "Croplands", "Urban/Built-up", "Snow/Ice", "Barren", "Water"),
    "ALL_CLASSES": ("Forest", "Shrubland", "Savanna", "Grassland", "Wetlands", "Croplands", "Urban/Built-up", "Snow/Ice", "Barren", "Water"),
}

DATASET_PATHS: dict[str, Any] = {"TIFF_GLOB": "*.tif", "NORMALIZATION_STATS_FILE": "dataset_data.json"}

NORM_CONFIG: dict[str, Any] = {"EPSILON": 1e-12}

VIZ_CONFIG: dict[str, Any] = {
    "BRIGHTNESS_S2HR": 3,
    "BRIGHTNESS_S2_ONLY": 3,
    "BRIGHTNESS_S1_ONLY": 1,
    "RGB_S2HR_S2LR": (3, 2, 1),
    "RGB_S2HR_ONLY": (2, 1, 0),
    "RGB_S1_ONLY": 0,
    "RGB_DEFAULT": 0,
    "EXPECTED_TILE_SIZE": (256, 256),
}

RESNET_CONFIG: dict[str, Any] = {"BOTTLENECK_EXPANSION": 4, "INITIAL_CHANNELS": 64}

MODEL_TYPES: tuple[str, ...] = (
    "deeplabv3+",
    "unet",
    "swin",
    "copernicus_fm",
    "unetformer",
    "croma",
    "fleximo",
    "dinov2",
    "oneformer",
    "deo_fm",
    "vit",
)

SWIN_VARIANTS: tuple[str, ...] = ("swin_tiny", "swin_small", "swin_base", "swin_large")

COPERNICUS_FM_VARIANTS: tuple[str, ...] = ("base", "large", "small")


DINOV2_VARIANTS: tuple[str, ...] = ("dinov2_small", "dinov2_base", "dinov2_large", "dinov2_giant")


DEO_FM_VARIANTS: tuple[str, ...] = ("swin_b", "vit_base")

DEO_FM_INPUT_MODES: tuple[str, ...] = ("auto", "rgb", "ms")


VIT_VARIANTS: tuple[str, ...] = ("vit_tiny", "vit_small", "vit_base", "vit_large")

# MAE weights (He et al., CVPR 2022) exist for vit_base / vit_large only;
# requesting them for a smaller variant falls back to imagenet_21k.
VIT_PRETRAINED_TYPES: tuple[str, ...] = ("imagenet_1k", "imagenet_21k", "mae")

VIT_STEM_MODES: tuple[str, ...] = ("project", "inflate")

VIT_NECK_TYPES: tuple[str, ...] = ("multiscale", "none")


DATASET_TYPES: tuple[str, ...] = ("sen12ms_holdout", "dfc2020_val", "dfc2020_test", "tiff_dir")

TRAIN_DATASET_TYPES: tuple[str, ...] = ("sen12ms_holdout", "dfc2020_val", "dfc2020_test")

SEN12MS_SUBSETS: tuple[str, ...] = ("train", "holdout")

DFC2020_SUBSETS: tuple[str, ...] = ("val", "test")

LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

NORMALIZATION_MODES: tuple[str, ...] = ("global", "per_scene", "histogram_eq", "none")

LR_SCHEDULER_TYPES: tuple[str, ...] = ("none", "cosine", "linear", "exponential")

GRADUAL_UNFREEZE_UNITS: tuple[str, ...] = ("epochs", "training_steps")

LOSS_TYPES: tuple[str, ...] = (
    "cross_entropy",
    "dice",
    "cross_entropy_dice",
    "focal",
    "generalized_cross_entropy",
    "symmetric_cross_entropy",
    "label_smoothing_cross_entropy",
)

CLASS_WEIGHT_METHODS: tuple[str, ...] = ("inverse", "effective", "median")

REPORT_FORMATS: dict[str, Any] = {"PDF": "pdf", "HTML": "html", "ALL": "all"}

REPORT_SOURCES: dict[str, Any] = {
    "CONTINUOUS_RUN": "continuous_run",
    "TRAIN": "train",
    "PREDICT": "predict",
    "TESTS": "tests",
}

REPORT_STAGES: dict[str, Any] = {
    "TRAINING": "training",
    "PREDICTION": "prediction",
    "ANALYSIS": "analysis",
    "TESTS": "tests",
}

LOSS_FUNCTION_DEFAULTS: dict[str, Any] = {
    "FOCAL_GAMMA": 2.0,
    "FOCAL_ALPHA": 0.25,
    "GCE_Q": 0.7,
    "SCE_ALPHA": 1.0,
    "SCE_BETA": 1.0,
    "DICE_SMOOTH": 1.0,
}

OPTIMIZER_DEFAULTS: dict[str, Any] = {"ADAM_BETAS": (0.9, 0.999), "DEEPLAB_LR_MULTIPLIER": 10}

SCHEDULER_DEFAULTS: dict[str, Any] = {"TOTAL_STEPS_FALLBACK_MULTIPLIER": 10}

VISUALIZATION_DEFAULTS: dict[str, Any] = {"CONFUSION_MATRIX_FIGSIZE": (10, 8), "IMAGE_NORMALIZATION_EPSILON": 1e-08}

SPECIAL_VALUES: dict[str, Any] = {"IGNORE_LABEL": 255}

# Expose band constants for convenience
S2_BANDS_HR = SENTINEL_BANDS["S2_HR"]
S2_BANDS_MR = SENTINEL_BANDS["S2_MR"]
S2_BANDS_LR = SENTINEL_BANDS["S2_LR"]

ANALYSIS_DEFAULTS: dict[str, Any] = {"FIGURE_DPI": 150}

PLOT_DEFAULTS: dict[str, Any] = {"FIGURE_DPI": 150, "MARKER_SIZE": 4, "LINE_WIDTH": 2, "ALPHA_GRID": 0.3}

# Shared default groups

LOGGING_DEFAULTS: dict[str, Any] = {"log_level": "INFO", "log_batch_freq": 1}

EXPERIMENT_DEFAULTS: dict[str, Any] = {"experiment_name": "experiment", "log_dir": None, "seed": 42, **LOGGING_DEFAULTS}

OUTPUT_DEFAULTS: dict[str, Any] = {"out_dir": "predictions"}

ENCODER_FREEZE_DEFAULTS: dict[str, Any] = {"freeze_encoder": False, "freeze_encoder_layers": 0, "freeze_encoder_layers_mode": "individual"}

ENCODER_LR_MULTIPLIER_DEFAULTS: dict[str, Any] = {"encoder_lr_multiplier": 0.1}

TRANSFORMER_RATE_DEFAULTS: dict[str, Any] = {"drop_rate": 0.0, "attn_drop_rate": 0.0, "drop_path_rate": 0.1}

DECODER_DEFAULTS: dict[str, Any] = {"decoder_type": "upernet", "decoder_channels": 512, "decoder_dropout": 0.1, "decoder_align_corners": False}

AUXILIARY_HEAD_DEFAULTS: dict[str, Any] = {
    "use_auxiliary_head": True,
    "auxiliary_loss_weight": 0.4,
    "auxiliary_head_channels": 256,
    "auxiliary_head_index": 2,
}

MODEL_IMAGE_DEFAULTS: dict[str, Any] = {"img_size": 256}

MODEL_USE_PRETRAIN_DEFAULTS: dict[str, Any] = {"use_pretrained": False}

INPUT_PROJECTION_DEFAULTS: dict[str, Any] = {
    "input_projection_type": "linear",
    "input_projection_hidden": 8,
    "input_projection_dims": None,
}

TrainArgDefaults: dict[str, Any] = {
    **EXPERIMENT_DEFAULTS,
    **ENCODER_FREEZE_DEFAULTS,
    "val_freq": 1000,
    "save_freq": 1000,
    "log_freq": 100,
    "enable_timing": False,
    "enable_profiling": False,
    "profile_steps": 5,
    "profile_output_dir": None,
    "enable_tensorboard_logging": True,
    "tensorboard_log_freq": 100,
    "log_model_graph": True,
    "log_distributions": True,
    "log_batch_images": True,
    "log_weight_histograms": False,
    "log_gradient_histograms": False,
    "log_learning_rate": True,
    "log_gradient_norm": True,
    "lr_scheduler": "cosine",
    "lr_warmup_epochs": 5,
    "lr_min": 0.0,
    "lr_decay_rate": 0.99,
    "enable_gradual_unfreezing": False,
    "gradual_unfreeze_every_n": 1,
    "gradual_unfreeze_unit": "epochs",
    "loss_type": "cross_entropy",
    "focal_gamma": 2.0,
    "focal_alpha": 0.25,
    "gce_q": 0.7,
    "sce_alpha": 1.0,
    "sce_beta": 1.0,
    "label_smoothing": 0.0,
    "dice_smooth": 1.0,
    "oneformer_contrastive_weight": 0.0,
    "oneformer_contrastive_temperature": 0.07,
    "oneformer_contrastive_use_gt": True,
    "use_class_weights": False,
    "class_weight_method": "effective",
    "class_weight_beta": 0.9999,
    "use_precomputed_class_weights": True,
    "enable_noisy_label_weighting": False,
    "noisy_label_warmup_steps": 1000,
    "noisy_label_ema_alpha": 0.9,
    "noisy_label_min_weight": 0.2,
    "use_s2hr": True,
    "use_s2mr": True,
    "use_s2lr": True,
    "use_s1": True,
    "no_savanna": True,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "batch_size": 32,
    "workers": 4,
    "max_epochs": 10,
    "num_steps": None,
    "data_dir_train": None,
    "dataset_val": "sen12ms_holdout",
    "data_dir_val": None,
    "accumulation_steps": None,
    "continue_learning_from_previous": False,
    "normalization_mode": "global",
    "border_exclusion": False,
    "border_exclusion_size": 10,
}

PredictArgDefaults: dict[str, Any] = {
    **LOGGING_DEFAULTS,
    "seed": EXPERIMENT_DEFAULTS["seed"],
    "config_file": "checkpoints/args.pkl",
    "checkpoint_file": "checkpoints/final.pth",
    "batch_size": 32,
    "workers": 4,
    "enable_timing": False,
    "score": True,
    "dataset": "dfc2020_test",
    "data_dir": None,
    "out_dir": "predictions",
    "preview_dir": None,
    "predict_subset_ratio": 1,
}

AnalysisArgDefaults: dict[str, Any] = {
    **EXPERIMENT_DEFAULTS,
    **OUTPUT_DEFAULTS,
    "analysis_dir": None,
    "class_names": None,
}

PlotArgDefaults: dict[str, Any] = {**EXPERIMENT_DEFAULTS, **OUTPUT_DEFAULTS}

ReportArgDefaults: dict[str, Any] = {
    **EXPERIMENT_DEFAULTS,
    "report_format": "all",
    "report_num_samples": 5,
    "report_sample_ids": None,
}

DeepLabArgDefaults: dict[str, Any] = {
    "model": "deeplabv3+",
    "backbone": "resnet",
    "use_pretrained": False,
    "sync_bn": False,
    "n_in": 3,
    "out_stride": 16,
    "aspp_out_channels": 256,
    "decoder_low_level_channels": 48,
    "decoder_out_channels": 256,
    "use_depthwise_separable": True,
    "interpolation_mode": "bilinear",
    "align_corners": True,
    "freeze_bn": False,
    **ENCODER_LR_MULTIPLIER_DEFAULTS,
    **INPUT_PROJECTION_DEFAULTS,
}

UNetArgDefaults: dict[str, Any] = {
    "model": "unet",
    "bilinear": False,
    **MODEL_USE_PRETRAIN_DEFAULTS,
    "pretrained_encoder": "resnet18",
    "unet_channels": (64, 128, 256, 512, 1024),
    "dropout": 0.0,
    **ENCODER_LR_MULTIPLIER_DEFAULTS,
    **INPUT_PROJECTION_DEFAULTS,
}

SwinTransformerArgDefaults: dict[str, Any] = {
    "model": "swin",
    "variant": "swin_tiny",
    **MODEL_IMAGE_DEFAULTS,
    "patch_size": 4,
    "window_size": 7,
    "embed_dim": 96,
    "depths": "2,2,6,2",
    "num_heads": "3,6,12,24",
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    **TRANSFORMER_RATE_DEFAULTS,
    **DECODER_DEFAULTS,
    **AUXILIARY_HEAD_DEFAULTS,
    **MODEL_USE_PRETRAIN_DEFAULTS,
    "pretrained_type": "imagenet_1k",
    **ENCODER_LR_MULTIPLIER_DEFAULTS,
    **INPUT_PROJECTION_DEFAULTS,
}

CopernicusFMArgDefaults: dict[str, Any] = {
    "model": "copernicus_fm",
    "variant": "base",
    **MODEL_IMAGE_DEFAULTS,
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    **TRANSFORMER_RATE_DEFAULTS,
    **DECODER_DEFAULTS,
    "use_spectral_hypernetwork": True,
    "use_variable_hypernetwork": False,
    "variable_embed_dim": 768,
    "use_metadata_encoding": True,
    "metadata_dropout_prob": 0.0,
    "wv_planes": 128,
    "use_adapters": False,
    "adapter_rank": 8,
    **MODEL_USE_PRETRAIN_DEFAULTS,
    "pretrained_path": "wangyi111/Copernicus-FM",
    **ENCODER_LR_MULTIPLIER_DEFAULTS,
}


UNetFormerArgDefaults: dict[str, Any] = {
    "model": "unetformer",
    "variant": "unetformer_small",
    "encoder_channels": "64,128,256,512",
    "decoder_channels": 64,
    "num_heads": 8,
    "window_size": 8,
    "mlp_ratio": 4.0,
    **TRANSFORMER_RATE_DEFAULTS,
    **MODEL_USE_PRETRAIN_DEFAULTS,
    **ENCODER_LR_MULTIPLIER_DEFAULTS,
    **INPUT_PROJECTION_DEFAULTS,
}

CROMAArgDefaults: dict[str, Any] = {
    "model": "croma",
    "variant": None,
    **MODEL_IMAGE_DEFAULTS,
    "patch_size": 8,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "qkv_bias": False,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "use_alibi": True,
    "croma_drop_optical_cirrus": True,
    **DECODER_DEFAULTS,
    **MODEL_USE_PRETRAIN_DEFAULTS,
    "pretrained_path": "antofuller/CROMA",
    **ENCODER_LR_MULTIPLIER_DEFAULTS,
}


FlexiMoArgDefaults: dict[str, Any] = {
    "model": "fleximo",
    "variant": "base",
    **MODEL_IMAGE_DEFAULTS,
    "img_size": 224,
    "patch_size": 16,
    "target_patch_size": None,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "wv_planes": 128,
    "pos_embed_type": "learnable",
    "task_type": "segmentation",
    "classifier_dropout": 0.0,
    "wavelengths": None,
    **TRANSFORMER_RATE_DEFAULTS,
    **DECODER_DEFAULTS,
    **MODEL_USE_PRETRAIN_DEFAULTS,
    "pretrained_path": "earthflow/DOFA",
    **ENCODER_LR_MULTIPLIER_DEFAULTS,
}

DINOv2ArgDefaults: dict[str, Any] = {
    "model": "dinov2",
    "variant": "dinov2_small",
    **MODEL_IMAGE_DEFAULTS,
    "patch_size": 14,
    "input_resize": None,
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    **TRANSFORMER_RATE_DEFAULTS,
    **DECODER_DEFAULTS,
    **MODEL_USE_PRETRAIN_DEFAULTS,
    "pretrained_path": "facebookresearch/dinov2",
    **ENCODER_LR_MULTIPLIER_DEFAULTS,
    **INPUT_PROJECTION_DEFAULTS,
}


DEOFMArgDefaults: dict[str, Any] = {
    "model": "deo_fm",
    "variant": "swin_b",
    **MODEL_IMAGE_DEFAULTS,
    "patch_size": 8,
    "deo_input_mode": "auto",
    **DECODER_DEFAULTS,
    **MODEL_USE_PRETRAIN_DEFAULTS,
    "pretrained_path": "SolaireTheSun/DEO",
    **ENCODER_LR_MULTIPLIER_DEFAULTS,
    **INPUT_PROJECTION_DEFAULTS,
}


# Defaults follow the original paper (Dosovitskiy et al., ICLR 2021): ViT-B/16
# is the canonical model, MLP size is 4x the hidden dim, QKV bias is enabled,
# and the headline results come from ImageNet-21k supervised pretraining.
ViTArgDefaults: dict[str, Any] = {
    "model": "vit",
    "variant": "vit_base",
    **MODEL_IMAGE_DEFAULTS,
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "vit_stem_mode": "project",
    "vit_neck": "multiscale",
    **TRANSFORMER_RATE_DEFAULTS,
    **DECODER_DEFAULTS,
    **MODEL_USE_PRETRAIN_DEFAULTS,
    "pretrained_type": "imagenet_21k",
    "pretrained_path": "",
    **ENCODER_LR_MULTIPLIER_DEFAULTS,
    **INPUT_PROJECTION_DEFAULTS,
}


OneFormerArgDefaults: dict[str, Any] = {
    "model": "oneformer",
    "variant": "swin_large",
    **MODEL_IMAGE_DEFAULTS,
    "pretrained_path": "shi-labs/oneformer_cityscapes_swin_large",
    "oneformer_task": "semantic",
    "oneformer_task_seq_length": 77,
    # Hungarian matching loss hyperparameters
    "oneformer_num_queries": 250,
    "oneformer_no_object_weight": 0.1,
    "oneformer_class_weight": 2.0,
    "oneformer_mask_weight": 5.0,
    "oneformer_dice_weight": 5.0,
    "oneformer_aux_loss_weight": 1.0,
    **MODEL_USE_PRETRAIN_DEFAULTS,
    **ENCODER_LR_MULTIPLIER_DEFAULTS,
    **INPUT_PROJECTION_DEFAULTS,
}


TIMING_FORMAT: dict[str, Any] = {"MAX_DISPLAY_COLLECTION_SIZE": 5, "MAX_DISPLAY_STRING_LENGTH": 50}

# Reporting constants

ReportDefaults: dict[str, int] = {
    "FIGURE_WIDTH": 10,
    "FIGURE_HEIGHT": 6,
}

VALID_FORMATS: set[str] = set(REPORT_FORMATS.values())

VALID_SOURCES: set[str] = set(REPORT_SOURCES.values())

VALID_STAGES: set[str] = set(REPORT_STAGES.values())

EXCLUDED_CONFIG_PARAMS: set[str] = {"experiment_name"}

CONFIG_PARAMETER_GROUPS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "Training",
            {
                "enable_key": None,
                "params": [
                    "lr",
                    "momentum",
                    "weight_decay",
                    "batch_size",
                    "max_epochs",
                    "num_steps",
                    "accumulation_steps",
                    "workers",
                    "seed",
                    "continue_learning_from_previous",
                    "log_dir",
                ],
                "always_show": [
                    "lr",
                    "momentum",
                    "weight_decay",
                    "batch_size",
                    "max_epochs",
                    "num_steps",
                ],
            },
        ),
        (
            "Loss Function",
            {
                "enable_key": None,
                "selector_key": "loss_type",
                "params": [
                    "loss_type",
                    "focal_gamma",
                    "focal_alpha",
                    "gce_q",
                    "sce_alpha",
                    "sce_beta",
                    "label_smoothing",
                    "dice_smooth",
                ],
                "children_by_value": {
                    "cross_entropy": [],
                    "dice": ["dice_smooth"],
                    "cross_entropy_dice": ["dice_smooth"],
                    "focal": ["focal_gamma", "focal_alpha"],
                    "generalized_cross_entropy": ["gce_q"],
                    "symmetric_cross_entropy": ["sce_alpha", "sce_beta"],
                    "label_smoothing_cross_entropy": ["label_smoothing"],
                },
                "always_show": ["loss_type"],
            },
        ),
        (
            "Class Weights",
            {
                "enable_key": "use_class_weights",
                "params": [
                    "use_class_weights",
                    "class_weight_method",
                    "class_weight_beta",
                    "use_precomputed_class_weights",
                ],
                "always_show": [],
            },
        ),
        (
            "Noisy Label Weighting",
            {
                "enable_key": "enable_noisy_label_weighting",
                "params": [
                    "enable_noisy_label_weighting",
                    "noisy_label_warmup_steps",
                    "noisy_label_ema_alpha",
                    "noisy_label_min_weight",
                ],
                "always_show": [],
            },
        ),
        (
            "LR Scheduler",
            {
                "enable_key": None,
                "selector_key": "lr_scheduler",
                "params": ["lr_scheduler", "lr_warmup_epochs", "lr_min", "lr_decay_rate"],
                "children_by_value": {
                    "none": [],
                    "cosine": ["lr_warmup_epochs", "lr_min"],
                    "linear": ["lr_warmup_epochs"],
                    "exponential": ["lr_warmup_epochs", "lr_decay_rate"],
                },
                "always_show": [],
            },
        ),
        (
            "Data",
            {
                "enable_key": None,
                "selector_key": "border_exclusion",
                "params": [
                    "use_s2hr",
                    "use_s2mr",
                    "use_s2lr",
                    "use_s1",
                    "no_savanna",
                    "data_dir_train",
                    "dataset_val",
                    "data_dir_val",
                    "normalization_mode",
                    "border_exclusion",
                    "border_exclusion_size",
                ],
                "children_by_value": {
                    "false": [],
                    "true": ["border_exclusion_size"],
                },
                "always_show": ["border_exclusion", "border_exclusion_size"],
            },
        ),
        (
            "Prediction",
            {
                "enable_key": None,
                "params": [
                    "config_file",
                    "checkpoint_file",
                    "score",
                    "dataset",
                    "data_dir",
                    "out_dir",
                    "preview_dir",
                    "predict_subset_ratio",
                ],
                "always_show": [],
            },
        ),
        (
            "Gradual Unfreezing",
            {
                "enable_key": "enable_gradual_unfreezing",
                "params": [
                    "enable_gradual_unfreezing",
                    "gradual_unfreeze_every_n",
                    "gradual_unfreeze_unit",
                ],
                "always_show": [],
            },
        ),
        (
            "TensorBoard / Logging",
            {
                "enable_key": "enable_tensorboard_logging",
                "params_prefix": "log_",
                "extra_params": [
                    "enable_tensorboard_logging",
                    "tensorboard_log_freq",
                    "log_freq",
                    "log_level",
                ],
                "exclude_from_children": ["log_batch_freq"],
                "always_show": [],
            },
        ),
        (
            "Timing & Profiling",
            {
                "enable_key": "enable_timing",
                "params": [
                    "enable_timing",
                    "enable_profiling",
                    "profile_steps",
                    "profile_output_dir",
                ],
                "always_show": [],
            },
        ),
        (
            "Pretrained Weights",
            {
                "enable_key": "use_pretrained",
                "params": [
                    "use_pretrained",
                    "pretrained_path",
                    "pretrained_encoder",
                ],
                "always_show": [],
            },
        ),
        (
            "Model-Specific Parameters",
            {
                "enable_key": None,
                "selector_key": "model",
                "params": [
                    "croma_drop_optical_cirrus",
                    "deo_input_mode",
                    "vit_stem_mode",
                    "vit_neck",
                    "oneformer_task",
                    "oneformer_task_seq_length",
                    "oneformer_num_queries",
                    "oneformer_no_object_weight",
                    "oneformer_class_weight",
                    "oneformer_mask_weight",
                    "oneformer_dice_weight",
                    "oneformer_aux_loss_weight",
                    "oneformer_contrastive_weight",
                    "oneformer_contrastive_temperature",
                    "oneformer_contrastive_use_gt",
                ],
                "children_by_value": {
                    "unet": [],
                    "deeplabv3+": [],
                    "swin": [],
                    "unetformer": [],
                    "oneformer": [
                        "oneformer_task",
                        "oneformer_task_seq_length",
                        "oneformer_num_queries",
                        "oneformer_no_object_weight",
                        "oneformer_class_weight",
                        "oneformer_mask_weight",
                        "oneformer_dice_weight",
                        "oneformer_aux_loss_weight",
                        "oneformer_contrastive_weight",
                        "oneformer_contrastive_temperature",
                        "oneformer_contrastive_use_gt",
                    ],
                    "fleximo": [],
                    "copernicus_fm": [],
                    "dinov2": [],
                    "deo_fm": [
                        "deo_input_mode",
                    ],
                    "vit": [
                        "vit_stem_mode",
                        "vit_neck",
                    ],
                    "croma": [
                        "croma_drop_optical_cirrus",
                    ],
                },
                "always_show": [],
            },
        ),
    ]
)

IGBP_CLASSES: list[str] = list(IGBP_CLASS_NAMES["ALL_CLASSES"])
IGBP_CLASSES_NO_SAVANNA: list[str] = list(IGBP_CLASS_NAMES["NO_SAVANNA"])

SEN12MS_DATASET_SIZES: dict[str, int] = {
    "train": 162556,
    "holdout": 18106,
}


def get_sen12ms_dataset_size(subset: str) -> int:
    if subset in SEN12MS_DATASET_SIZES:
        return SEN12MS_DATASET_SIZES[subset]
    msg = f"Unknown SEN12MS subset: {subset}"
    raise ValueError(msg)
