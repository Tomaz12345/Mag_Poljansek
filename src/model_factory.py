from __future__ import annotations

import logging
import os
import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn as nn

from .constants import (
    ENCODER_FREEZE_DEFAULTS,
    ENCODER_LR_MULTIPLIER_DEFAULTS,
    INPUT_PROJECTION_DEFAULTS,
    LR_SCHEDULER_TYPES,
    MODEL_TYPES,
    OPTIMIZER_DEFAULTS,
    SCHEDULER_DEFAULTS,
    CopernicusFMArgDefaults,
    CROMAArgDefaults,
    DeepLabArgDefaults,
    DEOFMArgDefaults,
    DINOv2ArgDefaults,
    FlexiMoArgDefaults,
    OneFormerArgDefaults,
    SwinTransformerArgDefaults,
    TrainArgDefaults,
    UNetArgDefaults,
    UNetFormerArgDefaults,
    ViTArgDefaults,
)
from .utils import log_and_raise, validate_string_value

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, dict[str, object]] = {
    "deeplabv3+": DeepLabArgDefaults,
    "unet": UNetArgDefaults,
    "swin": SwinTransformerArgDefaults,
    "copernicus_fm": CopernicusFMArgDefaults,
    "unetformer": UNetFormerArgDefaults,
    "croma": CROMAArgDefaults,
    "fleximo": FlexiMoArgDefaults,
    "dinov2": DINOv2ArgDefaults,
    "oneformer": OneFormerArgDefaults,
    "deo_fm": DEOFMArgDefaults,
    "vit": ViTArgDefaults,
}


def setup_seeds(
    seed: int | None,
    fold_offset: int = 0,
    deterministic: bool = True,
) -> tuple[Callable[[int], None], torch.Generator]:
    """Set up RNG seeds for reproducibility.

    Args:
        seed: Base random seed. If None, only the generator is created.
        fold_offset: Offset added to seed (for reproducibility variants).
        deterministic: If True, enable PyTorch deterministic algorithms.

    Returns:
        Tuple of (seed_worker function for DataLoader, torch.Generator).
    """
    if seed is not None:
        effective_seed = int(seed) + fold_offset
        torch.manual_seed(effective_seed)
        random.seed(effective_seed)
        np.random.seed(effective_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed((int(seed) + fold_offset) if seed is not None else fold_offset)

    return seed_worker, g


def create_model(
    args: Namespace,
    n_classes: int,
    n_inputs: int,
    inference: bool = False,
) -> nn.Module:
    """Create a model instance from configuration.

    Args:
        args: Namespace with model configuration (merged train + model args).
        n_classes: Number of output classes.
        n_inputs: Number of input channels.
        inference: If True, disables pretrained weight loading and encoder freezing
            (for prediction where weights come from a checkpoint).

    Returns:
        Instantiated model (not moved to GPU).
    """
    model_type = validate_string_value(args.model, MODEL_TYPES, "model type")
    model: nn.Module

    # common model parameters (not necessarily used by all models, but simplifies argument handling)
    freeze_encoder = getattr(args, "freeze_encoder", ENCODER_FREEZE_DEFAULTS["freeze_encoder"]) and not inference
    freeze_encoder_layers = (
        ENCODER_FREEZE_DEFAULTS["freeze_encoder_layers"]
        if freeze_encoder
        else getattr(args, "freeze_encoder_layers", ENCODER_FREEZE_DEFAULTS["freeze_encoder_layers"])
    )
    freeze_encoder_layers_mode = getattr(args, "freeze_encoder_layers_mode", ENCODER_FREEZE_DEFAULTS["freeze_encoder_layers_mode"])

    input_projection_type = getattr(args, "input_projection_type", INPUT_PROJECTION_DEFAULTS["input_projection_type"])
    input_projection_hidden = getattr(args, "input_projection_hidden", INPUT_PROJECTION_DEFAULTS["input_projection_hidden"])
    input_projection_dims = getattr(args, "input_projection_dims", INPUT_PROJECTION_DEFAULTS["input_projection_dims"])

    if model_type == "deeplabv3+":
        from .models.deeplab import DeepLab

        _dd = DeepLabArgDefaults

        model = DeepLab(
            num_classes=n_classes,
            backbone="resnet",
            use_pretrained=False if inference else getattr(args, "use_pretrained", _dd["use_pretrained"]),
            output_stride=getattr(args, "out_stride", _dd["out_stride"]),
            sync_bn=False,
            freeze_bn=getattr(args, "freeze_bn", _dd["freeze_bn"]),
            freeze_encoder=freeze_encoder,
            freeze_encoder_layers=freeze_encoder_layers,
            n_in=n_inputs,
            aspp_out_channels=getattr(args, "aspp_out_channels", _dd["aspp_out_channels"]),
            decoder_low_level_channels=getattr(args, "decoder_low_level_channels", _dd["decoder_low_level_channels"]),
            decoder_out_channels=getattr(args, "decoder_out_channels", _dd["decoder_out_channels"]),
            use_depthwise_separable=getattr(args, "use_depthwise_separable", _dd["use_depthwise_separable"]),
            interpolation_mode=getattr(args, "interpolation_mode", _dd["interpolation_mode"]),
            align_corners=getattr(args, "align_corners", _dd["align_corners"]),
            input_projection_type=input_projection_type,
            input_projection_hidden=input_projection_hidden,
            input_projection_dims=input_projection_dims,
        )

    elif model_type == "unet":
        from .models.unet import UNet

        _und = UNetArgDefaults
        raw_unet_channels = getattr(args, "unet_channels", None)
        if raw_unet_channels and isinstance(raw_unet_channels, str):
            unet_channels: tuple[int, ...] = tuple(int(c.strip()) for c in raw_unet_channels.split(","))
        elif isinstance(raw_unet_channels, (tuple, list)):
            unet_channels = tuple(raw_unet_channels)
        else:
            unet_channels = UNetArgDefaults["unet_channels"]

        use_pretrained = getattr(args, "use_pretrained", _und["use_pretrained"])
        pretrained_encoder = getattr(args, "pretrained_encoder", _und["pretrained_encoder"])

        if use_pretrained and raw_unet_channels is not None and unet_channels != _und["unet_channels"]:
            logger.warning(f"unet_channels is ignored when use_pretrained=True. Using {pretrained_encoder} channel configuration.")

        model = UNet(
            n_channels=n_inputs,
            n_classes=n_classes,
            bilinear=getattr(args, "bilinear", _und["bilinear"]),
            channels=unet_channels,
            use_pretrained=use_pretrained,
            pretrained_encoder=pretrained_encoder,
            inference=inference,
            freeze_encoder=freeze_encoder,
            freeze_encoder_layers=freeze_encoder_layers,
            dropout=getattr(args, "dropout", _und["dropout"]),
            input_projection_type=input_projection_type,
            input_projection_hidden=input_projection_hidden,
            input_projection_dims=input_projection_dims,
        )

    elif model_type == "swin":
        from .models.swin_transformer import SwinTransformer

        _sd = SwinTransformerArgDefaults

        depths: tuple[int, ...] | None = None
        raw_depths = getattr(args, "depths", _sd["depths"])
        if raw_depths:
            if isinstance(raw_depths, str):
                depths = tuple(int(d.strip()) for d in raw_depths.split(","))
            elif isinstance(raw_depths, (tuple, list)):
                depths = tuple(raw_depths)

        num_heads: tuple[int, ...] | None = None
        raw_num_heads = getattr(args, "num_heads", _sd["num_heads"])
        if raw_num_heads:
            if isinstance(raw_num_heads, str):
                num_heads = tuple(int(h.strip()) for h in raw_num_heads.split(","))
            elif isinstance(raw_num_heads, (tuple, list)):
                num_heads = tuple(raw_num_heads)

        model = SwinTransformer(
            n_channels=n_inputs,
            n_classes=n_classes,
            img_size=getattr(args, "img_size", _sd["img_size"]),
            patch_size=getattr(args, "patch_size", _sd["patch_size"]),
            embed_dim=getattr(args, "embed_dim", _sd["embed_dim"]),
            depths=depths or tuple(int(d.strip()) for d in str(_sd["depths"]).split(",")),
            num_heads=num_heads or tuple(int(h.strip()) for h in str(_sd["num_heads"]).split(",")),
            window_size=getattr(args, "window_size", _sd["window_size"]),
            mlp_ratio=getattr(args, "mlp_ratio", _sd["mlp_ratio"]),
            qkv_bias=getattr(args, "qkv_bias", _sd["qkv_bias"]),
            drop_rate=getattr(args, "drop_rate", _sd["drop_rate"]),
            attn_drop_rate=getattr(args, "attn_drop_rate", _sd["attn_drop_rate"]),
            drop_path_rate=getattr(args, "drop_path_rate", _sd["drop_path_rate"]),
            decoder_type=getattr(args, "decoder_type", _sd["decoder_type"]),
            decoder_channels=getattr(args, "decoder_channels", _sd["decoder_channels"]),
            decoder_dropout=getattr(args, "decoder_dropout", _sd["decoder_dropout"]),
            decoder_align_corners=getattr(args, "decoder_align_corners", _sd["decoder_align_corners"]),
            use_auxiliary_head=getattr(args, "use_auxiliary_head", _sd["use_auxiliary_head"]),
            auxiliary_loss_weight=getattr(args, "auxiliary_loss_weight", _sd["auxiliary_loss_weight"]),
            auxiliary_head_channels=getattr(args, "auxiliary_head_channels", _sd["auxiliary_head_channels"]),
            auxiliary_head_index=getattr(args, "auxiliary_head_index", _sd["auxiliary_head_index"]),
            use_pretrained=getattr(args, "use_pretrained", _sd["use_pretrained"]) and not inference,
            pretrained_type=getattr(args, "pretrained_type", _sd["pretrained_type"]),
            variant=getattr(args, "variant", _sd["variant"]) or None,
            freeze_encoder=freeze_encoder,
            freeze_encoder_layers=freeze_encoder_layers,
            freeze_encoder_layers_mode=freeze_encoder_layers_mode,
            input_projection_type=input_projection_type,
            input_projection_hidden=input_projection_hidden,
            input_projection_dims=input_projection_dims,
        )

    elif model_type == "copernicus_fm":
        from .models.copernicus_fm import CopernicusFM, build_bandwidth_list_nm, build_wavelength_list_nm

        _cd = CopernicusFMArgDefaults

        # Build wavelength/bandwidth lists from sensor flags
        _use_s2hr = getattr(args, "use_s2hr", TrainArgDefaults["use_s2hr"])
        _use_s2mr = getattr(args, "use_s2mr", TrainArgDefaults["use_s2mr"])
        _use_s2lr = getattr(args, "use_s2lr", TrainArgDefaults["use_s2lr"])
        _use_s1 = getattr(args, "use_s1", TrainArgDefaults["use_s1"])
        _wavelengths = build_wavelength_list_nm(_use_s2hr, _use_s2mr, _use_s2lr, _use_s1)
        _bandwidths = build_bandwidth_list_nm(_use_s2hr, _use_s2mr, _use_s2lr, _use_s1)

        model = CopernicusFM(
            n_channels=n_inputs,
            n_classes=n_classes,
            img_size=getattr(args, "img_size", _cd["img_size"]),
            patch_size=getattr(args, "patch_size", _cd["patch_size"]),
            embed_dim=getattr(args, "embed_dim", _cd["embed_dim"]),
            depth=getattr(args, "depth", _cd["depth"]),
            num_heads=getattr(args, "num_heads", _cd["num_heads"]),
            mlp_ratio=getattr(args, "mlp_ratio", _cd["mlp_ratio"]),
            qkv_bias=getattr(args, "qkv_bias", _cd["qkv_bias"]),
            drop_rate=getattr(args, "drop_rate", _cd["drop_rate"]),
            attn_drop_rate=getattr(args, "attn_drop_rate", _cd["attn_drop_rate"]),
            drop_path_rate=getattr(args, "drop_path_rate", _cd["drop_path_rate"]),
            decoder_type=getattr(args, "decoder_type", _cd["decoder_type"]),
            decoder_channels=getattr(args, "decoder_channels", _cd["decoder_channels"]),
            decoder_dropout=getattr(args, "decoder_dropout", _cd["decoder_dropout"]),
            use_spectral_hypernetwork=getattr(args, "use_spectral_hypernetwork", _cd["use_spectral_hypernetwork"]),
            use_variable_hypernetwork=getattr(args, "use_variable_hypernetwork", _cd["use_variable_hypernetwork"]),
            variable_embed_dim=getattr(args, "variable_embed_dim", _cd["variable_embed_dim"]),
            use_metadata_encoding=getattr(args, "use_metadata_encoding", _cd["use_metadata_encoding"]),
            metadata_dropout_prob=getattr(args, "metadata_dropout_prob", _cd["metadata_dropout_prob"]),
            wv_planes=getattr(args, "wv_planes", _cd["wv_planes"]),
            wavelengths=_wavelengths,
            bandwidths=_bandwidths,
            use_adapters=getattr(args, "use_adapters", _cd["use_adapters"]),
            adapter_rank=getattr(args, "adapter_rank", _cd["adapter_rank"]),
            use_pretrained=getattr(args, "use_pretrained", _cd["use_pretrained"]) and not inference,
            pretrained_path=getattr(args, "pretrained_path", _cd["pretrained_path"]),
            freeze_encoder=freeze_encoder,
            freeze_encoder_layers=freeze_encoder_layers,
            freeze_encoder_layers_mode=freeze_encoder_layers_mode,
            variant=getattr(args, "variant", _cd["variant"]) or None,
        )

    elif model_type == "unetformer":
        from .models.unetformer import UNetFormer

        _ud = UNetFormerArgDefaults
        encoder_channels: tuple[int, ...] | None = None
        if hasattr(args, "encoder_channels") and args.encoder_channels:
            encoder_channels = tuple(int(c.strip()) for c in args.encoder_channels.split(","))

        model = UNetFormer(
            n_channels=n_inputs,
            n_classes=n_classes,
            encoder_channels=encoder_channels or tuple(int(c.strip()) for c in str(_ud["encoder_channels"]).split(",")),
            decoder_channels=getattr(args, "decoder_channels", _ud["decoder_channels"]),
            num_heads=getattr(args, "num_heads", _ud["num_heads"]),
            window_size=getattr(args, "window_size", _ud["window_size"]),
            mlp_ratio=getattr(args, "mlp_ratio", _ud["mlp_ratio"]),
            drop_rate=getattr(args, "drop_rate", _ud["drop_rate"]),
            attn_drop_rate=getattr(args, "attn_drop_rate", _ud["attn_drop_rate"]),
            drop_path_rate=getattr(args, "drop_path_rate", _ud["drop_path_rate"]),
            use_pretrained=getattr(args, "use_pretrained", _ud["use_pretrained"]) and not inference,
            variant=getattr(args, "variant", _ud["variant"]) or None,
            freeze_encoder=freeze_encoder,
            freeze_encoder_layers=freeze_encoder_layers,
            input_projection_type=input_projection_type,
            input_projection_hidden=input_projection_hidden,
            input_projection_dims=input_projection_dims,
        )

    elif model_type == "croma":
        from .models.croma import CROMA

        _cr = CROMAArgDefaults

        # Determine SAR and optical channel counts from sensor config
        n_channels_sar = 2 if getattr(args, "use_s1", TrainArgDefaults["use_s1"]) else 0
        n_channels_optical = 0
        if getattr(args, "use_s2hr", TrainArgDefaults["use_s2hr"]):
            n_channels_optical += 4
        if getattr(args, "use_s2mr", TrainArgDefaults["use_s2mr"]):
            n_channels_optical += 6
        if getattr(args, "use_s2lr", TrainArgDefaults["use_s2lr"]):
            n_channels_optical += 3

        # Paper CROMA pretraining uses 12 S2 channels (cirrus removed).
        use_pretrained = getattr(args, "use_pretrained", _cr["use_pretrained"])
        croma_drop_optical_cirrus = getattr(args, "croma_drop_optical_cirrus", _cr["croma_drop_optical_cirrus"])
        if use_pretrained and croma_drop_optical_cirrus and n_channels_optical == 13:
            n_channels_optical = 12

        model = CROMA(
            n_channels=n_inputs,
            n_channels_sar=n_channels_sar,
            n_channels_optical=n_channels_optical,
            n_classes=n_classes,
            img_size=getattr(args, "img_size", _cr["img_size"]),
            patch_size=getattr(args, "patch_size", _cr["patch_size"]),
            embed_dim=getattr(args, "embed_dim", _cr["embed_dim"]),
            depth=getattr(args, "depth", _cr["depth"]),
            num_heads=getattr(args, "num_heads", _cr["num_heads"]),
            mlp_ratio=getattr(args, "mlp_ratio", _cr["mlp_ratio"]),
            qkv_bias=getattr(args, "qkv_bias", _cr["qkv_bias"]),
            drop_rate=getattr(args, "drop_rate", _cr["drop_rate"]),
            attn_drop_rate=getattr(args, "attn_drop_rate", _cr["attn_drop_rate"]),
            use_alibi=getattr(args, "use_alibi", _cr["use_alibi"]),
            decoder_type=getattr(args, "decoder_type", _cr["decoder_type"]),
            decoder_channels=getattr(args, "decoder_channels", _cr["decoder_channels"]),
            decoder_dropout=getattr(args, "decoder_dropout", _cr["decoder_dropout"]),
            use_pretrained=getattr(args, "use_pretrained", _cr["use_pretrained"]) and not inference,
            pretrained_path=getattr(args, "pretrained_path", _cr["pretrained_path"]),
            freeze_encoder=freeze_encoder,
            freeze_encoder_layers=freeze_encoder_layers,
            freeze_encoder_layers_mode=freeze_encoder_layers_mode,
            croma_drop_optical_cirrus=croma_drop_optical_cirrus,
            variant=getattr(args, "variant", _cr["variant"]) or None,
        )

    elif model_type == "fleximo":
        from .models.fleximo import FlexiMo

        _fd = FlexiMoArgDefaults

        model = FlexiMo(
            n_channels=n_inputs,
            n_classes=n_classes,
            img_size=getattr(args, "img_size", _fd["img_size"]),
            patch_size=getattr(args, "patch_size", _fd["patch_size"]),
            target_patch_size=getattr(args, "target_patch_size", _fd["target_patch_size"]) or None,
            embed_dim=getattr(args, "embed_dim", _fd["embed_dim"]),
            depth=getattr(args, "depth", _fd["depth"]),
            num_heads=getattr(args, "num_heads", _fd["num_heads"]),
            mlp_ratio=getattr(args, "mlp_ratio", _fd["mlp_ratio"]),
            qkv_bias=getattr(args, "qkv_bias", _fd["qkv_bias"]),
            drop_rate=getattr(args, "drop_rate", _fd["drop_rate"]),
            attn_drop_rate=getattr(args, "attn_drop_rate", _fd["attn_drop_rate"]),
            drop_path_rate=getattr(args, "drop_path_rate", _fd["drop_path_rate"]),
            wv_planes=getattr(args, "wv_planes", _fd["wv_planes"]),
            pos_embed_type=getattr(args, "pos_embed_type", _fd["pos_embed_type"]),
            decoder_type=getattr(args, "decoder_type", _fd["decoder_type"]),
            decoder_channels=getattr(args, "decoder_channels", _fd["decoder_channels"]),
            decoder_dropout=getattr(args, "decoder_dropout", _fd["decoder_dropout"]),
            task_type=getattr(args, "task_type", _fd["task_type"]),
            classifier_dropout=getattr(args, "classifier_dropout", _fd["classifier_dropout"]),
            wavelengths=getattr(args, "wavelengths", _fd["wavelengths"]),
            use_s2hr=getattr(args, "use_s2hr", TrainArgDefaults["use_s2hr"]),
            use_s2mr=getattr(args, "use_s2mr", TrainArgDefaults["use_s2mr"]),
            use_s2lr=getattr(args, "use_s2lr", TrainArgDefaults["use_s2lr"]),
            use_s1=getattr(args, "use_s1", TrainArgDefaults["use_s1"]),
            use_pretrained=getattr(args, "use_pretrained", _fd["use_pretrained"]) and not inference,
            pretrained_path=getattr(args, "pretrained_path", _fd["pretrained_path"]),
            freeze_encoder=freeze_encoder,
            freeze_encoder_layers=freeze_encoder_layers,
            freeze_encoder_layers_mode=freeze_encoder_layers_mode,
            variant=getattr(args, "variant", _fd["variant"]) or None,
        )

    elif model_type == "dinov2":
        from .models.dinov2 import DINOv2

        _dv = DINOv2ArgDefaults

        model = DINOv2(
            n_channels=n_inputs,
            n_classes=n_classes,
            img_size=getattr(args, "img_size", _dv["img_size"]),
            patch_size=getattr(args, "patch_size", _dv["patch_size"]),
            input_resize=getattr(args, "input_resize", _dv["input_resize"]) or None,
            embed_dim=getattr(args, "embed_dim", _dv["embed_dim"]),
            depth=getattr(args, "depth", _dv["depth"]),
            num_heads=getattr(args, "num_heads", _dv["num_heads"]),
            mlp_ratio=getattr(args, "mlp_ratio", _dv["mlp_ratio"]),
            qkv_bias=getattr(args, "qkv_bias", _dv["qkv_bias"]),
            drop_rate=getattr(args, "drop_rate", _dv["drop_rate"]),
            attn_drop_rate=getattr(args, "attn_drop_rate", _dv["attn_drop_rate"]),
            drop_path_rate=getattr(args, "drop_path_rate", _dv["drop_path_rate"]),
            decoder_type=getattr(args, "decoder_type", _dv["decoder_type"]),
            decoder_channels=getattr(args, "decoder_channels", _dv["decoder_channels"]),
            decoder_dropout=getattr(args, "decoder_dropout", _dv["decoder_dropout"]),
            use_pretrained=getattr(args, "use_pretrained", _dv["use_pretrained"]) and not inference,
            pretrained_path=getattr(args, "pretrained_path", _dv["pretrained_path"]),
            freeze_encoder=freeze_encoder,
            freeze_encoder_layers=freeze_encoder_layers,
            freeze_encoder_layers_mode=freeze_encoder_layers_mode,
            variant=getattr(args, "variant", _dv["variant"]) or None,
            input_projection_type=input_projection_type,
            input_projection_hidden=input_projection_hidden,
            input_projection_dims=input_projection_dims,
        )

    elif model_type == "oneformer":
        from .models.oneformer import OneFormer

        _of = OneFormerArgDefaults

        model = OneFormer(
            n_channels=n_inputs,
            n_classes=n_classes,
            img_size=getattr(args, "img_size", _of["img_size"]),
            use_pretrained=getattr(args, "use_pretrained", _of["use_pretrained"]) and not inference,
            pretrained_path=getattr(args, "pretrained_path", _of["pretrained_path"]),
            freeze_encoder=freeze_encoder,
            freeze_encoder_layers=freeze_encoder_layers,
            freeze_encoder_layers_mode=freeze_encoder_layers_mode,
            variant=getattr(args, "variant", _of["variant"]) or None,
            oneformer_task=getattr(args, "oneformer_task", _of["oneformer_task"]),
            oneformer_task_seq_length=getattr(args, "oneformer_task_seq_length", _of["oneformer_task_seq_length"]),
            input_projection_type=input_projection_type,
            input_projection_hidden=input_projection_hidden,
            input_projection_dims=input_projection_dims,
        )

    elif model_type == "deo_fm":
        from .models.deo_fm import DEOFM

        _de = DEOFMArgDefaults

        model = DEOFM(
            n_channels=n_inputs,
            n_classes=n_classes,
            img_size=getattr(args, "img_size", _de["img_size"]),
            patch_size=getattr(args, "patch_size", _de["patch_size"]),
            decoder_type=getattr(args, "decoder_type", _de["decoder_type"]),
            decoder_channels=getattr(args, "decoder_channels", _de["decoder_channels"]),
            decoder_dropout=getattr(args, "decoder_dropout", _de["decoder_dropout"]),
            use_pretrained=getattr(args, "use_pretrained", _de["use_pretrained"]) and not inference,
            pretrained_path=getattr(args, "pretrained_path", _de["pretrained_path"]),
            variant=getattr(args, "variant", _de["variant"]) or None,
            deo_input_mode=getattr(args, "deo_input_mode", _de["deo_input_mode"]),
            freeze_encoder=freeze_encoder,
            freeze_encoder_layers=freeze_encoder_layers,
            freeze_encoder_layers_mode=freeze_encoder_layers_mode,
            input_projection_type=input_projection_type,
            input_projection_hidden=input_projection_hidden,
            input_projection_dims=input_projection_dims,
        )

    elif model_type == "vit":
        from .models.vit import ViT

        _vt = ViTArgDefaults

        model = ViT(
            n_channels=n_inputs,
            n_classes=n_classes,
            img_size=getattr(args, "img_size", _vt["img_size"]),
            patch_size=getattr(args, "patch_size", _vt["patch_size"]),
            embed_dim=getattr(args, "embed_dim", _vt["embed_dim"]),
            depth=getattr(args, "depth", _vt["depth"]),
            num_heads=getattr(args, "num_heads", _vt["num_heads"]),
            mlp_ratio=getattr(args, "mlp_ratio", _vt["mlp_ratio"]),
            qkv_bias=getattr(args, "qkv_bias", _vt["qkv_bias"]),
            drop_rate=getattr(args, "drop_rate", _vt["drop_rate"]),
            attn_drop_rate=getattr(args, "attn_drop_rate", _vt["attn_drop_rate"]),
            drop_path_rate=getattr(args, "drop_path_rate", _vt["drop_path_rate"]),
            decoder_type=getattr(args, "decoder_type", _vt["decoder_type"]),
            decoder_channels=getattr(args, "decoder_channels", _vt["decoder_channels"]),
            decoder_dropout=getattr(args, "decoder_dropout", _vt["decoder_dropout"]),
            use_pretrained=getattr(args, "use_pretrained", _vt["use_pretrained"]) and not inference,
            pretrained_type=getattr(args, "pretrained_type", _vt["pretrained_type"]),
            pretrained_path=getattr(args, "pretrained_path", _vt["pretrained_path"]),
            vit_stem_mode=getattr(args, "vit_stem_mode", _vt["vit_stem_mode"]),
            vit_neck=getattr(args, "vit_neck", _vt["vit_neck"]),
            freeze_encoder=freeze_encoder,
            freeze_encoder_layers=freeze_encoder_layers,
            freeze_encoder_layers_mode=freeze_encoder_layers_mode,
            variant=getattr(args, "variant", _vt["variant"]) or None,
            input_projection_type=input_projection_type,
            input_projection_hidden=input_projection_hidden,
            input_projection_dims=input_projection_dims,
        )

    else:
        log_and_raise(ValueError, f"unknown model type: {args.model}", logger_instance=logger)

    return model


def create_optimizer(
    model: nn.Module,
    args: Namespace,
    model_args: Namespace,
) -> torch.optim.Optimizer:
    """Create an optimizer with model-specific learning rate schedules.

    Args:
        model: The model (may be wrapped in DataParallel).
        args: Training args (lr, momentum, weight_decay, etc.).
        model_args: Model-specific args (model name, use_pretrained, etc.).

    Returns:
        Configured optimizer.
    """
    base_model: Any = cast(nn.Module, model.module) if isinstance(model, nn.DataParallel) else model
    model_name = model_args.model

    # common values
    encoder_lr = args.lr * ENCODER_LR_MULTIPLIER_DEFAULTS["encoder_lr_multiplier"]

    if model_name == "deeplabv3+":
        train_params: list[dict[str, Any]] = [
            {"params": base_model.get_1x_lr_params(), "lr": args.lr},
            {
                "params": base_model.get_10x_lr_params(),
                "lr": args.lr * OPTIMIZER_DEFAULTS["DEEPLAB_LR_MULTIPLIER"],
            },
        ]
        return torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay)

    _unet_pretrained = getattr(model_args, "use_pretrained", UNetArgDefaults["use_pretrained"])
    if model_name == "unet" and _unet_pretrained:
        train_params = [
            {"params": base_model.get_encoder_params(), "lr": encoder_lr},
            {"params": base_model.get_decoder_params(), "lr": args.lr},
        ]
        return torch.optim.AdamW(train_params, weight_decay=args.weight_decay)

    if model_name == "swin":
        if getattr(model_args, "use_pretrained", SwinTransformerArgDefaults["use_pretrained"]):
            train_params = [
                {"params": base_model.get_encoder_params(), "lr": encoder_lr},
                {"params": base_model.get_decoder_params(), "lr": args.lr},
            ]
        else:
            train_params = [{"params": model.parameters(), "lr": args.lr}]
        return torch.optim.AdamW(
            train_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=OPTIMIZER_DEFAULTS["ADAM_BETAS"],
        )

    if model_name in ("copernicus_fm", "unetformer", "croma", "fleximo", "dinov2", "oneformer", "deo_fm", "vit"):
        train_params = [
            {"params": base_model.get_encoder_params(), "lr": encoder_lr},
            {"params": base_model.get_decoder_params(), "lr": args.lr},
        ]
        return torch.optim.AdamW(
            train_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=OPTIMIZER_DEFAULTS["ADAM_BETAS"],
        )

    # Default fallback (unet without pretrained, or unknown)
    return torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args: Namespace,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        args: Training args with scheduler configuration.
        steps_per_epoch: Number of training steps per epoch.

    Returns:
        LR scheduler instance or None if scheduler is disabled.
    """
    scheduler_type = validate_string_value(getattr(args, "lr_scheduler", TrainArgDefaults["lr_scheduler"]), LR_SCHEDULER_TYPES, "LR scheduler type")

    if scheduler_type == "none":
        return None

    warmup_epochs = getattr(args, "lr_warmup_epochs", TrainArgDefaults["lr_warmup_epochs"])
    warmup_steps = warmup_epochs * steps_per_epoch
    lr_min = getattr(args, "lr_min", TrainArgDefaults["lr_min"])

    if scheduler_type == "cosine":
        total_steps = args.max_epochs * steps_per_epoch if hasattr(args, "max_epochs") else None
        if total_steps is None:
            total_steps = args.num_steps if hasattr(args, "num_steps") else warmup_steps * SCHEDULER_DEFAULTS["TOTAL_STEPS_FALLBACK_MULTIPLIER"]

        return CosineAnnealingWithWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=lr_min,
        )

    if scheduler_type == "linear":
        return LinearWarmup(
            optimizer,
            warmup_steps=warmup_steps,
        )

    if scheduler_type == "exponential":
        decay_rate = float(getattr(args, "lr_decay_rate", TrainArgDefaults["lr_decay_rate"]))
        return ExponentialDecayWithWarmup(
            optimizer,
            decay_rate=decay_rate,
            warmup_steps=warmup_steps,
        )

    return None


class LinearWarmup(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return self.base_lrs


class CosineAnnealingWithWarmup(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        import math

        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # Cosine annealing
        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.eta_min + (base_lr - self.eta_min) * cosine_factor for base_lr in self.base_lrs]


class ExponentialDecayWithWarmup(torch.optim.lr_scheduler.LRScheduler):
    """Exponential decay scheduler with linear warmup.

    After warmup, LR decays as: lr = base_lr * (decay_rate ^ step).
    Paper uses decay_rate=0.99 for pretraining.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        decay_rate: float = 0.99,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ):
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        steps_after_warmup = self.last_epoch - self.warmup_steps
        decay_factor = self.decay_rate**steps_after_warmup
        return [base_lr * decay_factor for base_lr in self.base_lrs]
