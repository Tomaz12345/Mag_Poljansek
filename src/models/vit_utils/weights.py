# Implemented from: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image
#   Recognition at Scale", ICLR 2021 (arXiv:2010.11929)
# Reference implementation: https://github.com/google-research/vision_transformer

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..pretrained_loading import log_pretrained_load_result

logger = logging.getLogger(__name__)

# Variant -> timm architecture name (the pretraining resolution is baked into
# the name; pos_embed is interpolated to whatever img_size we actually use).
TIMM_MODEL_NAMES: dict[str, str] = {
    "vit_tiny": "vit_tiny_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "vit_base": "vit_base_patch16_224",
    "vit_large": "vit_large_patch16_224",
}

# pretrained_type -> timm checkpoint tag.
TIMM_WEIGHT_TAGS: dict[str, str] = {
    "imagenet_1k": "augreg_in21k_ft_in1k",
    "imagenet_21k": "augreg_in21k",
    "mae": "mae",
}

# MAE checkpoints were only released for ViT-B, ViT-L and ViT-H.
MAE_SUPPORTED_VARIANTS: tuple[str, ...] = ("vit_base", "vit_large")

# Fallback used when a requested tag has no checkpoint for the chosen variant.
_FALLBACK_WEIGHT_TYPE: str = "imagenet_21k"

# Checkpoint keys that belong to the classification head we replace with a decoder.
_SKIP_PREFIXES: tuple[str, ...] = ("head.", "pre_logits.", "fc_norm.")


def resolve_timm_name(variant: str, pretrained_type: str) -> str:
    """Build the fully-qualified timm model name for a variant/weight pair.

    Falls back to :data:`_FALLBACK_WEIGHT_TYPE` (with a warning) when MAE
    weights are requested for a variant that has none.

    Args:
        variant: One of :data:`TIMM_MODEL_NAMES`.
        pretrained_type: One of :data:`TIMM_WEIGHT_TAGS`.

    Returns:
        A timm model name such as ``vit_base_patch16_224.mae``.
    """
    if pretrained_type == "mae" and variant not in MAE_SUPPORTED_VARIANTS:
        logger.warning(
            f"ViT: no MAE checkpoint exists for variant '{variant}' "
            f"(MAE was released for {MAE_SUPPORTED_VARIANTS} only). "
            f"Falling back to pretrained_type='{_FALLBACK_WEIGHT_TYPE}'."
        )
        pretrained_type = _FALLBACK_WEIGHT_TYPE

    return f"{TIMM_MODEL_NAMES[variant]}.{TIMM_WEIGHT_TAGS[pretrained_type]}"


def _fetch_state_dict(variant: str, pretrained_type: str, pretrained_path: str) -> dict[str, torch.Tensor] | None:
    """Return a checkpoint state dict, or ``None`` if one cannot be obtained."""
    if pretrained_path and os.path.isfile(pretrained_path):
        logger.info(f"ViT: loading pretrained weights from local file {pretrained_path}")
        raw: Any = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        for key in ("state_dict", "model", "module"):
            if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
        return dict(raw)

    try:
        import timm
    except ImportError:
        logger.warning("timm is not installed. Skipping pretrained weight loading. Install with: pip install timm")
        return None

    timm_name = resolve_timm_name(variant, pretrained_type)
    logger.info(f"ViT: loading pretrained weights via timm.create_model('{timm_name}', pretrained=True)")
    try:
        pretrained_model = timm.create_model(timm_name, pretrained=True)
    except Exception as exc:
        logger.warning(f"Failed to fetch ViT weights '{timm_name}': {exc}")
        return None

    state_dict: dict[str, torch.Tensor] = pretrained_model.state_dict()
    return state_dict


def interpolate_pos_embed(pos_embed: torch.Tensor, grid_size: int, num_prefix_tokens: int = 1) -> torch.Tensor:
    """Bicubically resize the patch part of a positional embedding.

    Args:
        pos_embed: Checkpoint embedding of shape (1, num_prefix + N_old, D).
        grid_size: Target grid side length.
        num_prefix_tokens: Number of leading non-patch tokens (1 for the CLS token).

    Returns:
        Embedding of shape (1, num_prefix + grid_size**2, D).
    """
    prefix_pos = pos_embed[:, :num_prefix_tokens, :]
    patch_pos = pos_embed[:, num_prefix_tokens:, :]

    n_orig = patch_pos.shape[1]
    grid_orig = int(n_orig**0.5)
    embed_dim = patch_pos.shape[2]

    patch_pos = patch_pos.reshape(1, grid_orig, grid_orig, embed_dim).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(grid_size, grid_size), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_size**2, embed_dim)

    logger.info(f"ViT: interpolated pos_embed {grid_orig}x{grid_orig} -> {grid_size}x{grid_size}")

    return torch.cat([prefix_pos, patch_pos], dim=1)


def inflate_patch_embed(weight: torch.Tensor, in_channels: int) -> torch.Tensor:
    """Expand an RGB patch-embedding kernel to an arbitrary channel count.

    Replicates the channel-wise mean and rescales by ``3 / in_channels`` so the
    summed activation magnitude is preserved. This is the standard adaptation
    used when porting RGB backbones to multispectral EO input.

    Args:
        weight: Pretrained kernel of shape (D, 3, P, P).
        in_channels: Desired number of input channels.

    Returns:
        Kernel of shape (D, in_channels, P, P).
    """
    orig_channels = weight.shape[1]
    if orig_channels == in_channels:
        return weight

    mean_kernel = weight.mean(dim=1, keepdim=True)
    inflated = mean_kernel.repeat(1, in_channels, 1, 1) * (orig_channels / in_channels)

    logger.info(
        f"ViT: inflated patch_embed kernel {orig_channels} -> {in_channels} channels (mean-replicated, rescaled by {orig_channels}/{in_channels})"
    )

    return inflated


def adapt_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    grid_size: int,
    in_channels: int,
) -> dict[str, torch.Tensor]:
    """Remap checkpoint keys and reshape stem/positional weights to fit the model.

    Args:
        state_dict: Raw checkpoint weights in the canonical flat ViT layout.
        grid_size: Target patch-grid side length.
        in_channels: Number of channels the model's patch embedding expects.

    Returns:
        A state dict keyed to match :class:`~src.models.vit.ViT`.
    """
    adapted: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if any(key.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue

        if key.startswith("blocks."):
            new_key = key.replace("blocks.", "encoder.blocks.", 1)
        elif key.startswith("norm."):
            new_key = key.replace("norm.", "encoder.norm.", 1)
        else:
            new_key = key

        adapted[new_key] = value

    if "pos_embed" in adapted and adapted["pos_embed"].shape[1] != 1 + grid_size**2:
        adapted["pos_embed"] = interpolate_pos_embed(adapted["pos_embed"], grid_size)

    stem_key = "patch_embed.proj.weight"
    if stem_key in adapted and adapted[stem_key].shape[1] != in_channels:
        adapted[stem_key] = inflate_patch_embed(adapted[stem_key], in_channels)

    return adapted


def load_vit_pretrained_weights(
    model: nn.Module,
    *,
    variant: str,
    pretrained_type: str,
    pretrained_path: str,
    grid_size: int,
    in_channels: int,
    expected_missing_prefixes: list[str] | None = None,
) -> None:
    """Fetch, adapt and load pretrained ViT weights into ``model``.

    Never raises on acquisition failure — logs a warning and leaves the model
    randomly initialised, matching how every other backbone in this repo
    degrades.

    Args:
        model: The :class:`~src.models.vit.ViT` instance to load into.
        variant: Architecture variant name.
        pretrained_type: ``imagenet_1k``, ``imagenet_21k`` or ``mae``.
        pretrained_path: Optional local ``.pth`` overriding the timm download.
        grid_size: Target patch-grid side length.
        in_channels: Channels the model's patch embedding expects.
        expected_missing_prefixes: Key prefixes that are intentionally absent.
    """
    try:
        state_dict = _fetch_state_dict(variant, pretrained_type, pretrained_path)
    except Exception as exc:
        logger.warning(f"Failed to load ViT pretrained weights: {exc}")
        logger.info("Continuing with random initialization")
        return

    if state_dict is None:
        logger.info("Continuing with random initialization")
        return

    try:
        adapted = adapt_state_dict(state_dict, grid_size=grid_size, in_channels=in_channels)
        missing, unexpected = model.load_state_dict(adapted, strict=False)
    except Exception as exc:
        logger.warning(f"Failed to apply ViT pretrained weights: {exc}")
        logger.info("Continuing with random initialization")
        return

    log_pretrained_load_result(
        missing,
        unexpected,
        logger,
        model_name="ViT",
        expected_missing_prefixes=expected_missing_prefixes or ["decoder.", "neck.", "input_proj."],
    )

    logger.info("Successfully loaded ViT pretrained weights")
