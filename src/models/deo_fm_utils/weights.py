# Implemented from: Wolf et al., "Brewing Stronger Features: Dual-Teacher Distillation for
#   Multispectral Earth Observation", CVPR 2026 (arXiv:2602.19863)
# Reference implementation: https://github.com/wolfilip/DEO-FM

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

from ..pretrained_loading import log_pretrained_load_result

logger = logging.getLogger(__name__)


def _strip_checkpoint(raw: dict[str, object]) -> dict[str, torch.Tensor]:
    # Match upstream DEO/utils/misc.py::load_pretrained_weights exactly:
    # pull the "student" sub-dict, then apply ``str.replace`` (which strips
    # the prefix wherever it appears in the key) for each of the known
    # wrapper prefixes, in this order.
    #
    # ``feat_extr.`` is the prefix used by the published HuggingFace
    # checkpoints (SolaireTheSun/DEO): every one of their 355 keys is
    # ``feat_extr.features.*``, while the load target is the inner torchvision
    # Swin whose keys start at ``features.*``. Without stripping it, all 355
    # keys miss and the encoder silently trains from random init.
    inner: dict[str, torch.Tensor] = raw.get("student", raw)  # type: ignore[assignment]
    stripped: dict[str, torch.Tensor] = dict(inner)
    for prefix in ("_orig_mod.", "module.", "backbone.", "feat_extr."):
        stripped = {k.replace(prefix, ""): v for k, v in stripped.items()}
    return stripped


def load_deo_pretrained_weights(
    target: nn.Module,
    checkpoint_path: str,
    *,
    model_name: str,
    expected_missing_prefixes: list[str],
) -> None:
    """Load a DEO checkpoint into *target*.

    Args:
        target: The submodule whose ``state_dict`` keys match the checkpoint
            (Swin: the inner ``torchvision`` swin_b; ViT: the
            ``DEOVisionTransformer``).
        checkpoint_path: Local path to the ``.pth`` file.
        model_name: Label for log messages (e.g. ``"DEO-FM (swin_b)"``).
        expected_missing_prefixes: Keys that legitimately do not exist in the
            checkpoint (e.g. the unused multispectral stem when running with
            an RGB input — both stems are always allocated for compatibility).
    """
    if not os.path.isfile(checkpoint_path):
        logger.warning(f"DEO weights not found at {checkpoint_path}; using random init")
        return

    logger.info(f"Loading DEO weights from {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _strip_checkpoint(raw)

    missing, unexpected = target.load_state_dict(state_dict, strict=False)
    log_pretrained_load_result(
        missing,
        unexpected,
        logger,
        model_name=model_name,
        expected_missing_prefixes=expected_missing_prefixes,
    )
