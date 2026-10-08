from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from typing import Any, cast

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .constants import (
    ENCODER_FREEZE_DEFAULTS,
    GRADUAL_UNFREEZE_UNITS,
    TrainArgDefaults,
)
from .utils import validate_string_value

logger = logging.getLogger(__name__)


def build_checkpoint_payload(items: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return dict(items)


class NoisyLabelScoreEMA:
    """Track EMA scores for noisy-label weighting."""

    def __init__(self) -> None:
        self.scores: list[tuple[str, float]] = []

    def get(self, sample_id: str, default: float) -> float:
        for existing_id, value in self.scores:
            if existing_id == sample_id:
                return value
        return default

    def set(self, sample_id: str, value: float) -> None:
        for index, (existing_id, _) in enumerate(self.scores):
            if existing_id == sample_id:
                self.scores[index] = (sample_id, value)
                return
        self.scores.append((sample_id, value))


class GradualUnfreezer:
    """Apply gradual encoder unfreezing during training."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.enabled: bool = bool(getattr(args, "enable_gradual_unfreezing", TrainArgDefaults["enable_gradual_unfreezing"]))
        self.interval: int = int(getattr(args, "gradual_unfreeze_every_n", TrainArgDefaults["gradual_unfreeze_every_n"]))
        self.unit: str = validate_string_value(
            getattr(args, "gradual_unfreeze_unit", TrainArgDefaults["gradual_unfreeze_unit"]),
            GRADUAL_UNFREEZE_UNITS,
            "gradual unfreeze unit",
        )
        self.initial_frozen_layers: int | None = None
        self.last_applied_target_frozen_layers: int | None = None
        self._disabled_reason_logged: bool = False

    def maybe_apply(
        self,
        model: nn.Module,
        current_epoch: int,
        current_step: int,
        writer: SummaryWriter,
    ) -> None:
        """Apply scheduled unfreezing if target frozen layers changed."""
        if not self.enabled:
            return

        base_model = model.module if isinstance(model, nn.DataParallel) else model

        if not self._supports_gradual_unfreezing(base_model):
            if not self._disabled_reason_logged:
                logger.warning("Gradual unfreezing enabled but model does not expose required methods; disabling schedule.")
                self._disabled_reason_logged = True
            return

        if self.initial_frozen_layers is None:
            frozen_info = self._get_frozen_layer_info(base_model)
            if frozen_info is None:
                return
            self.initial_frozen_layers = int(frozen_info.get("frozen_layers", 0))
            if self.initial_frozen_layers <= 0:
                if not self._disabled_reason_logged:
                    logger.info("Gradual unfreezing enabled, but no encoder stages are currently frozen; nothing to schedule.")
                    self._disabled_reason_logged = True
                return

        elapsed_units = current_epoch // self.interval if self.unit == "epochs" else current_step // self.interval
        target_frozen_layers = max(self.initial_frozen_layers - elapsed_units, 0)

        if self.last_applied_target_frozen_layers == target_frozen_layers:
            return

        self._apply_target_frozen_layers(base_model, target_frozen_layers)
        self.last_applied_target_frozen_layers = target_frozen_layers

        frozen_info = self._get_frozen_layer_info(base_model)
        if frozen_info is None:
            return

        frozen_layers = int(frozen_info.get("frozen_layers", target_frozen_layers))
        total_layers = int(frozen_info.get("total_layers", 0))
        frozen_layer_names_raw = frozen_info.get("frozen_layer_names", [])
        frozen_layer_names = (
            ", ".join(str(name) for name in frozen_layer_names_raw) if isinstance(frozen_layer_names_raw, list) else str(frozen_layer_names_raw)
        )

        logger.info(
            "Gradual unfreezing update (%s every %d): frozen_layers=%d/%d (%s)",
            self.unit,
            self.interval,
            frozen_layers,
            total_layers,
            frozen_layer_names if frozen_layer_names else "none",
        )
        writer.add_scalar("model/frozen_layers_count", frozen_layers, global_step=current_step)

    def _supports_gradual_unfreezing(self, model: nn.Module) -> bool:
        return all(hasattr(model, attr) for attr in ("get_encoder_params", "_freeze_encoder_layers", "get_frozen_layer_info"))

    def _get_frozen_layer_info(self, model: nn.Module) -> Mapping[str, Any] | None:
        get_info_fn = getattr(model, "get_frozen_layer_info", None)
        if not callable(get_info_fn):
            return None
        info = get_info_fn()
        if not isinstance(info, Mapping):
            return None
        return cast(Mapping[str, Any], info)

    def _apply_target_frozen_layers(self, model: nn.Module, target_frozen_layers: int) -> None:
        model_api: Any = model
        get_encoder_params = model_api.get_encoder_params
        for param in get_encoder_params():
            param.requires_grad = True

        if target_frozen_layers <= 0:
            return

        freeze_layers_fn = model_api._freeze_encoder_layers
        freeze_mode = getattr(self.args, "freeze_encoder_layers_mode", ENCODER_FREEZE_DEFAULTS["freeze_encoder_layers_mode"])

        try:
            freeze_layers_fn(target_frozen_layers, mode=freeze_mode)
        except TypeError:
            freeze_layers_fn(target_frozen_layers)
