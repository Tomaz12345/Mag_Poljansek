import argparse
import logging
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .class_weights import get_dfc2020_class_weights
from .constants import CLASS_WEIGHT_METHODS, LOSS_FUNCTION_DEFAULTS, LOSS_TYPES, TrainArgDefaults
from .utils import log_and_raise, validate_string_value

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Multi-class focal loss with ignore-index support."""

    weight: torch.Tensor | None

    def __init__(
        self,
        gamma: float = LOSS_FUNCTION_DEFAULTS["FOCAL_GAMMA"],
        alpha: float = LOSS_FUNCTION_DEFAULTS["FOCAL_ALPHA"],
        ignore_index: int = -100,
        weight: torch.Tensor | None = None,
    ) -> None:
        """Initialize FocalLoss.

        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Weighting factor
            ignore_index: Index to ignore in loss computation
            weight: Optional class weights tensor (shape: [n_classes])

        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Model output (shape: [B, C, H, W])
            target: Ground truth labels (shape: [B, H, W])

        Returns:
            Scalar loss value

        """
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return logits.new_zeros(())

        ce_valid = ce[valid_mask]
        pt = torch.exp(-ce_valid)
        focal_term = (1.0 - pt) ** self.gamma
        rez = self.alpha * focal_term * ce_valid
        return cast(torch.Tensor, rez.mean())


class GeneralizedCrossEntropyLoss(nn.Module):
    """Generalized Cross Entropy (GCE), robust to noisy labels."""

    weight: torch.Tensor | None

    def __init__(
        self,
        q: float = LOSS_FUNCTION_DEFAULTS["GCE_Q"],
        ignore_index: int = -100,
        weight: torch.Tensor | None = None,
    ) -> None:
        """Initialize GeneralizedCrossEntropyLoss.

        Args:
            q: GCE parameter (0 < q <= 1)
            ignore_index: Index to ignore in loss computation
            weight: Optional class weights tensor (shape: [n_classes])

        """
        super().__init__()
        self.q = q
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute generalized cross entropy loss.

        Args:
            logits: Model output (shape: [B, C, H, W])
            target: Ground truth labels (shape: [B, H, W])

        Returns:
            Scalar loss value

        """
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return logits.new_zeros(())

        probs = F.softmax(logits, dim=1)
        probs = probs.permute(0, 2, 3, 1)[valid_mask]
        target_valid = target[valid_mask].long()
        p_t = probs.gather(1, target_valid.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
        loss = (1.0 - p_t.pow(self.q)) / self.q

        # Apply class weights if provided
        if self.weight is not None:
            weights = self.weight[target_valid]
            loss = loss * weights

        return cast(torch.Tensor, loss.mean())


class SymmetricCrossEntropyLoss(nn.Module):
    """Symmetric Cross Entropy (SCE): CE + reverse CE for label-noise robustness."""

    weight: torch.Tensor | None

    def __init__(
        self,
        alpha: float = LOSS_FUNCTION_DEFAULTS["SCE_ALPHA"],
        beta: float = LOSS_FUNCTION_DEFAULTS["SCE_BETA"],
        ignore_index: int = -100,
        eps: float = 1e-06,
        weight: torch.Tensor | None = None,
    ) -> None:
        """Initialize SymmetricCrossEntropyLoss.

        Args:
            alpha: Weight for forward CE term
            beta: Weight for reverse CE term
            ignore_index: Index to ignore in loss computation
            eps: Small constant for numerical stability
            weight: Optional class weights tensor (shape: [n_classes])

        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.eps = eps
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute symmetric cross entropy loss.

        Args:
            logits: Model output (shape: [B, C, H, W])
            target: Ground truth labels (shape: [B, H, W])

        Returns:
            Scalar loss value

        """
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return logits.new_zeros(())

        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        ce = ce[valid_mask]

        probs = F.softmax(logits, dim=1)
        probs = probs.permute(0, 2, 3, 1)[valid_mask]
        target_valid = target[valid_mask].long()
        num_classes = logits.shape[1]
        one_hot = F.one_hot(target_valid, num_classes=num_classes).float()
        one_hot = one_hot.clamp_min(self.eps).clamp_max(1.0)

        rce = -(probs * one_hot.log()).sum(dim=1)
        return (self.alpha * ce + self.beta * rce).mean()


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss with ignore-index support."""

    weight: torch.Tensor | None

    def __init__(
        self,
        smooth: float = LOSS_FUNCTION_DEFAULTS["DICE_SMOOTH"],
        ignore_index: int = -100,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss over classes, excluding ignored pixels."""
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return logits.new_zeros(())

        target_safe = torch.where(valid_mask, target, torch.zeros_like(target))
        target_one_hot = F.one_hot(target_safe.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

        valid_mask_f = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask_f
        target_one_hot = target_one_hot * valid_mask_f

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        cardinality = torch.sum(probs + target_one_hot, dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.weight is not None:
            class_weights = self.weight.to(dice_per_class.device, dtype=dice_per_class.dtype)
            class_weights = class_weights / class_weights.sum().clamp_min(1e-12)
            dice_score = torch.sum(dice_per_class * class_weights)
        else:
            dice_score = dice_per_class.mean()

        return cast(torch.Tensor, 1.0 - dice_score)


class CrossEntropyDiceLoss(nn.Module):
    """Principal loss used by UNetFormer paper: CE + Dice."""

    def __init__(
        self,
        smooth: float = LOSS_FUNCTION_DEFAULTS["DICE_SMOOTH"],
        ignore_index: int = -100,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction="mean")
        self.dice = DiceLoss(smooth=smooth, ignore_index=ignore_index, weight=weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.cross_entropy(logits, target) + self.dice(logits, target))


def create_loss_function(args: argparse.Namespace, ignore_index: int, num_classes: int | None = None) -> nn.Module:
    """Create loss function from training arguments.

    Args:
        args: Configuration arguments namespace
        ignore_index: Label index to ignore in loss computation
        num_classes: Number of classes (required if using class weights)

    Returns:
        Configured loss function module

    """
    loss_type = validate_string_value(getattr(args, "loss_type", TrainArgDefaults["loss_type"]), LOSS_TYPES, "loss type")

    # Get class weights if enabled
    weight: torch.Tensor | None = None
    if getattr(args, "use_class_weights", TrainArgDefaults["use_class_weights"]) and num_classes is not None:
        weight_method = validate_string_value(
            getattr(args, "class_weight_method", TrainArgDefaults["class_weight_method"]),
            CLASS_WEIGHT_METHODS,
            "class weight method",
        )
        weight_beta = float(getattr(args, "class_weight_beta", TrainArgDefaults["class_weight_beta"]))

        # Use pre-computed DFC2020 weights or compute from dataset
        if getattr(args, "use_precomputed_class_weights", TrainArgDefaults["use_precomputed_class_weights"]):
            weight = get_dfc2020_class_weights(
                n_classes=num_classes,
                method=weight_method,
                beta=weight_beta,
            )
        # else: weight would be computed from dataset (not implemented in this function)

    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction="mean")
    if loss_type == "dice":
        return DiceLoss(
            smooth=float(getattr(args, "dice_smooth", TrainArgDefaults["dice_smooth"])),
            ignore_index=ignore_index,
            weight=weight,
        )
    if loss_type == "cross_entropy_dice":
        return CrossEntropyDiceLoss(
            smooth=float(getattr(args, "dice_smooth", TrainArgDefaults["dice_smooth"])),
            ignore_index=ignore_index,
            weight=weight,
        )
    if loss_type == "focal":
        return FocalLoss(
            gamma=float(getattr(args, "focal_gamma", TrainArgDefaults["focal_gamma"])),
            alpha=float(getattr(args, "focal_alpha", TrainArgDefaults["focal_alpha"])),
            ignore_index=ignore_index,
            weight=weight,
        )
    if loss_type == "generalized_cross_entropy":
        return GeneralizedCrossEntropyLoss(
            q=float(getattr(args, "gce_q", TrainArgDefaults["gce_q"])),
            ignore_index=ignore_index,
            weight=weight,
        )
    if loss_type == "symmetric_cross_entropy":
        return SymmetricCrossEntropyLoss(
            alpha=float(getattr(args, "sce_alpha", TrainArgDefaults["sce_alpha"])),
            beta=float(getattr(args, "sce_beta", TrainArgDefaults["sce_beta"])),
            ignore_index=ignore_index,
            weight=weight,
        )
    if loss_type == "label_smoothing_cross_entropy":
        smoothing = float(getattr(args, "label_smoothing", TrainArgDefaults["label_smoothing"]))
        return nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=smoothing,
            ignore_index=ignore_index,
            reduction="mean",
        )

    # Unreachable due to validate_string_value, but kept for defensive programming.
    log_and_raise(ValueError, f"Unsupported loss type: {loss_type}", logger_instance=logger)
