from __future__ import annotations

import logging
from typing import NamedTuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import SPECIAL_VALUES, OneFormerArgDefaults

logger = logging.getLogger(__name__)


def _dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft Dice loss between sigmoid predictions and binary targets.

    Args:
        pred: (N,) or (N, ...) sigmoid probabilities.
        target: Same shape as pred, binary float.

    Returns:
        Scalar mean Dice loss.
    """
    pred = pred.flatten(1)
    target = target.flatten(1)
    numerator = 2.0 * (pred * target).sum(1)
    denominator = pred.sum(1) + target.sum(1)
    loss = 1.0 - (numerator + 1.0) / (denominator + 1.0)
    return cast(torch.Tensor, loss.mean())


def _sigmoid_bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean binary cross-entropy with logits."""
    return F.binary_cross_entropy_with_logits(pred, target, reduction="mean")


class MatchResult(NamedTuple):
    """Indices of matched (query, gt) pairs for one image."""

    query_indices: torch.Tensor  # (M,) int64
    gt_indices: torch.Tensor  # (M,) int64


class OneFormerHungarianLoss(nn.Module):
    """Bipartite matching loss for OneFormer.

    Args:
        num_classes: Number of semantic classes (excluding no-object).
        num_queries: Number of transformer decoder queries.
        no_object_weight: Loss weight for no-object (unmatched) queries.
        class_weight: Relative weight of classification cost/loss.
        mask_weight: Relative weight of binary mask BCE cost/loss.
        dice_weight: Relative weight of Dice mask cost/loss.
        aux_loss_weight: Weight applied to each auxiliary decoder layer loss
            (paper uses 1.0 — equal weight to the final layer loss).
        ignore_index: Pixel value to ignore in target maps.
    """

    def __init__(
        self,
        num_classes: int,
        num_queries: int = OneFormerArgDefaults["oneformer_num_queries"],
        no_object_weight: float = OneFormerArgDefaults["oneformer_no_object_weight"],
        class_weight: float = OneFormerArgDefaults["oneformer_class_weight"],
        mask_weight: float = OneFormerArgDefaults["oneformer_mask_weight"],
        dice_weight: float = OneFormerArgDefaults["oneformer_dice_weight"],
        aux_loss_weight: float = OneFormerArgDefaults["oneformer_aux_loss_weight"],
        ignore_index: int = SPECIAL_VALUES["IGNORE_LABEL"],
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.no_object_weight = no_object_weight
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.aux_loss_weight = aux_loss_weight
        self.ignore_index = ignore_index

        # CE weight vector: classes 0..C-1 weight 1.0, no-object class C weight no_object_weight
        ce_weights = torch.ones(num_classes + 1)
        ce_weights[num_classes] = no_object_weight
        self.register_buffer("ce_weights", ce_weights)

    # GT conversion

    def _semantic_to_instances(self, target: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Convert a semantic label map to a list of binary masks + class ids.

        Args:
            target: (H, W) int64 semantic map.

        Returns:
            gt_classes: list of scalar tensors, one per unique class present.
            gt_masks:   list of (H, W) bool tensors, one per unique class.
        """
        gt_classes: list[torch.Tensor] = []
        gt_masks: list[torch.Tensor] = []
        unique = target.unique()
        for cls in unique:
            cls_int = cls.item()
            if cls_int == self.ignore_index:
                continue
            mask = target == cls  # (H, W) bool
            gt_classes.append(cls.long())
            gt_masks.append(mask)
        return gt_classes, gt_masks

    @torch.no_grad()
    def _compute_cost_matrix(
        self,
        class_logits: torch.Tensor,
        mask_logits: torch.Tensor,
        gt_classes: list[torch.Tensor],
        gt_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Build (Q, S) cost matrix for one image.

        Matching is done at the mask_logits resolution (typically H/4, W/4)
        for efficiency.

        Args:
            class_logits: (Q, C) float.
            mask_logits:  (Q, Hm, Wm) float (logits, not sigmoid).
            gt_classes:   S-length list of scalar int64 tensors.
            gt_masks:     S-length list of (H, W) bool tensors.

        Returns:
            cost: (Q, S) float tensor on CPU.
        """
        Hm, Wm = mask_logits.shape[1], mask_logits.shape[2]
        device = class_logits.device

        # --- Classification cost: negative softmax probability of GT class
        # Strip no-object class (last dim) before computing cost — matching is
        # only between semantic classes and GT instances, not the no-object slot.
        class_probs = class_logits[..., : self.num_classes].softmax(-1)  # (Q, C)
        gt_cls_tensor = torch.stack(gt_classes).to(device)  # (S,)
        # cost_class[q, s] = -prob[q, gt_classes[s]]
        cost_class = -class_probs[:, gt_cls_tensor]  # (Q, S)

        # --- Mask costs at low resolution
        # Downsample GT masks to (Hm, Wm)
        gt_masks_tensor = torch.stack([m.float() for m in gt_masks]).to(device)  # (S, H, W)
        gt_masks_lr = F.interpolate(gt_masks_tensor.unsqueeze(1), size=(Hm, Wm), mode="nearest").squeeze(1)  # (S, Hm, Wm)

        pred_masks_sigmoid = mask_logits.sigmoid()  # (Q, Hm, Wm)

        # Flatten for pairwise computation
        pred_flat = pred_masks_sigmoid.flatten(1)  # (Q, Hm*Wm)
        gt_flat = gt_masks_lr.flatten(1)  # (S, Hm*Wm)

        # BCE cost: mean over pixels for each (q, s) pair
        # bce(q,s) = mean_p[ -gt[s,p]*log(pred[q,p]) - (1-gt[s,p])*log(1-pred[q,p]) ]
        cost_bce = -(
            gt_flat @ torch.log(pred_flat + 1e-6).T  # (S, Q) → .T → (Q, S)
            + (1 - gt_flat) @ torch.log(1 - pred_flat + 1e-6).T
        ).T / (Hm * Wm)  # (Q, S)

        # Dice cost: 1 - 2*(pred·gt)/(pred.sum + gt.sum)
        numerator = 2.0 * pred_flat @ gt_flat.T  # (Q, S)
        denominator = pred_flat.sum(1, keepdim=True) + gt_flat.sum(1).unsqueeze(0)  # (Q, S)
        cost_dice = 1.0 - (numerator + 1.0) / (denominator + 1.0)  # (Q, S)

        cost = self.class_weight * cost_class + self.mask_weight * cost_bce + self.dice_weight * cost_dice
        return cast(torch.Tensor, cost.cpu())

    def _hungarian_match(self, cost: torch.Tensor) -> MatchResult:
        """Run scipy linear_sum_assignment on the cost matrix.

        Args:
            cost: (Q, S) float CPU tensor.

        Returns:
            MatchResult with matched query and GT indices.
        """
        from scipy.optimize import linear_sum_assignment

        cost_np = cost.numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        return MatchResult(
            query_indices=torch.as_tensor(row_ind, dtype=torch.long),
            gt_indices=torch.as_tensor(col_ind, dtype=torch.long),
        )

    def _forward_single(
        self,
        class_logits: torch.Tensor,
        mask_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Hungarian matching loss for one set of query predictions.

        Used for both the final decoder layer and each auxiliary layer.

        Args:
            class_logits: (B, Q, C+1) query class logits (includes no-object).
            mask_logits:  (B, Q, Hm, Wm) query mask logits (low-res).
            targets:      (B, H, W) int64 semantic label maps.

        Returns:
            Scalar loss tensor.
        """
        B = class_logits.shape[0]
        device = class_logits.device
        H, W = targets.shape[1], targets.shape[2]

        total_class_loss = torch.tensor(0.0, device=device)
        total_mask_loss = torch.tensor(0.0, device=device)
        total_dice_loss = torch.tensor(0.0, device=device)
        n_matched_total = 0

        for b in range(B):
            gt_classes, gt_masks = self._semantic_to_instances(targets[b])
            S = len(gt_classes)

            if S == 0:
                # No valid GT — all queries target the no-object class (index num_classes)
                no_obj = torch.full((self.num_queries,), self.num_classes, dtype=torch.long, device=device)
                total_class_loss = total_class_loss + F.cross_entropy(
                    class_logits[b],
                    no_obj,
                    weight=cast(torch.Tensor | None, self.ce_weights),
                )
                continue

            # --- Cost matrix and matching (cost uses only semantic class probs, no no-object)
            cost = self._compute_cost_matrix(class_logits[b], mask_logits[b], gt_classes, gt_masks)
            match = self._hungarian_match(cost)
            q_idx = match.query_indices.to(device)
            g_idx = match.gt_indices.to(device)
            n_matched_total += len(q_idx)

            # --- Classification loss
            # Matched queries → GT semantic class; unmatched → no-object class (index num_classes)
            tgt_cls = torch.full((self.num_queries,), self.num_classes, dtype=torch.long, device=device)
            gt_cls_tensor = torch.stack(gt_classes).to(device)  # (S,)
            tgt_cls[q_idx] = gt_cls_tensor[g_idx]
            total_class_loss = total_class_loss + F.cross_entropy(
                class_logits[b],
                tgt_cls,
                weight=cast(torch.Tensor | None, self.ce_weights),
            )

            # --- Mask losses (matched pairs only, full resolution)
            if len(q_idx) > 0:
                # Upsample matched mask logits to (H, W)
                matched_mask_logits = mask_logits[b][q_idx]  # (M, Hm, Wm)
                matched_mask_logits_up = F.interpolate(
                    matched_mask_logits.unsqueeze(1),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)  # (M, H, W)

                gt_masks_tensor = torch.stack([m.float() for m in gt_masks]).to(device)  # (S, H, W)
                matched_gt_masks = gt_masks_tensor[g_idx]  # (M, H, W)

                total_mask_loss = total_mask_loss + _sigmoid_bce_loss(matched_mask_logits_up, matched_gt_masks)
                total_dice_loss = total_dice_loss + _dice_loss(matched_mask_logits_up.sigmoid(), matched_gt_masks)

        # Average over batch
        loss = self.class_weight * total_class_loss / B + self.mask_weight * total_mask_loss / B + self.dice_weight * total_dice_loss / B
        return loss

    def forward(
        self,
        class_logits: torch.Tensor,
        mask_logits: torch.Tensor,
        targets: torch.Tensor,
        auxiliary_predictions: list[dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Compute Hungarian matching loss on the final layer and all auxiliary layers.

        Args:
            class_logits:            (B, Q, C+1) final query class logits.
            mask_logits:             (B, Q, Hm, Wm) final query mask logits.
            targets:                 (B, H, W) int64 semantic label maps.
            auxiliary_predictions:   List of per-layer dicts with keys
                                     'class_queries_logits' and 'masks_queries_logits'.
                                     Applied with aux_loss_weight (paper: 1.0).

        Returns:
            Scalar total loss.
        """
        loss = self._forward_single(class_logits, mask_logits, targets)

        if auxiliary_predictions:
            for aux in auxiliary_predictions:
                aux_cls = aux["class_queries_logits"]
                aux_mask = aux["masks_queries_logits"]
                loss = loss + self.aux_loss_weight * self._forward_single(aux_cls, aux_mask, targets)

        return loss
