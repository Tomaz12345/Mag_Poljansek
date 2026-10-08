# Implemented from: Jain et al., "OneFormer: One Transformer to Rule Universal Image Segmentation",
#   CVPR 2023 (arXiv:2211.06220)
# Reference implementation: https://github.com/SHI-Labs/OneFormer
# Wraps the Hugging Face transformers implementation (OneFormerForUniversalSegmentation).

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .input_projection import InputProjection
from .pretrained_loading import log_pretrained_load_result

logger = logging.getLogger(__name__)


ONEFORMER_VARIANTS: dict[str, dict[str, Any]] = {
    "swin_large": {
        "hf_checkpoint": "shi-labs/oneformer_cityscapes_swin_large",
    },
    "swin_base": {
        "hf_checkpoint": "shi-labs/oneformer_ade20k_swin_base",
    },
    "dinat_large": {
        "hf_checkpoint": "shi-labs/oneformer_cityscapes_dinat_large",
        # Note: requires 'natten' package for dilated neighborhood attention.
    },
}


class OneFormer(nn.Module):
    """Full OneFormer pipeline wrapped for semantic segmentation.

    Loads ``OneFormerForUniversalSegmentation`` from HuggingFace Hub and
    prepends a learned ``input_proj`` to handle multi-spectral inputs.
    The class/mask query outputs are fused into dense per-pixel logits
    compatible with the project's standard cross-entropy loss.

    Args:
        n_channels: Number of input channels (e.g. 13 for S1+S2).
        n_classes: Number of output segmentation classes.
        img_size: Input image size (square images assumed).
        use_pretrained: Whether to load pretrained HuggingFace weights.
        pretrained_path: HuggingFace Hub model ID.
        freeze_encoder: Freeze all encoder parameters.
        freeze_encoder_layers: Number of encoder layers to freeze.
        freeze_encoder_layers_mode: Freezing mode ('individual' or 'grouped').
        variant: Shortcut name in ONEFORMER_VARIANTS (overrides pretrained_path).
        oneformer_task: Task prompt string ('semantic', 'instance', 'panoptic').
        oneformer_task_seq_length: Max token length for task token encoding.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        img_size: int,
        use_pretrained: bool,
        pretrained_path: str,
        freeze_encoder: bool,
        freeze_encoder_layers: int,
        freeze_encoder_layers_mode: str,
        variant: str | None,
        oneformer_task: str,
        oneformer_task_seq_length: int,
        input_projection_type: str = "linear",
        input_projection_hidden: int = 32,
        input_projection_dims: str | list[int] | None = None,
    ) -> None:
        super().__init__()

        # Apply variant config (overrides pretrained_path)
        if variant is not None:
            variant_lower = str(variant).lower()
            if variant_lower in ONEFORMER_VARIANTS:
                vcfg = ONEFORMER_VARIANTS[variant_lower]
                pretrained_path = cast(str, vcfg["hf_checkpoint"])
                logger.info(f"Using OneFormer variant '{variant_lower}': {vcfg}")
            else:
                logger.warning(f"Unknown OneFormer variant '{variant_lower}', ignoring.")

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.pretrained_path = pretrained_path
        self.oneformer_task = oneformer_task
        self.oneformer_task_seq_length = oneformer_task_seq_length

        self._oneformer_tokenizer: Any = None

        # Project arbitrary input channels → 3 for the Swin backbone
        self.input_proj = InputProjection(
            n_channels=n_channels,
            target_channels=3,
            projection_type=input_projection_type,
            hidden_channels=input_projection_hidden,
            dims=input_projection_dims,
        )

        # HuggingFace full OneFormer model (initialised below)
        self.encoder: Any  # OneFormerForUniversalSegmentation

        if use_pretrained:
            self._load_pretrained(pretrained_path)
        else:
            self._build_random_encoder(pretrained_path)

        if freeze_encoder:
            self._freeze_encoder()
        elif freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers, mode=freeze_encoder_layers_mode)

        logger.info(
            f"OneFormer initialised: n_channels={n_channels}, n_classes={n_classes}, "
            f"variant={variant}, use_pretrained={use_pretrained}, task={oneformer_task}"
        )

    def _load_pretrained(self, pretrained_path: str) -> None:
        """Load full pretrained OneFormerForUniversalSegmentation weights from HuggingFace Hub."""
        try:
            from transformers import OneFormerForUniversalSegmentation

            logger.info(f"Loading OneFormer pretrained weights from '{pretrained_path}' …")
            result: Any = OneFormerForUniversalSegmentation.from_pretrained(
                pretrained_path,
                num_labels=self.n_classes,
                ignore_mismatched_sizes=True,
                output_loading_info=True,
            )
            self.encoder, loading_info = result
            log_pretrained_load_result(
                loading_info.get("missing_keys", []),
                loading_info.get("unexpected_keys", []),
                logger,
                model_name="OneFormer",
                expected_missing_prefixes=[
                    "model.transformer_module.decoder.class_predictor.",
                    "model.transformer_module.decoder.mask_embed.",
                ],
            )
            logger.info("Successfully loaded OneFormer pretrained weights.")
        except Exception as exc:
            logger.warning(f"Failed to load pretrained weights: {exc}")
            logger.info("Falling back to random initialisation.")
            self._build_random_encoder(pretrained_path)

    def _build_random_encoder(self, pretrained_path: str) -> None:
        """Build a randomly initialised OneFormerForUniversalSegmentation from the checkpoint config."""
        try:
            from transformers import OneFormerConfig, OneFormerForUniversalSegmentation

            logger.info(f"Downloading OneFormer config from '{pretrained_path}' (no weights) …")
            config = OneFormerConfig.from_pretrained(pretrained_path)
            config.num_labels = self.n_classes
            self.encoder = OneFormerForUniversalSegmentation(config)
            logger.info("OneFormer initialised with random weights.")
        except Exception as exc:
            raise RuntimeError(
                f"Could not build OneFormerForUniversalSegmentation from config at '{pretrained_path}': {exc}\n"
                "Ensure 'transformers>=4.40.0' is installed and the checkpoint ID is correct."
            ) from exc

    # Padding helpers for Swin window-attention size constraints

    def _get_size_multiple(self) -> int:
        """Return the spatial size multiple required by the Swin backbone."""
        try:
            cfg = self.encoder.config
            backbone_cfg = getattr(cfg, "backbone_config", cfg)
            patch_size = getattr(backbone_cfg, "patch_size", 4)
            window_size = getattr(backbone_cfg, "window_size", 8)
            return int(patch_size) * int(window_size)
        except Exception:
            return 32

    def _pad_to_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Pad spatial dims of x to the nearest multiple required by Swin."""
        multiple = self._get_size_multiple()
        _, _, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    def _build_task_inputs(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Build task token IDs for OneFormer task conditioning."""
        task_texts = [f"the task is {self.oneformer_task}"] * batch_size
        if self._oneformer_tokenizer is None:
            self._oneformer_tokenizer = self._load_tokenizer()

        tokenized: Any = self._oneformer_tokenizer(
            task_texts,
            padding="max_length",
            truncation=True,
            max_length=self.oneformer_task_seq_length,
            return_tensors="pt",
        )
        return cast(torch.Tensor, tokenized["input_ids"].to(device=device))

    def build_text_inputs(self, class_names: list[str], device: torch.device) -> torch.Tensor:
        """Tokenize class names for OneFormer query-text contrastive loss."""
        if self._oneformer_tokenizer is None:
            self._oneformer_tokenizer = self._load_tokenizer()

        tokenized: Any = self._oneformer_tokenizer(
            class_names,
            padding="max_length",
            truncation=True,
            max_length=self.oneformer_task_seq_length,
            return_tensors="pt",
        )
        return cast(torch.Tensor, tokenized["input_ids"].to(device=device))

    def _load_tokenizer(self) -> Any:
        """Load CLIPTokenizerFast (or CLIPTokenizer as fallback)."""
        try:
            from transformers import CLIPTokenizerFast  # type: ignore[attr-defined]

            return CLIPTokenizerFast.from_pretrained(self.pretrained_path)
        except Exception:
            from transformers import CLIPTokenizer

            return CLIPTokenizer.from_pretrained(self.pretrained_path)

    def forward(
        self,
        x: torch.Tensor,
        text_inputs: torch.Tensor | None = None,
        task_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass through the full OneFormer pipeline.

        During training (``self.training=True``) returns a dict with keys:
          - ``"main"``:         fused semantic logits (B, n_classes, H, W)
          - ``"class_logits"``: query class logits (B, Q, n_classes) — for Hungarian loss
          - ``"mask_logits"``:  query mask logits (B, Q, Hm, Wm)    — for Hungarian loss

        During inference (``self.training=False``) returns only the fused
        semantic logits tensor (B, n_classes, H, W) so that validation,
        metrics, and prediction pipelines work unchanged.

        Args:
            x: Input tensor of shape (B, C, H, W).
            text_inputs: Optional pre-built text token IDs (B, n_classes, seq_len).
            task_inputs: Optional pre-built task token IDs (B, seq_len).
        """
        _B, _C, H, W = x.shape

        # 1. Project to 3 channels for Swin backbone
        x_3ch = self.input_proj(x)  # (B, 3, H, W)

        # 2. Pad to Swin window-size multiple
        x_padded, pad_h, pad_w = self._pad_to_multiple(x_3ch)

        # 3. Build task token if not provided
        if task_inputs is None:
            task_inputs = self._build_task_inputs(x.shape[0], device=x.device)

        # 4. Build pixel mask (False on padded regions)
        pixel_mask = torch.ones(
            (x_padded.shape[0], x_padded.shape[2], x_padded.shape[3]),
            dtype=torch.bool,
            device=x.device,
        )
        if pad_h > 0:
            pixel_mask[:, -pad_h:, :] = False
        if pad_w > 0:
            pixel_mask[:, :, -pad_w:] = False

        # 5. Run full OneFormer pipeline
        # output_hidden_states=True makes the model return auxiliary_predictions
        # (class + mask logits from each transformer decoder layer) needed for
        # the auxiliary Hungarian matching loss applied during training.
        outputs = self.encoder(
            pixel_values=x_padded,
            task_inputs=task_inputs,
            text_inputs=text_inputs,
            pixel_mask=pixel_mask,
            return_dict=True,
            output_hidden_states=self.training,
        )

        # 6. Extract class and mask query logits
        class_logits = getattr(outputs, "class_queries_logits", None)
        if class_logits is None:
            class_logits = getattr(outputs, "transformer_decoder_class_predictions", None)
        mask_logits = getattr(outputs, "masks_queries_logits", None)
        if mask_logits is None:
            mask_logits = getattr(outputs, "transformer_decoder_mask_predictions", None)
        if class_logits is None or mask_logits is None:
            raise RuntimeError("OneFormer outputs did not include query class/mask predictions.")

        # Remove no-object class if present (use only semantic classes for fusion)
        if class_logits.shape[-1] == self.n_classes + 1:
            class_logits_sem = class_logits[..., :-1]
        elif class_logits.shape[-1] != self.n_classes:
            logger.warning(
                f"OneFormer class logits dim {class_logits.shape[-1]} does not match n_classes={self.n_classes}. Slicing to first n_classes."
            )
            class_logits_sem = class_logits[..., : self.n_classes]
        else:
            class_logits_sem = class_logits

        # 7. Fuse into dense per-pixel semantic logits
        mask_logits_up = F.interpolate(mask_logits, size=(H, W), mode="bilinear", align_corners=False)
        fused: torch.Tensor = torch.einsum("bqc,bqhw->bchw", class_logits_sem, mask_logits_up)

        if self.training:
            # Return raw query outputs for Hungarian matching loss.
            # class_logits keeps the no-object class (C+1) so unmatched queries
            # are trained to predict it explicitly, matching the paper design.
            # auxiliary_predictions contains per-layer outputs for aux losses.
            aux: list[dict[str, torch.Tensor]] = list(getattr(outputs, "auxiliary_predictions", None) or [])
            return {
                "main": fused,
                "class_logits": class_logits,  # (B, Q, C+1)
                "mask_logits": mask_logits,  # (B, Q, Hm, Wm)
                "auxiliary_predictions": aux,  # list of {class_queries_logits, masks_queries_logits}
            }
        return fused

    # Parameter groups for differential learning rates

    def get_encoder_params(self) -> Iterator[nn.Parameter]:
        """Encoder params: input_proj + Swin backbone (lower LR for fine-tuning)."""
        yield from self.input_proj.parameters()
        yield from self.encoder.model.pixel_level_module.encoder.parameters()

    def get_decoder_params(self) -> Iterator[nn.Parameter]:
        """Decoder params: pixel decoder + transformer module (incl. class/mask embed heads) (base LR)."""
        yield from self.encoder.model.pixel_level_module.decoder.parameters()
        yield from self.encoder.model.transformer_module.parameters()

    # Encoder freezing

    def _get_swin_backbone(self) -> nn.Module:
        return cast(nn.Module, self.encoder.model.pixel_level_module.encoder)

    def _freeze_encoder(self) -> None:
        """Freeze all encoder (input_proj + Swin backbone) parameters."""
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for param in self._get_swin_backbone().parameters():
            param.requires_grad = False
        logger.info("OneFormer encoder weights frozen.")

    def _freeze_encoder_layers(self, n_layers: int, mode: str = "individual") -> None:
        """Freeze first N layers of the encoder.

        Layer numbering:
          - Layer 0: input_proj
          - Layers 1-4: Swin backbone stages
          - 'grouped' mode: groups the 4 backbone stages into 2 logical groups.

        Args:
            n_layers: Number of layers/groups to freeze.
            mode: 'individual' or 'grouped'.
        """
        if n_layers <= 0:
            return

        try:
            backbone_stages: list[nn.Module] = cast(list[nn.Module], self._get_swin_backbone().model.encoder.layers)  # type: ignore[union-attr]
        except AttributeError:
            logger.warning("Could not access Swin backbone stages for layer freezing. Freezing input_proj only.")
            backbone_stages = []

        if mode == "individual":
            max_layers = 1 + len(backbone_stages)
            n_layers = min(n_layers, max_layers)

            frozen_count = 0
            if n_layers > 0:
                for param in self.input_proj.parameters():
                    param.requires_grad = False
                frozen_count = 1

            for i in range(min(n_layers - 1, len(backbone_stages))):
                for param in backbone_stages[i].parameters():
                    param.requires_grad = False
                frozen_count += 1

            logger.info(f"Froze {frozen_count} OneFormer encoder layers (out of {max_layers}, individual mode).")

        elif mode == "grouped":
            num_stages = len(backbone_stages)
            group_size = max(1, num_stages // 2)
            max_groups = 3  # input_proj + 2 stage groups
            n_layers = min(n_layers, max_groups)

            frozen_count = 0
            if n_layers > 0:
                for param in self.input_proj.parameters():
                    param.requires_grad = False
                frozen_count = 1

            for group_idx in range(1, n_layers):
                start = (group_idx - 1) * group_size
                end = min(group_idx * group_size, num_stages)
                for stage_idx in range(start, end):
                    for param in backbone_stages[stage_idx].parameters():
                        param.requires_grad = False
                frozen_count += 1

            logger.info(f"Froze {frozen_count} OneFormer encoder groups (out of {max_groups}, grouped mode).")

        else:
            logger.warning(f"Unknown freeze_encoder_layers_mode '{mode}', using 'individual'.")
            self._freeze_encoder_layers(n_layers, mode="individual")
