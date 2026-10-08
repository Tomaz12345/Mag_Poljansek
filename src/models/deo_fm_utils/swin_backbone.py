# Implemented from: Wolf et al., "Brewing Stronger Features: Dual-Teacher Distillation for
#   Multispectral Earth Observation", CVPR 2026 (arXiv:2602.19863)
# Reference implementation: https://github.com/wolfilip/DEO-FM
# The Swin-B trunk is torchvision.models.swin_b.

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from torchvision.models.swin_transformer import SwinTransformer
from torchvision.ops.misc import Permute

# swin_b config (from torchvision/models/swin_transformer.py::swin_b),
# overridden with window_size=12 to match the DEO checkpoint.
_SWIN_B_CONFIG: dict[str, object] = {
    "patch_size": [4, 4],
    "embed_dim": 128,
    "depths": [2, 2, 18, 2],
    "num_heads": [4, 8, 16, 32],
    "window_size": [12, 12],
    "stochastic_depth_prob": 0.5,
}


class DEOSwinBackbone(nn.Module):
    """Dual-stem Swin-B trunk emitting 4 multi-scale feature maps."""

    # Channel dims of the 4 emitted stages for swin_b (embed_dim=128).
    feature_dims: tuple[int, int, int, int] = (128, 256, 512, 1024)

    def __init__(self) -> None:
        super().__init__()

        self.backbone = SwinTransformer(**_SWIN_B_CONFIG)

        # Remove the original 3-channel stem; replace with two parallel stems.
        # Read embed_dim from the first SwinTransformerBlock (now at index 0
        # after the stem deletion). This mirrors upstream DEO_weights.py.
        del self.backbone.features[0]
        embed_dim = self.backbone.features[0][0].norm1.normalized_shape[0]

        norm = partial(nn.LayerNorm, eps=1e-5)
        self.backbone.conv_ms = nn.Sequential(
            nn.Conv2d(10, embed_dim, kernel_size=(4, 4), stride=(4, 4)),
            Permute([0, 2, 3, 1]),
            norm(embed_dim),
        )
        self.backbone.conv_rgb = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=(4, 4), stride=(4, 4)),
            Permute([0, 2, 3, 1]),
            norm(embed_dim),
        )

    def forward(self, x: torch.Tensor, *, use_ms: bool) -> list[torch.Tensor]:
        x = self.backbone.conv_ms(x) if use_ms else self.backbone.conv_rgb(x)
        features: list[torch.Tensor] = []
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i in (0, 2, 4, 6):
                # x is (B, H, W, C) in torchvision swin; convert to (B, C, H, W).
                features.append(x.permute(0, 3, 1, 2).contiguous())
        return features
