from __future__ import annotations

import torch
import torch.nn as nn

# GroupNorm groups used between the two transposed convolutions of the 4x branch.
_GROUP_NORM_GROUPS: int = 32


class ViTMultiScaleNeck(nn.Module):
    """Rescale four same-stride ViT feature maps into a 4-level pyramid.

    The four inputs are transformed as::

        f0 -> ConvTranspose2d(2x) -> GroupNorm -> GELU -> ConvTranspose2d(2x)   (4x up)
        f1 -> ConvTranspose2d(2x)                                               (2x up)
        f2 -> Identity                                                          (1x)
        f3 -> MaxPool2d(2)                                                      (2x down)

    Args:
        embed_dim: Channel dimension of every input (and output) feature map.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        num_groups = min(_GROUP_NORM_GROUPS, embed_dim)
        while embed_dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups, embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.up2 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.identity = nn.Identity()
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Rescale the feature pyramid.

        Args:
            features: Exactly four tensors of shape (B, embed_dim, H, W).

        Returns:
            Four tensors at 4x, 2x, 1x and 0.5x the input resolution.

        Raises:
            ValueError: If ``features`` does not contain exactly four maps.
        """
        if len(features) != 4:
            raise ValueError(f"ViTMultiScaleNeck expects exactly 4 feature maps, got {len(features)}.")

        return [
            self.up4(features[0]),
            self.up2(features[1]),
            self.identity(features[2]),
            self.down2(features[3]),
        ]
