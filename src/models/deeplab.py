# author: Jianfeng Zhang (https://github.com/jfzhang95/pytorch-deeplab-xception)
#
# MIT License
#
# Copyright (c) 2018 Pyjcsx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
from collections.abc import Iterator
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.messages import MODEL_INIT_ERRORS
from src.models.deeplab_utils import resnet
from src.models.deeplab_utils.aspp import build_aspp
from src.models.deeplab_utils.decoder import build_decoder
from src.models.input_projection import InputProjection
from src.utils import log_and_raise

logger = logging.getLogger(__name__)

VALID_BACKBONES = ("resnet",)
VALID_OUTPUT_STRIDES = (8, 16)


class DeepLab(nn.Module):
    def __init__(
        self,
        backbone: str,
        use_pretrained: bool,
        output_stride: int,
        num_classes: int,
        sync_bn: bool,
        freeze_bn: bool,
        freeze_encoder: bool,
        freeze_encoder_layers: int,
        n_in: int,
        aspp_out_channels: int | None,
        decoder_low_level_channels: int | None,
        decoder_out_channels: int | None,
        use_depthwise_separable: bool,
        interpolation_mode: str,
        align_corners: bool,
        input_projection_type: str = "linear",
        input_projection_hidden: int = 32,
        input_projection_dims: str | list[int] | None = None,
    ) -> None:
        super().__init__()

        # Validate parameters
        if num_classes <= 0:
            msg = MODEL_INIT_ERRORS.NUM_CLASSES_POSITIVE.format(num_classes=num_classes)
            log_and_raise(ValueError, msg, logger_instance=logger)
        if n_in <= 0:
            msg = MODEL_INIT_ERRORS.NUM_CHANNELS_POSITIVE.format(n_channels=n_in)
            log_and_raise(ValueError, msg, logger_instance=logger)
        if backbone not in VALID_BACKBONES:
            msg = f"backbone must be one of {VALID_BACKBONES}, got '{backbone}'"
            log_and_raise(ValueError, msg, logger_instance=logger)
        if output_stride not in VALID_OUTPUT_STRIDES:
            msg = f"output_stride must be 8 or 16, got {output_stride}. Use 8 for higher resolution output, 16 for faster inference."
            log_and_raise(ValueError, msg, logger_instance=logger)
        if interpolation_mode not in ("bilinear", "nearest"):
            msg = f"interpolation_mode must be 'bilinear' or 'nearest', got '{interpolation_mode}'"
            log_and_raise(ValueError, msg, logger_instance=logger)

        # Store forward pass configuration
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners if interpolation_mode == "bilinear" else None

        # NOTE: Synchronized BatchNorm is intentionally disabled
        # Standard BatchNorm2d is used instead for simplicity
        if sync_bn:
            msg = "Synchronized BatchNorm is not supported. Use sync_bn=False."
            log_and_raise(NotImplementedError, msg, logger_instance=logger)
        else:
            BatchNorm = nn.BatchNorm2d

        # Input projection: n_in -> 3 (backbone always receives 3-channel input)
        self.input_proj = InputProjection(
            n_channels=n_in,
            target_channels=3,
            projection_type=input_projection_type,
            hidden_channels=input_projection_hidden,
            dims=input_projection_dims,
        )

        self.backbone = resnet.ResNet101(output_stride, BatchNorm, pretrained=use_pretrained, n_in=3)
        self.aspp = build_aspp(
            backbone,
            output_stride,
            BatchNorm,
            out_channels=aspp_out_channels,
            use_depthwise_separable=use_depthwise_separable,
        )
        self.decoder = build_decoder(
            num_classes,
            backbone,
            BatchNorm,
            low_level_channels=decoder_low_level_channels,
            out_channels=decoder_out_channels,
            aspp_out_channels=aspp_out_channels,
            use_depthwise_separable=use_depthwise_separable,
        )

        self.freeze_bn_flag = freeze_bn

        # Freeze encoder if requested (freeze_encoder=True takes precedence)
        if freeze_encoder:
            self._freeze_encoder()
        elif freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(input)
        x, low_level_feat = self.backbone(projected)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(
            x,
            size=input.size()[2:],
            mode=self.interpolation_mode,
            align_corners=self.align_corners,
        )

        return cast(torch.Tensor, x)

    def freeze_bn_layers(self) -> None:
        for m in self.modules():
            # Only standard BatchNorm2d is supported (SynchronizedBatchNorm2d disabled)
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self) -> Iterator[torch.nn.Parameter]:
        modules: list[nn.Module] = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn_flag:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            yield p
                else:
                    # Only standard BatchNorm2d is supported (SynchronizedBatchNorm2d disabled)
                    if isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d)):
                        for p in m[1].parameters():
                            yield p

    def get_encoder_params(self) -> Iterator[torch.nn.Parameter]:
        yield from self.backbone.parameters()

    def get_decoder_params(self) -> Iterator[torch.nn.Parameter]:
        """Get decoder parameters (ASPP + decoder head)."""
        yield from self.aspp.parameters()
        yield from self.decoder.parameters()

    def _freeze_encoder(self) -> None:
        """Freeze encoder (backbone) parameters so they are not updated during training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Encoder weights frozen")

    def _freeze_encoder_layers(self, n_layers: int) -> None:
        """Freeze first N layers of the encoder.

        For ResNet101 backbone:
        - Layer 0: conv1 + bn1 (initial feature extraction)
        - Layer 1: layer1 (64 channels)
        - Layer 2: layer2 (128 channels)
        - Layer 3: layer3 (256 channels)
        - Layer 4: layer4 (512 channels, bottleneck)

        Args:
            n_layers: Number of layers to freeze (1-5)
        """
        if n_layers <= 0:
            return

        max_layers = 5
        if n_layers > max_layers:
            logger.warning(f"freeze_encoder_layers={n_layers} exceeds max layers {max_layers}, using {max_layers}")
            n_layers = max_layers

        # Get ResNet components in order
        backbone = self.backbone
        layer_groups: list[tuple[str, list[nn.Module]]] = [
            ("conv1+bn1", [cast(nn.Module, backbone.conv1), cast(nn.Module, backbone.bn1)]),
            ("layer1", [cast(nn.Module, backbone.layer1)]),
            ("layer2", [cast(nn.Module, backbone.layer2)]),
            ("layer3", [cast(nn.Module, backbone.layer3)]),
            ("layer4", [cast(nn.Module, backbone.layer4)]),
        ]

        frozen_count = 0
        for i in range(n_layers):
            layer_name, modules = layer_groups[i]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
            frozen_count += 1
            logger.debug(f"Froze encoder layer {i}: {layer_name}")

        logger.info(f"Froze first {frozen_count} encoder layers (out of {max_layers})")

    def get_frozen_layer_info(self) -> dict[str, int | list[str]]:
        """Get information about frozen layers for logging.

        Returns:
            Dictionary with total_layers, frozen_layers count,
            and list of frozen layer names
        """
        backbone = self.backbone
        layer_groups: list[tuple[str, list[nn.Module]]] = [
            ("conv1+bn1", [cast(nn.Module, backbone.conv1), cast(nn.Module, backbone.bn1)]),
            ("layer1", [cast(nn.Module, backbone.layer1)]),
            ("layer2", [cast(nn.Module, backbone.layer2)]),
            ("layer3", [cast(nn.Module, backbone.layer3)]),
            ("layer4", [cast(nn.Module, backbone.layer4)]),
        ]

        frozen_layers = []
        for layer_name, modules in layer_groups:
            is_frozen = all(not param.requires_grad for module in modules for param in module.parameters())
            if is_frozen:
                frozen_layers.append(layer_name)

        return {
            "total_layers": len(layer_groups),
            "frozen_layers": len(frozen_layers),
            "frozen_layer_names": frozen_layers,
        }

    def get_10x_lr_params(self) -> Iterator[torch.nn.Parameter]:
        modules: list[nn.Module] = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn_flag:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            yield p
                else:
                    # Only standard BatchNorm2d is supported (SynchronizedBatchNorm2d disabled)
                    if isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d)):
                        for p in m[1].parameters():
                            yield p
