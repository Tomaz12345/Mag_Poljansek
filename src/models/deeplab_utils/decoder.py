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
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.messages import MODEL_ERRORS
from src.utils import log_and_raise

# from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

logger = logging.getLogger(__name__)


class SeparableConvBnReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, BatchNorm: type[nn.Module]) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.pointwise(y)
        y = self.bn(y)
        y = self.relu(y)
        return cast(torch.Tensor, y)


class Decoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str,
        BatchNorm: type[nn.Module],
        low_level_channels: int | None = None,
        out_channels: int | None = None,
        aspp_out_channels: int | None = None,
        use_depthwise_separable: bool = True,
    ) -> None:
        super().__init__()

        if backbone != "resnet":
            msg = MODEL_ERRORS.PARAMETER_VALUE_NOT_SUPPORTED.format(value=backbone, parameter="backbone")
            log_and_raise(NotImplementedError, msg, logger_instance=logger)
        low_level_inplanes = 256

        # Use provided values or fall back to centralized defaults
        low_level_ch = low_level_channels if low_level_channels is not None else 48
        out_ch = out_channels if out_channels is not None else 256
        aspp_ch = aspp_out_channels if aspp_out_channels is not None else 256

        combined_ch = aspp_ch + low_level_ch

        self.conv1 = nn.Conv2d(
            low_level_inplanes,
            low_level_ch,
            1,
            bias=False,
        )
        self.bn1 = BatchNorm(low_level_ch)
        self.relu = nn.ReLU()

        if use_depthwise_separable:
            conv_blocks: list[nn.Module] = [
                SeparableConvBnReLU(combined_ch, out_ch, BatchNorm),
                SeparableConvBnReLU(out_ch, out_ch, BatchNorm),
            ]
        else:
            conv_blocks = [
                nn.Conv2d(
                    combined_ch,
                    out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                BatchNorm(out_ch),
                nn.ReLU(),
                nn.Conv2d(
                    out_ch,
                    out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                BatchNorm(out_ch),
                nn.ReLU(),
            ]

        self.last_conv = nn.Sequential(
            *conv_blocks,
            nn.Conv2d(out_ch, num_classes, kernel_size=1, stride=1),
        )
        self._init_weight()

    def forward(self, x: torch.Tensor, low_level_feat: torch.Tensor) -> torch.Tensor:
        low = self.conv1(low_level_feat)
        low = self.bn1(low)
        low = self.relu(low)

        y = F.interpolate(x, size=low.size()[2:], mode="bilinear", align_corners=True)
        y = torch.cat((y, low), dim=1)
        y = self.last_conv(y)

        return cast(torch.Tensor, y)

    def _init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            # Only standard BatchNorm2d is supported (SynchronizedBatchNorm2d disabled)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(
    num_classes: int,
    backbone: str,
    BatchNorm: type[nn.Module],
    low_level_channels: int | None = None,
    out_channels: int | None = None,
    aspp_out_channels: int | None = None,
    use_depthwise_separable: bool = True,
) -> Decoder:
    return Decoder(
        num_classes,
        backbone,
        BatchNorm,
        low_level_channels=low_level_channels,
        out_channels=out_channels,
        aspp_out_channels=aspp_out_channels,
        use_depthwise_separable=use_depthwise_separable,
    )
