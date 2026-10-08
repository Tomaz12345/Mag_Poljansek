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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        BatchNorm: type[nn.Module],
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
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


class _ASPPModule(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        BatchNorm: type[nn.Module],
        use_depthwise_separable: bool,
    ) -> None:
        super().__init__()
        self.block: nn.Module
        if use_depthwise_separable and kernel_size == 3:
            self.block = SeparableConvBnReLU(
                in_channels=inplanes,
                out_channels=planes,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                BatchNorm=BatchNorm,
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                ),
                BatchNorm(planes),
                nn.ReLU(),
            )

        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return cast(torch.Tensor, y)

    def _init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            # Only standard BatchNorm2d is supported (SynchronizedBatchNorm2d disabled)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(
        self,
        backbone: str,
        output_stride: int,
        BatchNorm: type[nn.Module],
        out_channels: int | None = None,
        use_depthwise_separable: bool = True,
    ) -> None:
        super().__init__()

        if backbone != "resnet":
            msg = MODEL_ERRORS.PARAMETER_VALUE_NOT_SUPPORTED.format(value=backbone, parameter="backbone")
            log_and_raise(NotImplementedError, msg, logger_instance=logger)
        inplanes = 2048

        # Use provided values or fall back to centralized defaults
        out_ch = out_channels if out_channels is not None else 256

        # Combined channels = 5 * out_channels (5 ASPP branches)
        combined_channels = 5 * out_ch

        # Use centralized dilation rate configuration
        if output_stride == 16:
            dilations = (1, 6, 12, 18)
        elif output_stride == 8:
            dilations = (1, 12, 24, 36)
        else:
            msg = MODEL_ERRORS.PARAMETER_VALUE_NOT_SUPPORTED.format(value=output_stride, parameter="output_stride")
            log_and_raise(NotImplementedError, msg, logger_instance=logger)

        self.aspp1 = _ASPPModule(
            inplanes,
            out_ch,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm,
            use_depthwise_separable=use_depthwise_separable,
        )
        self.aspp2 = _ASPPModule(
            inplanes,
            out_ch,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
            use_depthwise_separable=use_depthwise_separable,
        )
        self.aspp3 = _ASPPModule(
            inplanes,
            out_ch,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
            use_depthwise_separable=use_depthwise_separable,
        )
        self.aspp4 = _ASPPModule(
            inplanes,
            out_ch,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm,
            use_depthwise_separable=use_depthwise_separable,
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, out_ch, 1, stride=1, bias=False),
            BatchNorm(out_ch),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(combined_channels, out_ch, 1, bias=False)
        self.bn1 = BatchNorm(out_ch)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        y = torch.cat((x1, x2, x3, x4, x5), dim=1)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        return cast(torch.Tensor, y)

    def _init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            # Only standard BatchNorm2d is supported (SynchronizedBatchNorm2d disabled)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(
    backbone: str,
    output_stride: int,
    BatchNorm: type[nn.Module],
    out_channels: int | None = None,
    use_depthwise_separable: bool = True,
) -> ASPP:
    return ASPP(
        backbone,
        output_stride,
        BatchNorm,
        out_channels=out_channels,
        use_depthwise_separable=use_depthwise_separable,
    )
