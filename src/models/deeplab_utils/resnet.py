from __future__ import annotations

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
# from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import logging
import math
from typing import Any, cast

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from src.constants import RESNET_CONFIG
from src.messages import MODEL_ERRORS
from src.utils import log_and_raise

logger = logging.getLogger(__name__)


class Bottleneck(nn.Module):
    expansion = RESNET_CONFIG["BOTTLENECK_EXPANSION"]

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        BatchNorm: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * RESNET_CONFIG["BOTTLENECK_EXPANSION"], kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * RESNET_CONFIG["BOTTLENECK_EXPANSION"])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return cast(torch.Tensor, out)


class ResNet(nn.Module):
    def __init__(
        self,
        block: type[Bottleneck],
        layers: tuple[int, ...],
        output_stride: int,
        BatchNorm: type[nn.Module],
        pretrained: bool = True,
        n_in: int = 3,
    ) -> None:
        super().__init__()
        self.inplanes: int = RESNET_CONFIG["INITIAL_CHANNELS"]
        blocks = (1, 2, 4)
        # Use centralized stride and dilation configuration
        if output_stride == 16:
            strides = (1, 2, 2, 1)
            dilations = (1, 1, 1, 2)
        elif output_stride == 8:
            strides = (1, 2, 1, 1)
            dilations = (1, 1, 2, 4)
        else:
            msg = MODEL_ERRORS.PARAMETER_VALUE_NOT_SUPPORTED.format(value=output_stride, parameter="output_stride")
            log_and_raise(NotImplementedError, msg, logger_instance=logger)

        # Modules - use centralized initial layer configuration
        self.conv1 = nn.Conv2d(
            n_in,
            RESNET_CONFIG["INITIAL_CHANNELS"],
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = BatchNorm(RESNET_CONFIG["INITIAL_CHANNELS"])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3],
        # dilation=dilations[3],  BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(
        self,
        block: type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        BatchNorm: type[nn.Module] = nn.BatchNorm2d,
    ) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(planes * block.expansion),
            )

        layer_list: list[nn.Module] = []
        layer_list.append(cast(nn.Module, block(self.inplanes, planes, stride, dilation, downsample, BatchNorm)))
        self.inplanes = planes * block.expansion
        for _i in range(1, blocks):
            layer_list.append(cast(nn.Module, block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm)))

        return nn.Sequential(*layer_list)

    def _make_MG_unit(
        self,
        block: type[Bottleneck],
        planes: int,
        blocks: tuple[int, ...],
        stride: int = 1,
        dilation: int = 1,
        BatchNorm: type[nn.Module] = nn.BatchNorm2d,
    ) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(planes * block.expansion),
            )

        layer_list: list[nn.Module] = []
        layer_list.append(
            cast(
                nn.Module,
                block(
                    self.inplanes,
                    planes,
                    stride,
                    dilation=blocks[0] * dilation,
                    downsample=downsample,
                    BatchNorm=BatchNorm,
                ),
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layer_list.append(
                cast(
                    nn.Module,
                    block(
                        self.inplanes,
                        planes,
                        stride=1,
                        dilation=blocks[i] * dilation,
                        BatchNorm=BatchNorm,
                    ),
                )
            )

        return nn.Sequential(*layer_list)

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            # Only standard BatchNorm2d is supported (SynchronizedBatchNorm2d disabled)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self) -> None:
        pretrain_dict: dict[str, Any] = model_zoo.load_url("https://download.pytorch.org/models/resnet101-5d3b4d8f.pth")
        model_dict: dict[str, Any] = {}
        state_dict: dict[str, Any] = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def ResNet101(
    output_stride: int,
    BatchNorm: type[nn.Module],
    pretrained: bool = True,
    n_in: int = 3,
) -> nn.Module:
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model: nn.Module = ResNet(
        Bottleneck,
        (3, 4, 23, 3),
        output_stride,
        BatchNorm,
        pretrained=pretrained,
        n_in=n_in,
    )
    return model


if __name__ == "__main__":
    import torch

    model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    logger.info(output.size())
    logger.info(low_level_feat.size())
