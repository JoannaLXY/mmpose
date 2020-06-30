import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      normal_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.utils import get_root_logger
from ..builder import BACKBONES
from .base_backbone import BaseBackbone


class SCConv(nn.Module):
    """SCConv for SCNet

    Args:
        planes (int): number of input channels
        stride (int): stride of SCConv
        pooling_r (int): size of pooling
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 planes,
                 stride,
                 pooling_r,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.1)):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.k3 = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, planes)[1],
        )
        self.k4 = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x),
                                              identity.size()[2:])))
        out = torch.mul(self.k3(x), out)
        out = self.k4(out)

        return out


class SCBottleneck(nn.Module):
    """SCBottleneck for SCNet

    Args:
        inplanes (int): Number of channels for the input in first
                        conv2d layer.
        planes (int): Number of channels produced by some norm/conv2d
                        layers.
        stride (int): Stride of SCConv.
        downsample (int): Whether to downsample.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    expansion = 4
    pooling_r = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.1)):

        super(SCBottleneck, self).__init__()
        planes = int(planes / 2)

        self.conv1_a = build_conv_layer(
            conv_cfg, inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1_a = build_norm_layer(norm_cfg, planes)[1]

        self.k1 = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False),
            build_norm_layer(norm_cfg, planes)[1], nn.ReLU(inplace=True))

        self.conv1_b = build_conv_layer(
            conv_cfg, inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1_b = build_norm_layer(norm_cfg, planes)[1]

        self.scconv = SCConv(planes, stride, self.pooling_r, conv_cfg,
                             norm_cfg)

        self.conv3 = build_conv_layer(
            conv_cfg,
            planes * 2,
            planes * 2 * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False)
        self.bn3 = build_norm_layer(norm_cfg, planes * 2 * self.expansion)[1]
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_a = self.relu(out_a)

        out_a = self.k1(out_a)

        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_b = self.relu(out_b)

        out_b = self.scconv(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


@BACKBONES.register_module()
class SCNet(BaseBackbone):
    """SCNet backbone.

    Improving Convolutional Networks with Self-Calibrated Convolutions,
    Jiang-Jiang Liu, Qibin Hou, Ming-Ming Cheng, Changhu Wang, Jiashi Feng,
    IEEE CVPR, 2020.
    http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf

    Args:
        depth (int): depth of SCNet
        in_channels (int): number of input channels
    """

    arch_settings = {
        50: (SCBottleneck, [3, 4, 6, 3]),
        101: (SCBottleneck, [3, 4, 23, 3])
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):

        super(SCNet, self).__init__()

        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg

        block, layers = self.arch_settings[depth]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, self.inplanes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.conv_cfg,
                  self.norm_cfg))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')
