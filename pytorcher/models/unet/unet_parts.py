""" Parts of the U-Net model """

import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F

from pytorcher.layers import SeparableConv2d, SinogramConv2d, SinogramSeparableConv2d


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, layer_type='Conv2d', **kwargs):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        assert layer_type in globals().keys(), f"Layer type '{layer_type}' is not recognized."
        conv_layer = globals()[layer_type]
        self.double_conv = nn.Sequential(
            conv_layer(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, **kwargs),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            conv_layer(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, layer_type='Conv2d', **kwargs):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, layer_type=layer_type, **kwargs)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, layer_type='Conv2d', **kwargs):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, layer_type=layer_type)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, layer_type=layer_type)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_layer_type='Conv2d', **kwargs):
        super(OutConv, self).__init__()
        conv_layer = globals()[conv_layer_type]
        self.conv = conv_layer(in_channels, out_channels, kernel_size=1, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        x = nn.Sigmoid()(x)
        return x