import math

import torch
import torch.nn as nn


class Conv2dBatchActivate(nn.Module):
    """
    https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )  # verify bias false
        self.bn = nn.BatchNorm2d(
            out_channels,
            eps=0.001,  # value found in tensorflow
            momentum=0.1,  # default pytorch value
            affine=True,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2dBatch(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )  # verify bias false
        self.bn = nn.BatchNorm2d(
            out_channels,
            eps=0.001,  # value found in tensorflow
            momentum=0.1,  # default pytorch value
            affine=True,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ActivateConv2dBatch(nn.Module):
    """
    https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )  # verify bias false
        self.bn = nn.BatchNorm2d(
            out_channels,
            eps=0.001,  # value found in tensorflow
            momentum=0.1,  # default pytorch value
            affine=True,
        )

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class Concatenate(nn.Module):
    """
    https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py
    """

    def __init__(self, tensors):
        super().__init__()
        self.tensors = tensors

    def forward(self, x):
        x = torch.cat([t(x) for t in self.tensors])
        return x


class SeparableConv2d(nn.Module):
    """
    https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py#L50
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class ActivateSeparableConv2dBatch(nn.Module):
    """
    https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py#L50
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv = SeparableConv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(
            out_channels,
            eps=0.001,  # value found in tensorflow
            momentum=0.1,  # default pytorch value
            affine=True,
        )

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


def conv2d_out_shape(width, height, Conv2d):
    """
    return (C , W , H)
    C: channels
    W: Width
    H: Height
    """
    # taken from:
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d

    _h = math.floor(((
        height
        + (2 * Conv2d.padding[0])
        - Conv2d.dilation[0] * (Conv2d.kernel_size[0] - 1)
        - 1
    ) / Conv2d.stride[0]) + 1)

    _w = math.floor(((
        width
        + (2 * Conv2d.padding[1])
        - Conv2d.dilation[1] * (Conv2d.kernel_size[1] - 1)
        - 1
    ) / Conv2d.stride[1]) + 1)

    return (Conv2d.out_channels, _w, _h)
