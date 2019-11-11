import math

import torch
import torch.nn as nn


class Conv2dBatchActivate(nn.Module):
    """
    https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(Conv2dBatchActivate, self).__init__()
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
        super(Conv2dBatch, self).__init__()
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
        super(ActivateConv2dBatch, self).__init__()
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
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class Concatenate(nn.Module):
    """
    https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py
    """

    def __init__(self, tensors):
        super(Concatenate, self).__init__()
        self.tensors = tensors

    def forward(self, x):
        x = torch.cat([t(x) for t in self.tensors])
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
