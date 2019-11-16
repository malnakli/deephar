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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
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


class SeparableConv2d(nn.Module):
    """
    https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py#L50
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)

        self.weights = [self.conv1.weight, self.pointwise.weight]

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class ActivateSeparableConv2dBatch(nn.Module):

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


class SeparableResidualModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = ActivateConv2dBatch(
            in_channels, out_channels, kernel_size=1, stride=1)

        self.conv2 = ActivateSeparableConv2dBatch(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        if self.in_channels == self.out_channels:
            x1 = x
        else:
            x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.add(x1, x2)
        return x


class Lambda(nn.Module):

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class SoftArgMax2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()

        self.num_rows = kernel_size if isinstance(
            kernel_size, int) else kernel_size[0]

        self.num_cols = kernel_size if isinstance(
            kernel_size, int) else kernel_size[1]

        self.num_filters = out_channels
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self._softmax(x)

        x_x = self._lin_interpolation_2d(x, dim=0)
        x_y = self._lin_interpolation_2d(x, dim=1)

        x = torch.cat([x_x, x_y], dim=-1)
        return x

    def _softmax(self, x):
        if x.ndim != 4:
            raise ValueError('This function is specific for 4D tensors. '
                             'Here, ndim=' + str(x.ndim))

        e = torch.exp(x - torch.max(x, dim=(1, 2), keepdim=True))
        s = torch.sum(e, dim=(1, 2), keepdim=True)
        return e / s

    def _lin_interpolation_2d(self, x, dim):

        conv = SeparableConv2d(
            self.in_channels, self.num_filters, (self.num_rows, self.num_cols), self.stride, self.padding, bias=False)
        x = conv(x)

        ws = conv.weights
        for w in ws:
            w.data = torch.zeros(w.shape)

        linspace = self._linspace_2d(self.num_rows, self.num_cols, dim=dim)

        for i in range(self.num_filters):
            ws[0].data[:, :, i, 0] = linspace[:, :]
            ws[1].data[0, 0, i, i] = 1.

        for p in self.conv1.parameters():
            p.requires_grad = False

        x = Lambda(lambda x: torch.squeeze(x, dim=1))(x)
        x = Lambda(lambda x: torch.squeeze(x, dim=1))(x)
        # dim 0 becasue the channel is first in pytorch.
        x = Lambda(lambda x: torch.unsqueeze(x, dim=0))(x)
        return x

    def _linspace_2d(self, nb_rols, nb_cols, dim=0):

        def _lin_sp_aux(size, nb_repeat, start, end):
            linsp = torch.linspace(start=start, end=end, num=size)
            x = torch.empty((nb_repeat, size), dtype=torch.float32)

            for d in range(nb_repeat):
                x[d] = linsp

            return x

        if dim == 1:
            return (_lin_sp_aux(nb_rols, nb_cols, 0.0, 1.0)).T
        return _lin_sp_aux(nb_cols, nb_rols, 0.0, 1.0)


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
