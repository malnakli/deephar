import math
import pdb

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
        self.relu = nn.ReLU()

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
        self.relu = nn.ReLU()

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

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
        )

        self.weights = [self.conv1.weight, self.pointwise.weight]

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class ActivateSeparableConv2dBatch(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
    ):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = SeparableConv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
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
            in_channels, out_channels, kernel_size=1, stride=1
        )

        self.conv2 = ActivateSeparableConv2dBatch(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )

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

        self.num_rows = kernel_size if isinstance(kernel_size, int) else kernel_size[0]

        self.num_cols = kernel_size if isinstance(kernel_size, int) else kernel_size[1]

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
            raise ValueError(
                "This function is specific for 4D tensors. " "Here, ndim=" + str(x.ndim)
            )

        e = torch.exp(x - _torch_max_dims(x, dims=[2, 3], keepdim=True))
        s = torch.sum(e, dim=(1, 2), keepdim=True)
        return e / s

    def _lin_interpolation_2d(self, x, dim):

        conv = SeparableConv2d(
            in_channels=self.in_channels,
            out_channels=self.num_filters,
            kernel_size=(self.num_rows, self.num_cols),
            stride=1,
            padding=0,
            bias=False,
        )
        x = conv(x)

        ws = conv.weights
        for w in ws:
            w.data = torch.zeros(w.shape)

        linspace = self._linspace_2d(self.num_rows, self.num_cols, dim=dim)

        for i in range(self.num_filters):
            ws[0].data[i, 0, :, :] = linspace[:, :]
            ws[1].data[i, i, 0, 0] = 1.0

        for _p in conv.parameters():
            _p.requires_grad = False

        x = Lambda(lambda x: torch.squeeze(x, dim=-1))(x)
        x = Lambda(lambda x: torch.squeeze(x, dim=-1))(x)
        x = Lambda(lambda x: torch.unsqueeze(x, dim=-1))(x)
        return x

    def _linspace_2d(self, nb_rols, nb_cols, dim=0):
        def _lin_sp_aux(size, nb_repeat, start, end):
            linsp = torch.linspace(start=start, end=end, steps=size)
            x = torch.empty((nb_repeat, size), dtype=torch.float32)

            for d in range(nb_repeat):
                x[d] = linsp

            return x

        if dim == 1:
            return (_lin_sp_aux(nb_rols, nb_cols, 0.0, 1.0)).T
        return _lin_sp_aux(nb_cols, nb_rols, 0.0, 1.0)


class BuildJointsProbability(nn.Module):
    def __init__(self, kernel_size=1, stride=1, padding=0):
        super().__init__()

        self.max = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.max(x)
        x = self.act(x)

        x = Lambda(lambda x: torch.squeeze(x, dim=-1))(x)
        x = Lambda(lambda x: torch.squeeze(x, dim=-1))(x)
        x = Lambda(lambda x: torch.unsqueeze(x, axis=-1))(x)
        return x


class BuildContextAggregation(nn.Module):
    def __init__(self, in_features, alpha, num_joints=16, num_context=1, num_frames=1):
        super().__init__()

        self.num_joints = num_joints
        self.num_frames = num_frames
        self.in_features = in_features
        self.alpha = alpha
        self.num_context = num_context

    def forward(self, ys, yc, pc):

        if self.num_frames == 1:
            # Define inputs
            # ys = torch.Tensor(2, self.num_joints)
            # yc = torch.Tensor(2, self.num_joints * self.num_context)
            # pc = torch.Tensor(1, self.num_joints * self.num_context)

            # Split contextual predictions in x and y and do computations separately
            xi = Lambda(lambda x: x[:, :, 0:1])(yc)
            yi = Lambda(lambda x: x[:, :, 1:2])(yc)
        else:
            # ys = Input(shape=(num_frames, num_joints, 2))
            # yc = Input(shape=(num_frames, num_joints * num_context, 2))
            # pc = Input(shape=(num_frames, num_joints * num_context, 1))

            # Split contextual predictions in x and y and do computations separately
            xi = Lambda(lambda x: x[:, :, :, 0:1])(yc)
            yi = Lambda(lambda x: x[:, :, :, 1:2])(yc)

        # Define auxiliary layers.
        mul_alpha = Lambda(lambda x: self.alpha * x)
        mul_1alpha = Lambda(lambda x: (1 - self.alpha) * x)

        tf_div = Lambda(lambda x: torch.div(x[0], x[1]))

        pxi = torch.mul(xi, pc)
        pyi = torch.mul(yi, pc)

        pc_sum = self._ctx_sum(pc, self.num_frames)
        pxi_sum = self._ctx_sum(pxi, self.num_frames)
        pyi_sum = self._ctx_sum(pyi, self.num_frames)

        pxi_div = tf_div([pxi_sum, pc_sum])
        pyi_div = tf_div([pyi_sum, pc_sum])
        yc_div = torch.cat([pxi_div, pyi_div], dim=-1)

        ys_alpha = mul_alpha(ys)
        yc_div_1alpha = mul_1alpha(yc_div)

        y = torch.add(ys_alpha, yc_div_1alpha)

        return y

    def _ctx_sum(self, inp, num_frames=1):
        class _Ctx(nn.Module):
            def __init__(self, in_features, num_joints=16, num_context=1):
                super().__init__()

                self.num_joints = num_joints
                self.num_context = num_context

                self.fc = nn.Linear(in_features, self.num_joints, bias=False)
                self.act = nn.Sigmoid()

            def forward(self, x):

                x = Lambda(lambda x: torch.squeeze(x, dim=-1))(x)
                x = self.fc(x)
                x = Lambda(lambda x: torch.unsqueeze(x, dim=-1))(x)

                _w = self.fc.weight
                _w.data = torch.zeros(_w.shape)  # w[0].fill(0)

                for j in range(self.num_joints):
                    _w.data[j, j * self.num_context : (j + 1) * self.num_context] = 1.0

                # d.trainable = False
                for _p in self.fc.parameters():
                    _p.requires_grad = False

                return x

        if num_frames > 1:
            raise NotImplementedError

        return _Ctx(
            in_features=self.in_features,
            num_joints=self.num_joints,
            num_context=self.num_context,
        )(inp)


class BuildReceptionBlock(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        self.layer0_b = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1_b = ActivateConv2dBatch(
            in_channels=576, out_channels=288, kernel_size=1
        )
        self.layer2_b = SeparableResidualModule(
            in_channels=288, out_channels=288, kernel_size=kernel_size, padding=2
        )

        self.layer3_c = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer4_c = SeparableResidualModule(
            in_channels=288, out_channels=288, kernel_size=kernel_size, padding=2
        )

        self.layer5_a = SeparableResidualModule(
            in_channels=576, out_channels=576, kernel_size=kernel_size, padding=2
        )
        self.layer5_b = SeparableResidualModule(
            in_channels=288, out_channels=288, kernel_size=kernel_size, padding=2
        )
        self.layer5_c = SeparableResidualModule(
            in_channels=288, out_channels=288, kernel_size=kernel_size, padding=2
        )

        self.layer6_c = SeparableResidualModule(
            in_channels=288, out_channels=288, kernel_size=kernel_size, padding=2
        )
        self.layer7_c = nn.Upsample(scale_factor=2)
        self.layer8_b = SeparableResidualModule(
            in_channels=288, out_channels=576, kernel_size=kernel_size, padding=2
        )
        self.layer9_b = nn.Upsample(scale_factor=2)

    def forward(self, x):
        _b = self.layer0_b(x)
        _b = self.layer1_b(_b)
        _b = self.layer2_b(_b)
        _c = self.layer3_c(_b)
        _c = self.layer4_c(_c)
        _a = self.layer5_a(x)
        _b = self.layer5_b(_b)
        _c = self.layer5_c(_c)
        _c = self.layer6_c(_c)
        _c = self.layer7_c(_c)
        _b = torch.add(_b, _c)
        _b = self.layer8_b(_b)
        _b = self.layer9_b(_b)
        x = torch.add(_a, _b)
        return x


def pose_regression_2d(h, sam_s_model, jprob_s_model):

    pose = sam_s_model(h)
    visible = jprob_s_model(h)

    return pose, visible, h


def pose_regression_2d_context(
    h, num_joints, sam_s_model, sam_c_model, jprob_c_model, agg_model, jprob_s_model
):
    # Split heatmaps for specialized and contextual information
    hs = Lambda(lambda x: x[:, :num_joints, :, :])(h)
    hc = Lambda(lambda x: x[:, num_joints:, :, :])(h)

    # Soft-argmax and joint probability for each heatmap
    ps = sam_s_model(hs)
    pc = sam_c_model(hc)
    vc = jprob_c_model(hc)

    pose = agg_model(ps, pc, vc)
    visible = jprob_s_model(hs)

    return pose, visible, hs


def conv2d_out_shape(width, height, Conv2d):
    """
    return (C , W , H)
    C: channels
    W: Width
    H: Height
    """
    # taken from:
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d

    _h = math.floor(
        (
            (
                height
                + (2 * Conv2d.padding[0])
                - Conv2d.dilation[0] * (Conv2d.kernel_size[0] - 1)
                - 1
            )
            / Conv2d.stride[0]
        )
        + 1
    )

    _w = math.floor(
        (
            (
                width
                + (2 * Conv2d.padding[1])
                - Conv2d.dilation[1] * (Conv2d.kernel_size[1] - 1)
                - 1
            )
            / Conv2d.stride[1]
        )
        + 1
    )

    return (Conv2d.out_channels, _w, _h)


def _torch_max_dims(x, dims=[0, 1], keepdim=False):
    for dim in dims:
        x = torch.max(x, dim=dim, keepdim=keepdim)[0]
    return x
