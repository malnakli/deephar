import pdb

import torch
import torch.nn as nn

from .helper import (
    ActivateConv2dBatch,
    ActivateSeparableConv2dBatch,
    BuildContextAggregation,
    BuildJointsProbability,
    BuildReceptionBlock,
    Conv2dBatch,
    Conv2dBatchActivate,
    SeparableResidualModule,
    SoftArgMax2d,
    pose_regression_2d,
    pose_regression_2d_context,
)


class MultitaskStemNet(nn.Module):
    """
    Shared network (entry flow) based on Inception-V4.
    """

    def __init__(self):
        super().__init__()
        self.layer0 = Conv2dBatchActivate(
            in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1
        )
        self.layer1 = Conv2dBatchActivate(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.layer2 = Conv2dBatchActivate(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.layer3_a = Conv2dBatchActivate(
            in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1
        )
        self.layer3_b = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer4_b = Conv2dBatchActivate(
            in_channels=160, out_channels=64, kernel_size=1, stride=1, padding=0
        )

        self.layer5_a = Conv2dBatchActivate(
            in_channels=160, out_channels=64, kernel_size=1, stride=1, padding=0
        )
        self.layer5_b = Conv2dBatchActivate(
            in_channels=64,
            out_channels=64,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0),
        )

        self.layer6_a = Conv2dBatch(
            in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1
        )
        self.layer6_b = Conv2dBatchActivate(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 5),
            stride=1,
            padding=(0, 2),
        )
        self.layer7_b = Conv2dBatch(
            in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1
        )

        self.layer8_a = ActivateConv2dBatch(
            in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=1
        )

        self.layer8_b = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer9 = SeparableResidualModule(
            in_channels=384, out_channels=576, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.cat([self.layer3_a(x), self.layer3_b(x)], dim=1)
        xa = self.layer5_a(x)
        xa = self.layer6_a(xa)

        xb = self.layer4_b(x)
        xb = self.layer5_b(xb)
        xb = self.layer6_b(xb)
        xb = self.layer7_b(xb)
        x = torch.cat([xa, xb], dim=1)
        x = torch.cat([self.layer8_a(x), self.layer8_b(x)], dim=1)
        x = self.layer9(x)
        return x

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features


class PredictionBlockPS(nn.Module):
    """
     Prediction block is implemented as multi-resolution CNN for Pose Estimation
    """

    def __init__(
        self,
        num_context_per_joint=2,
        Nj=16,
        Nd=16,
        num_blocks=8,
        kernel_size=(5, 5),
        export_heatmaps=False,
        concat_pose_confidence=True,
        dim=2,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.concat_pose_confidence = concat_pose_confidence
        self.export_heatmaps = export_heatmaps
        self.num_joints = Nj
        self.dim = dim
        self.num_context_per_joint = num_context_per_joint

        self.recep_block  = BuildReceptionBlock(self.kernel_size)
        self.layer10 = ActivateSeparableConv2dBatch(
            in_channels=576, out_channels=576, kernel_size=self.kernel_size, padding=2
        )

        self.sam_s_model = SoftArgMax2d(
            in_channels=self.num_joints,
            out_channels=self.num_joints,
            kernel_size=(32, 32),
        )
        self.jprob_s_model = BuildJointsProbability(kernel_size=(32, 32), padding=0)

        num_context_per_joint = (
            2 if num_context_per_joint is None else num_context_per_joint
        )
        _num_heatmaps = (num_context_per_joint + 1) * self.num_joints

        if self.dim == 3:
            _num_heatmaps = Nd * self.num_joints

        self.sam_c_model = SoftArgMax2d(
            in_channels=_num_heatmaps - self.num_joints,
            out_channels=_num_heatmaps - self.num_joints,
            kernel_size=(32, 32),
        )
        self.jprob_c_model = BuildJointsProbability(kernel_size=(32, 32), padding=0)

        self.agg_model = BuildContextAggregation(
            in_features=self.num_joints * num_context_per_joint,
            alpha=0.8,
            num_joints=self.num_joints,
            num_context=num_context_per_joint,
        )

        # heatmap
        self.layer11 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=576,
                out_channels=_num_heatmaps,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        self.layer12 = ActivateConv2dBatch(
            in_channels=_num_heatmaps, out_channels=576, kernel_size=1
        )

    def forward(self, x):

        outputs = []
        for block in range(self.num_blocks):
            x = self.recep_block(x)
            ident_map = x

            x = self.layer10(x)
            h = self.layer11(x)

            if self.dim == 2:
                if self.num_context_per_joint is not None:
                    pose, visible, hm = pose_regression_2d_context(
                        h,
                        self.num_joints,
                        self.sam_s_model,
                        self.sam_c_model,
                        self.jprob_c_model,
                        self.agg_model,
                        self.jprob_s_model,
                    )
                else:
                    pose, visible, hm = pose_regression_2d(
                        h, self.sam_s_model, self.jprob_s_model
                    )

            if self.concat_pose_confidence:
                outputs.append(torch.cat([pose, visible], dim=-1))
            else:
                outputs.append(pose)
                outputs.append(visible)

            if self.export_heatmaps:
                outputs.append(hm)

            if block < self.num_blocks - 1:
                h = self.layer12(h)
                x = torch.add(torch.add(ident_map, x), h)

        return outputs


# stem = MultitaskStemNet()
# net = PredictionBlockPS()
# x = torch.ones([1, 3, 256, 256])
# # print(net(x).shape)

# pdb.set_trace()
# # print(conv2d_out_shape(64, 64, nn.Conv2d(96, 96, (2, 2), 2, (0, 0))))
