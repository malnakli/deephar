import torch
import torch.nn as nn
from .helper import (
    Conv2dBatchActivate,
    Conv2dBatch,
    ActivateConv2dBatch,
    SeparableResidualModule,
    ActivateSeparableConv2dBatch,
    SoftArgMax2d,
    BuildJointsProbability,
    BuildContextAggregation,
    BuildReceptionBlock,
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
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.concat_pose_confidence = concat_pose_confidence
        self.export_heatmaps = export_heatmaps

        self.layer10 = ActivateSeparableConv2dBatch(
            in_channels=576, out_channels=576, kernel_size=self.kernel_size, padding=2
        )

        self.sam_s_model = SoftArgMax2d(
            in_channels=Nj, out_channels=Nj, kernel_size=(32, 32)
        )
        self.jprob_s_model = BuildJointsProbability(kernel_size=(32, 32), padding=0)

        _num_heatmaps = (num_context_per_joint + 1) * Nj

        self.sam_c_model = SoftArgMax2d(
            in_channels=_num_heatmaps, out_channels=_num_heatmaps, kernel_size=(32, 32)
        )
        self.jprob_c_model = BuildJointsProbability(kernel_size=(32, 32), padding=0)

        self.agg_model = BuildContextAggregation(
            in_features=1, alpha=0.8, num_joints=Nj, num_context=num_context_per_joint
        )

        # heatmap
        self.layer11 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=576,
                out_channels=Nd * Nj,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        self.layer12 = ActivateConv2dBatch(
            in_channels=Nd * Nj, out_channels=576, kernel_size=1
        )

    def forward(self, x):
        outputs = []
        for block in range(self.num_blocks):
            x = BuildReceptionBlock(self.kernel_size)(x)
            ident_map = x

            x = self.layer10(x)
            h = self.layer11(x)

            pose, visible, hm = pose_regression_2d_context(
                h,
                self.num_joints,
                self.sam_s_model,
                self.sam_c_model,
                self.jprob_c_model,
                self.agg_model,
                self.jprob_s_model,
            )

            if self.concat_pose_confidence:
                outputs.append(torch.cat([pose, visible]))
            else:
                outputs.append(pose)
                outputs.append(visible)

            if self.export_heatmaps:
                outputs.append(hm)

            if block < self.num_blocks - 1:
                h = self.layer12(h)
                x = torch.add(torch.add(ident_map, x), h)

        return torch.tensor(outputs)


net = PredictionBlockPS()
x = torch.ones([1, 576, 32, 32])
print(net(x).shape)
# print(conv2d_out_shape(64, 64, nn.Conv2d(96, 96, (2, 2), 2, (0, 0))))
