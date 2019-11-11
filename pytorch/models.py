import torch
import torch.nn as nn
import torch.nn.functional as F
from .helper import (
    conv2d_out_shape, Conv2dBatchActivate, Conv2dBatch, ActivateConv2dBatch, SeparableResidualModule)


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
            in_channels=64, out_channels=64, kernel_size=(5, 1), stride=1, padding=(2, 0)
        )

        self.layer6_a = Conv2dBatch(
            in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1
        )
        self.layer6_b = Conv2dBatchActivate(
            in_channels=64, out_channels=64, kernel_size=(1, 5), stride=1, padding=(0, 2)
        )
        self.layer7_b = Conv2dBatch(
            in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1
        )

        self.layer8_a = ActivateConv2dBatch(
            in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=1
        )

        self.layer8_b = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer9 = SeparableResidualModule(
            in_channels=384, out_channels=576)

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


net = MultitaskStemNet()
x = torch.ones([1, 3, 256, 256])
print(net(x).shape)
#print(conv2d_out_shape(64, 64, nn.Conv2d(96, 96, (2, 2), 2, (0, 0))))
