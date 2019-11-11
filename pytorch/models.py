import torch
import torch.nn as nn
import torch.nn.functional as F
from .helper import (
    conv2d_out_shape, Conv2dBatchActivate, Conv2dBatch, ActivateConv2dBatch, Concatenate)


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
        self.layer4 = Concatenate([self.layer3_a, self.layer3_b])

        self.layer5_b = Conv2dBatchActivate(
            in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0
        )

        self.layer6_a = Conv2dBatchActivate(
            in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0
        )
        self.layer6_b = Conv2dBatchActivate(
            in_channels=64, out_channels=64, kernel_size=(5, 1), stride=1, padding=(2, 0)
        )

        self.layer7_a = Conv2dBatch(
            in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1
        )
        self.layer7_b = Conv2dBatchActivate(
            in_channels=64, out_channels=64, kernel_size=(1, 5), stride=1, padding=(0, 2)
        )
        self.layer8_b = Conv2dBatch(
            in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1
        )

        self.layer9 = Concatenate([self.layer7_a, self.layer8_b])

        self.layer10_a = ActivateConv2dBatch(
            in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=1
        )

        self.layer10_b = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer11 = Concatenate([self.layer10_a, self.layer10_b])
        self.layer12 = 0

    def forward(self, x):
        x = self.branch0(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = MultitaskStemNet()
# print(net)
print(conv2d_out_shape(64, 64, nn.Conv2d(96, 96, (2, 2), 2, (0, 0))))
