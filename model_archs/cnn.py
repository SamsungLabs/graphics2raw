"""
Implementation of
Convolutional Mean: A Simple Convolutional Neural Network for Illuminant Estimation

@INPROCEEDINGS{convmean,
  title={Convolutional Mean: {A} Simple Convolutional Neural Network for Illuminant Estimation},
  author={Han Gong},
  booktitle={BMVC},
  year={2019},
}
"""

import torch.nn as nn


class IllumEstNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, num_filters=7):
        super(IllumEstNet, self).__init__()

        # conv1
        self.conv1_1 = nn.Conv2d(in_channels, num_filters, (3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1_1 = nn.ReLU(inplace=True)

        # conv2
        self.conv2_1 = nn.Conv2d(num_filters, num_filters * 2, (3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2_1 = nn.ReLU(inplace=True)

        # conv3
        self.conv3_1 = nn.Conv2d(num_filters * 2, out_channels, (1, 1), padding=0)  # 1x1 conv
        self.relu3_1 = nn.ReLU(inplace=True)

        # per channel global average pooling
        self.aap = nn.AdaptiveAvgPool2d((1, 1))  # b, c, 1, 1

    def forward(self, x):
        # x: b, 3, p, p
        x = self.conv1_1(x)  # b, 7, p, p
        x = self.pool1(x)  # b, 7, p/2, p/2
        x = self.relu1_1(x)

        x = self.conv2_1(x)  # b, 14, p/2, p/2
        x = self.pool2(x)  # b, 14, p/4, p/4  16 16
        x = self.relu2_1(x)

        x = self.conv3_1(x)  # b, 3, p/4, p/4
        x = self.relu3_1(x)
        x = self.aap(x)  # b, 3, 1, 1
        x = x.squeeze(dim=-1).squeeze(dim=-1)  # b, 3
        return x
