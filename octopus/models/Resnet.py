"""
Simple Residual block
"""

__author__ = 'ryanquinnnelson'

import logging

import torch.nn as nn

def _calc_output_size(input_size, padding, dilation, kernel_size, stride):
    input_size_padded = input_size + 2 * padding
    kernel_dilated = (kernel_size - 1) * (dilation - 1) + kernel_size
    output_size = (input_size_padded - kernel_dilated) // stride + 1
    return output_size


class SimpleResidualBlock(nn.Module):
    def __init__(self, channel_size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_size)
        if stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(channel_size, channel_size, kernel_size=1, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        shortcut = self.shortcut(x)

        out = self.relu(out + shortcut)

        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()

        # first conv layer
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        # second conv layer
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        # shortcut
        if stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.shortcut(x)

        out = self.relu(out + shortcut)

        return out


class ClassificationNetwork(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_features, 64, kernel_size=7, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(64),  # ?? include or not
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            SimpleResidualBlock(64),
            SimpleResidualBlock(64),
            SimpleResidualBlock(64),
            SimpleResidualBlock(64),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)

        if return_embedding:
            return embedding
        else:
            return self.linear(embedding)


class Resnet18(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_features, out_channels=64, kernel_size=7, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ResidualBlock(64, 64, stride=2),
            ResidualBlock(64, 64, stride=1),

            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),

            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),

            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)

        if return_embedding:
            return embedding
        else:
            return self.linear(embedding)
