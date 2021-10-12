"""
All things Resnet.
"""

__author__ = 'ryanquinnnelson'

import torch.nn as nn


# Identical to class presented in Recitation 6
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


# Inspiration from https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.stride = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(

            # first conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))

        # shortcut
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        # blocks
        out = self.blocks(x)

        # shortcut
        shortcut = self.shortcut(x)

        # combine
        out = self.activate(out + shortcut)

        return out


# Inspiration from https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
class Resnet18(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),

            # conv3..x
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),

            # conv4..x
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),

            # conv5..x
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(512, num_classes))
        # nn.Softmax(dim=1))  # removed because it stopped model from improving

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)

        if return_embedding:
            return embedding
        else:
            return self.linear(embedding)


class Resnet34(nn.Module):
    def __init__(self, in_features, num_classes, init_type=None):
        super().__init__()
        self.init_type = init_type

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),

            # conv3..x
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),

            # conv4..x
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),

            # conv5..x
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(512, num_classes))
        # nn.Softmax(dim=1))  # removed because it stopped model from improving and I don't know why. Could be dim=1.

    #     # apply initialization if defined
    #     self.layers.apply(self._initialize_weights)
    #
    # def _initialize_weights(self, seq):
    #     logging.info(f'Initializing weights using {self.init_type}...')

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)

        if return_embedding:
            return embedding
        else:
            return self.linear(embedding)


class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.stride = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(

            # first conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # third conv layer
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
        )

        # shortcut
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
        self.activate = nn.ReLU(inplace=True)


class Resnet50(nn.Module):
    def __init__(self, in_features, num_classes, init_type=None):
        super().__init__()
        self.init_type = init_type

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            BottleneckResidualBlock(64, 64),
            BottleneckResidualBlock(64, 64),
            BottleneckResidualBlock(64, 64),

            # conv3..x
            BottleneckResidualBlock(64, 128),
            BottleneckResidualBlock(128, 128),
            BottleneckResidualBlock(128, 128),
            BottleneckResidualBlock(128, 128),

            # conv4..x
            BottleneckResidualBlock(128, 256),
            BottleneckResidualBlock(256, 256),
            BottleneckResidualBlock(256, 256),
            BottleneckResidualBlock(256, 256),
            BottleneckResidualBlock(256, 256),
            BottleneckResidualBlock(256, 256),

            # conv5..x
            BottleneckResidualBlock(256, 512),
            BottleneckResidualBlock(512, 512),
            BottleneckResidualBlock(512, 512),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(512, num_classes))

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)

        if return_embedding:
            return embedding
        else:
            return self.linear(embedding)
