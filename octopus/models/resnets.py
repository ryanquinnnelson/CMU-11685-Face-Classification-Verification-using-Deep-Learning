"""
All things Resnet.
"""

__author__ = 'ryanquinnnelson'

import logging

import torch.nn as nn


def _init_weights(mod):
    if isinstance(mod, nn.Conv2d):
        nn.init.kaiming_normal_(mod.weight)


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

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(

            # first conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
            # initialize to kaiming normal
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))

        # shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # blocks
        out = self.blocks(x)

        if self.in_channels != self.out_channels:
            # shortcut
            shortcut = self.shortcut(x)

            # combine
            activate = nn.ReLU(inplace=True)
            out = activate(out + shortcut)

        return out


# uses identity rather than skip shortcut
# this improves learning rate quite a bit for some reason
class ResidualBlock2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(

            # first conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
            # initialize with Kaiming normal
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))

        # shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        # blocks
        out = self.blocks(x)

        # shortcut
        shortcut = self.shortcut(x)

        # combine
        out = self.activate(out + shortcut)

        return out


# includes kaiming initialization
class ResidualBlock3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = 2 if in_channels != out_channels else 1

        # first conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)

        # second conv layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv2.weight)

        self.blocks = nn.Sequential(

            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            self.conv2,
            nn.BatchNorm2d(out_channels))

        # shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        # blocks
        out = self.blocks(x)

        # shortcut
        shortcut = self.shortcut(x)

        # combine
        out = self.activate(out + shortcut)

        return out


# includes kaiming initialization in a way that doesn't introduce twice the number of layers
class ResidualBlock4(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = 2 if in_channels != out_channels else 1

        nn.init.kaiming_normal_(self.conv1.weight)

        nn.init.kaiming_normal_(self.conv2.weight)

        self.blocks = nn.Sequential(
            # first conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))

        # shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.activate = nn.ReLU(inplace=True)

        # initialize weights
        self.blocks.apply(_init_weights)
        self.shortcut.apply(_init_weights)

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
    def __init__(self, in_features, num_classes, feat_dim=2):
        super().__init__()
        self.feat_dim = feat_dim

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

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

        self.linear_feat_dim = nn.Linear(512, self.feat_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)
        embedding_out = self.activation(self.linear_feat_dim(embedding))
        output = self.linear(embedding)

        if return_embedding:
            return embedding_out, output
        else:
            return output


class Resnet34(nn.Module):
    def __init__(self, in_features, num_classes, feat_dim=2):
        super().__init__()
        self.feat_dim = feat_dim

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

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

        self.linear_feat_dim = nn.Linear(512, self.feat_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)
        embedding_out = self.activation(self.linear_feat_dim(embedding))
        output = self.linear(embedding)

        if return_embedding:
            return embedding_out, output
        else:
            return output


class Resnet34_v2(nn.Module):
    def __init__(self, in_features, num_classes, feat_dim=2):
        super().__init__()
        self.feat_dim = feat_dim

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            ResidualBlock2(64, 64),
            ResidualBlock2(64, 64),
            ResidualBlock2(64, 64),

            # conv3..x
            ResidualBlock2(64, 128),
            ResidualBlock2(128, 128),
            ResidualBlock2(128, 128),
            ResidualBlock2(128, 128),

            # conv4..x
            ResidualBlock2(128, 256),
            ResidualBlock2(256, 256),
            ResidualBlock2(256, 256),
            ResidualBlock2(256, 256),
            ResidualBlock2(256, 256),
            ResidualBlock2(256, 256),

            # conv5..x
            ResidualBlock2(256, 512),
            ResidualBlock2(512, 512),
            ResidualBlock2(512, 512),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(512, num_classes))

        self.linear_feat_dim = nn.Linear(512, self.feat_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)
        embedding_out = self.activation(self.linear_feat_dim(embedding))
        output = self.linear(embedding)

        if return_embedding:
            return embedding_out, output
        else:
            return output


# change kernel size of first layer to 3 to work with smaller images in our dataset
# kaiming initialization for initial conv layer
# remove max pool layer
class Resnet34_v3(nn.Module):
    def __init__(self, in_features, num_classes, feat_dim=2):
        super().__init__()
        self.feat_dim = feat_dim

        # conv1
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=3, stride=1, padding=3, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)

        self.layers = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            ResidualBlock3(64, 64),
            ResidualBlock3(64, 64),
            ResidualBlock3(64, 64),

            # conv3..x
            ResidualBlock3(64, 128),
            ResidualBlock3(128, 128),
            ResidualBlock3(128, 128),
            ResidualBlock3(128, 128),

            # conv4..x
            ResidualBlock3(128, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),

            # conv5..x
            ResidualBlock3(256, 512),
            ResidualBlock3(512, 512),
            ResidualBlock3(512, 512),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(512, num_classes))

        self.linear_feat_dim = nn.Linear(512, self.feat_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)
        embedding_out = self.activation(self.linear_feat_dim(embedding))
        output = self.linear(embedding)

        if return_embedding:
            return embedding_out, output
        else:
            return output


# change kernel size of first layer to 3 to work with smaller images in our dataset
# kaiming initialization for initial conv layer
# remove max pool layer
# initializes weights in a way that doesn't double the number of layers
class Resnet34_v4(nn.Module):
    def __init__(self, in_features, num_classes, feat_dim=512):
        super().__init__()
        self.feat_dim = feat_dim

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=3, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            ResidualBlock3(64, 64),
            ResidualBlock3(64, 64),
            ResidualBlock3(64, 64),

            # conv3..x
            ResidualBlock3(64, 128),
            ResidualBlock3(128, 128),
            ResidualBlock3(128, 128),
            ResidualBlock3(128, 128),

            # conv4..x
            ResidualBlock3(128, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),

            # conv5..x
            ResidualBlock3(256, 512),
            ResidualBlock3(512, 512),
            ResidualBlock3(512, 512),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(512, num_classes))

        self.linear_feat_dim = nn.Linear(512, self.feat_dim)
        self.activation = nn.ReLU(inplace=True)

        # initialize weights
        self.layers.apply(_init_weights)

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)
        embedding_out = self.activation(self.linear_feat_dim(embedding))
        output = self.linear(embedding)

        if return_embedding:
            return embedding_out, output
        else:
            return output


class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_shortcut, mid_stride, shortcut_stride):
        super().__init__()

        self.use_shortcut = use_shortcut

        self.stride = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(

            # first conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            #             nn.ReLU(inplace=True),

            # second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=mid_stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            #             nn.ReLU(inplace=True),

            # third conv layer
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(inplace=True)
        )

        #         # shortcut
        #         if in_channels == out_channels:
        #             self.shortcut = nn.Identity()
        #         else:

        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=shortcut_stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):

        # blocks
        out = self.blocks(x)

        if self.use_shortcut:
            # shortcut
            shortcut = self.shortcut(x)

            # combine
            activate = nn.ReLU(inplace=True)
            out = activate(out + shortcut)

        return out


class Resnet50(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            BottleneckResidualBlock(64, 64, use_shortcut=True, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(256, 64, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(256, 64, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv3..x
            BottleneckResidualBlock(256, 128, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv4..x
            BottleneckResidualBlock(512, 256, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv5..x
            BottleneckResidualBlock(1024, 512, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(2048, 512, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(2048, 512, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(2048, num_classes))

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)

        if return_embedding:
            return embedding
        else:
            return self.linear(embedding)


class Resnet101(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            BottleneckResidualBlock(64, 64, use_shortcut=True, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(256, 64, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(256, 64, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv3..x
            BottleneckResidualBlock(256, 128, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv4..x
            BottleneckResidualBlock(512, 256, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv5..x
            BottleneckResidualBlock(1024, 512, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(2048, 512, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(2048, 512, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(2048, num_classes))

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)

        if return_embedding:
            return embedding
        else:
            return self.linear(embedding)


class Resnet152(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            BottleneckResidualBlock(64, 64, use_shortcut=True, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(256, 64, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(256, 64, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv3..x
            BottleneckResidualBlock(256, 128, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv4..x
            BottleneckResidualBlock(512, 256, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv5..x
            BottleneckResidualBlock(1024, 512, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(2048, 512, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(2048, 512, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(2048, num_classes))

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)

        if return_embedding:
            return embedding
        else:
            return self.linear(embedding)
