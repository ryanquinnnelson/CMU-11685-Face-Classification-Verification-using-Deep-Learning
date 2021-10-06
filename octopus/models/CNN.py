"""
Defines all standard CNNs octopus can generate.
"""
__author__ = 'ryanquinnnelson'

import torch.nn as nn
from collections import OrderedDict


# refactor to avoid duplicating code
def _get_activation_function(activation_func):
    act = None

    if activation_func == 'ReLU':
        act = nn.ReLU(inplace=True)
    elif activation_func == 'LeakyReLU':
        act = nn.LeakyReLU(inplace=True)
    elif activation_func == 'Sigmoid':
        act = nn.Sigmoid()
    elif activation_func == 'Tanh':
        act = nn.Tanh()

    return act


def _get_pool_class(pool_class):
    pool = None

    if pool_class == 'MaxPool2d':
        pool = nn.MaxPool2d

    return pool


def _calc_output_size(input_size, padding, dilation, kernel_size, stride):
    input_size_padded = input_size + 2 * padding
    kernel_dilated = (kernel_size - 1) * (dilation - 1) + kernel_size
    output_size = (input_size_padded - kernel_dilated) // stride + 1
    return output_size


def _calc_output_size_from_dict(input_size, conv_dict):
    padding = conv_dict['padding']
    dilation = conv_dict['dilation']
    kernel_size = conv_dict['kernel_size']
    stride = conv_dict['stride']

    output_size = _calc_output_size(input_size, padding, dilation, kernel_size, stride)
    return output_size


def _build_cnn_sequence(input_size, conv_dicts, batch_norm, activation_func, pool_dicts, pool_class):
    # define convolutional layers
    sequence = []
    layer_input_size = input_size
    layer_output_size = None
    print('start',layer_input_size, layer_output_size)
    for i, conv_dict in enumerate(conv_dicts):  # create a layer for each parameter dictionary

        # convolution
        layer_name = 'conv' + str(i + 1)
        conv_tuple = (layer_name, nn.Conv2d(**conv_dict))
        sequence.append(conv_tuple)
        layer_output_size = _calc_output_size_from_dict(layer_input_size, conv_dict)
        print('conv',layer_input_size, layer_output_size)
        # batch normalization
        if batch_norm:
            layer_name = 'bn' + str(i + 1)
            bn_tuple = (layer_name, nn.BatchNorm2d(num_features=conv_dict['out_channels']))
            sequence.append(bn_tuple)

        # activation layer
        layer_name = activation_func + str(i + 1)
        activation_tuple = (layer_name, _get_activation_function(activation_func))
        sequence.append(activation_tuple)

        # pooling layer
        if len(pool_dicts) > i:
            pool_dict = pool_dicts[i]
            layer_name = 'pool' + str(i + 1)
            layer_pool_class = _get_pool_class(pool_class)
            pool_tuple = (layer_name, layer_pool_class(**pool_dict))
            sequence.append(pool_tuple)

            # update input and output sizes based on pooling layer
            layer_input_size = layer_output_size
            layer_output_size = _calc_output_size_from_dict(layer_input_size, pool_dict)
            print('pool',layer_input_size, layer_output_size)

    return sequence, layer_output_size


def _build_linear_sequence(input_size, output_size):
    sequence = [
        ('flat', nn.Flatten()),
        ('lin', nn.Linear(input_size, output_size))
    ]
    return sequence


class CNN2d(nn.Module):
    def __init__(self, input_size, output_size, activation_func, pool_class, batch_norm, conv_dicts, pool_dicts):
        super(CNN2d, self).__init__()

        cnn_sequence, cnn_output_size = _build_cnn_sequence(input_size, conv_dicts, batch_norm, activation_func,
                                                            pool_dicts, pool_class)
        self.cnn_layers = nn.Sequential(OrderedDict(cnn_sequence))

        # define linear layers
        linear_sequence = _build_linear_sequence(cnn_output_size, output_size)
        self.linear_layers = nn.Sequential(OrderedDict(linear_sequence))

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.linear_layers
        return x
