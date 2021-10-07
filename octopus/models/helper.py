from torch import nn as nn


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