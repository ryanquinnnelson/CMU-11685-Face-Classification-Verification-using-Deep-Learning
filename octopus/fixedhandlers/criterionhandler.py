"""
All things related to criterion.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch.nn as nn
from octopus.criterion.centerloss import CenterLoss


class CriterionHandler:

    def __init__(self, criterion_type):
        logging.info('Initializing criterion handling...')
        self.criterion_type = criterion_type

    def get_loss_function(self, **kwargs):
        criterion = None
        if self.criterion_type == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()

        elif self.criterion_type == 'CenterLoss':
            criterion = CenterLoss(**kwargs)

        logging.info(f'Criterion is set:{criterion}.')
        return criterion
