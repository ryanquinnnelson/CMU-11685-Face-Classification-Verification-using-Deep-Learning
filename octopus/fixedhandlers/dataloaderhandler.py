"""
All things related to data reading and writing.
"""
__author__ = 'ryanquinnnelson'

import os
import logging
from datetime import datetime

import pandas as pd
from torch.utils.data import DataLoader

from octopus.utilities import utilities


class DataLoaderHandler:

    def __init__(self,
                 batch_size,
                 num_workers,
                 pin_memory):
        logging.info('Initializing dataloader handling...')
        # parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self, dataset, device):
        # set arguments based on GPU or CPU destination
        if device.type == 'cuda':
            dl_args = dict(shuffle=True,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           pin_memory=self.pin_memory)
        else:
            dl_args = dict(shuffle=True,
                           batch_size=self.batch_size)

        logging.info(f'DataLoader settings for training dataset:{dl_args}')
        dl = DataLoader(dataset, **dl_args)
        return dl

    def val_dataloader(self, dataset, device):
        # set arguments based on GPU or CPU destination
        if device.type == 'cuda':
            dl_args = dict(shuffle=False,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           pin_memory=self.pin_memory)
        else:
            dl_args = dict(shuffle=False,
                           batch_size=self.batch_size)

        logging.info(f'DataLoader settings for validation dataset:{dl_args}')
        dl = DataLoader(dataset, **dl_args)
        return dl

    def test_dataloader(self, dataset, device):
        # set arguments based on GPU or CPU destination
        if device.type == 'cuda':
            dl_args = dict(shuffle=False,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           pin_memory=self.pin_memory)
        else:
            dl_args = dict(shuffle=False,
                           batch_size=self.batch_size)

        logging.info(f'DataLoader settings for test dataset:{dl_args}')
        dl = DataLoader(dataset, **dl_args)
        return dl

    def load(self, train_dataset, val_dataset, test_dataset, devicehandler):

        logging.info('Loading data...')

        # DataLoaders
        device = devicehandler.get_device()
        train_dl = self.train_dataloader(train_dataset, device)
        val_dl = self.val_dataloader(val_dataset, device)
        test_dl = self.test_dataloader(test_dataset, device)

        return train_dl, val_dl, test_dl
