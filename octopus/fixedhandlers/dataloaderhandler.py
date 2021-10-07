"""
All things related to data reading and writing.
"""
__author__ = 'ryanquinnnelson'

import logging

from torch.utils.data import DataLoader


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
        self.train_args = None
        self.val_args = None
        self.test_args = None

    def setup(self, device):
        # set arguments based on GPU or CPU destination
        if device.type == 'cuda':
            self.train_args = dict(shuffle=True,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           pin_memory=self.pin_memory)
        else:
            self.train_args = dict(shuffle=True,
                           batch_size=self.batch_size)

        # set arguments based on GPU or CPU destination
        if device.type == 'cuda':
            self.val_args = dict(shuffle=False,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           pin_memory=self.pin_memory)
        else:
            self.val_args = dict(shuffle=False,
                           batch_size=self.batch_size)

        # same as validation set
        self.test_args = self.val_args

        logging.info(f'DataLoader settings for training dataset:{self.train_args}')
        logging.info(f'DataLoader settings for validation dataset:{self.val_args}')
        logging.info(f'DataLoader settings for test dataset:{self.test_args}')

    def train_dataloader(self, dataset):
        dl = DataLoader(dataset, **self.train_args)
        return dl

    def val_dataloader(self, dataset):
        dl = DataLoader(dataset, **self.val_args)
        return dl

    def test_dataloader(self, dataset):
        dl = DataLoader(dataset, **self.test_args)
        return dl

    def load(self, inputhandler):

        logging.info('Loading data...')

        # Datasets
        train_dataset = inputhandler.get_train_dataset()
        val_dataset = inputhandler.get_val_dataset()
        test_dataset = inputhandler.get_test_dataset()

        # DataLoaders
        train_dl = self.train_dataloader(train_dataset)
        val_dl = self.val_dataloader(val_dataset)
        test_dl = self.test_dataloader(test_dataset)

        return train_dl, val_dl, test_dl
