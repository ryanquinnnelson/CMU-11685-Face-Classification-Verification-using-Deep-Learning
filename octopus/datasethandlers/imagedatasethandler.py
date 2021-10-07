"""
All things related to data reading and writing.
"""
__author__ = 'ryanquinnnelson'

import logging

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def _compose_transforms(transforms_list):
    t_list = []

    for each in transforms_list:
        if each == 'RandomHorizontalFlip':
            t_list.append(transforms.RandomHorizontalFlip())
        elif each == 'ToTensor':
            t_list.append(transforms.ToTensor())

    composition = transforms.Compose(t_list)

    return composition


class ImageDatasetHandler:

    def __init__(self,
                 data_dir,
                 train_dir,
                 val_dir,
                 test_dir,
                 transforms_list):
        logging.info('Initializing dataset handling...')
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.transforms_list = transforms_list

    def get_train_dataset(self):
        transform = _compose_transforms(self.transforms_list)
        imf = ImageFolder(self.train_dir, transform=transform)
        logging.info(f'Loaded {len(imf.imgs)} images as training data.')
        return imf

    def get_val_dataset(self):
        transform = _compose_transforms(self.transforms_list)
        imf = ImageFolder(self.val_dir, transform=transform)
        logging.info(f'Loaded {len(imf.imgs)} images as validation data.')
        return imf

    def get_test_dataset(self):
        imf = None
        # imf = ImageFolder(self.test_dir)
        # logging.info(f'Loaded {len(imf.imgs)} images as test data.')
        return imf