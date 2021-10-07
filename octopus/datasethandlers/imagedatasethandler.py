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
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.transforms_list = transforms_list

    def get_train_dataset(self):
        transform = _compose_transforms(self.transforms_list)
        return ImageFolder(self.train_dir, transform=transform)

    def get_val_dataset(self):
        transform = _compose_transforms(self.transforms_list)
        return ImageFolder(self.val_dir, transform=transform)

    def get_test_dataset(self):
        return ImageFolder(self.test_dir)

