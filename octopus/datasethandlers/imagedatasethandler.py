"""
All things related to data reading and writing.

Note 1:
ImageFolder uses alphabetical order when mapping directories to classes:

/train_data/0/<images>
/train_data/1/<images>
/train_data/2/<images>
:
/train_data/10/<images>

becomes

{'0': 0,
'1': 1,
'10': 2,
'2': 3}

where directory 10 maps to class 2 according to ImageFolder. This can cause issues if you write a custom Dataset
class for test data and index numerically. The simplest workaround is to remap the assigned labels from the test output
using the class_to_idx dictionary from ImageFolder.

mapping = imf.class_to_idx
test_label=2
list(mapping.keys())[list(mapping.values()).index(test_label)]  #'10'
"""
__author__ = 'ryanquinnnelson'

import logging
import json
import os

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from customized.datasets import TestDataset


def _compose_transforms(transforms_list):
    t_list = []

    for each in transforms_list:
        if each == 'RandomHorizontalFlip':
            t_list.append(transforms.RandomHorizontalFlip(0.1))
        elif each == 'ToTensor':
            t_list.append(transforms.ToTensor())
        elif each == 'RandomRotation':
            t_list.append(transforms.RandomRotation(degrees=15))
        elif each == 'Normalize':
            t_list.append(
                transforms.Normalize(mean=(0.229, 0.224, 0.225),
                                     std=(0.485, 0.456, 0.406)))  # tuple size == channels, imagenet values
        elif each == 'ColorJitter':
            t_list.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))

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
        self.should_normalize_val = True if 'Normalize' in transforms_list else False
        self.should_normalize_test = True if 'Normalize' in transforms_list else False

    def get_train_dataset(self):
        transform = _compose_transforms(self.transforms_list)
        imf = ImageFolder(self.train_dir, transform=transform)
        logging.info(f'Loaded {len(imf.imgs)} images as training data.')

        # dump json file with class to index mapping for use during testing (see Note 1 above)
        mapping = imf.class_to_idx

        mapping_dest = os.path.join(self.data_dir, 'class_to_idx.json')
        logging.info(f'Writing class_to_idx mapping to {mapping_dest}...')
        with open(mapping_dest, 'w') as file:
            json.dump(mapping, file)

        return imf

    def get_val_dataset(self):

        if self.should_normalize_val:
            logging.info('Normalizing validation data to match normalization of training data...')
            t = _compose_transforms(['ToTensor', 'Normalize'])
        else:
            t = _compose_transforms(['ToTensor'])

        imf = ImageFolder(self.val_dir, transform=t)
        logging.info(f'Loaded {len(imf.imgs)} images as validation data.')
        return imf

    def get_test_dataset(self):

        if self.should_normalize_test:
            logging.info('Normalizing test data to match normalization of training data...')
            t = _compose_transforms(['ToTensor', 'Normalize'])
        else:
            t = _compose_transforms(['ToTensor'])

        ds = TestDataset(self.test_dir, transform=t)
        logging.info(f'Loaded {ds.length} images as test data.')
        return ds
