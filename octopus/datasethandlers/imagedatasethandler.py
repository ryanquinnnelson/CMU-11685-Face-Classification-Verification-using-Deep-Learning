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
            t_list.append(transforms.RandomHorizontalFlip())
        elif each == 'ToTensor':
            t_list.append(transforms.ToTensor())
        elif each == 'RandomRotation':
            t_list.append(transforms.RandomRotation(degrees=30))
        elif each == 'RandomVerticalFlip':
            t_list.append(transforms.RandomVerticalFlip())
        elif each == 'Normalize':
            t_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))  # tuple size == channels
        elif each == 'RandomAffine':
            t_list.append(transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), shear=(-30, 30)))
        elif each == 'ColorJitter':
            t_list.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
        elif each == 'RandomResizedCrop':
            t_list.append(transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0)))

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

        # dump json file with class to index mapping for use during testing (see Note 1 above)
        mapping = imf.class_to_idx

        mapping_dest = os.path.join(self.data_dir, 'class_to_idx.json')
        logging.info(f'Writing class_to_idx mapping to {mapping_dest}...')
        with open(mapping_dest, 'w') as file:
            json.dump(mapping, file)

        return imf

    def get_val_dataset(self):
        imf = ImageFolder(self.val_dir, transform=transforms.Compose([transforms.ToTensor()]))
        logging.info(f'Loaded {len(imf.imgs)} images as validation data.')
        return imf

    def get_test_dataset(self):
        ds = TestDataset(self.test_dir)
        logging.info(f'Loaded {ds.length} images as test data.')
        return ds
