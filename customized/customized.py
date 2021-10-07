from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

__author__ = 'ryanquinnnelson'


class TrainValDataset:
    def __init__(self, image_dir, transforms_list):

        compose_list = []

        for each in transforms_list:
            if each == 'RandomHorizontalFlip':
                compose_list.append(transforms.RandomHorizontalFlip())
            elif each == 'ToTensor':
                compose_list.append(transforms.ToTensor())

        t = transforms.Compose(compose_list)

        self.imagefolder = ImageFolder(image_dir, transform=t)


class TestDataset:
    def __init__(self, image_dir, transforms_list):

        compose_list = []

        for each in transforms_list:
            if each == 'RandomHorizontalFlip':
                compose_list.append(transforms.RandomHorizontalFlip())
            elif each == 'ToTensor':
                compose_list.append(transforms.ToTensor())

        t = transforms.Compose(compose_list)

        self.imagefolder = ImageFolder(image_dir, transform=t)


class Evaluation:
    def __init__(self, val_loader, loss_func, devicehandler):
        pass


class OutputFormatter:
    pass
