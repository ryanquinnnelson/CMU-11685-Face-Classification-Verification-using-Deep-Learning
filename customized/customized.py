from torch.utils.data import Dataset

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
__author__ = 'ryanquinnnelson'
# class Train_Test_Dataset(Dataset):
#
#     def __init__(self, data, labels):
#         pass
#
#     def __len__(self):
#         pass
#
#     def __getitem(self, index):
#         pass

p = '/Users/ryanqnelson/Desktop/test/content/competitions/idl-fall21-hw2p2s1-face-classification/idl-fall21-hw2p2s1-face-classification/val_data'
imf = ImageFolder(p)

dl = DataLoader(imf, batch_size=10, shuffle=False)


class TrainValDataset:
    pass


class TestDataset:
    pass


class Evaluation:
    pass


class OutputFormatter:
    pass
