import glob
import os

from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms


class TestDataset(Dataset):

    def __init__(self, test_dir):
        # get all filenames in numerical order
        filenames = glob.glob(test_dir + '/*.jpg')
        filenames.sort(key=lambda e: int(os.path.basename(e).split('.')[0]))
        self.imgs = filenames

        # # process images
        # tensors = []
        #
        # for f in filenames:
        #
        #
        # # stack all tensors into a single tensor
        # self.data = torch.stack(tensors)
        self.length = len(filenames)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        f = self.imgs[index]

        # open image
        img = Image.open(f).convert('RGB')

        # convert into a Tensor
        transform = transforms.ToTensor()
        return transform(img)