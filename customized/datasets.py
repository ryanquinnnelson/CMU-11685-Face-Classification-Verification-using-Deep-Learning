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

        # process images
        tensors = []
        transform = transforms.ToTensor()
        for f in filenames:
            # open image
            img = Image.open(f).convert('RGB')

            # convert into a Tensor
            tensors.append(transform(img))

        # stack all tensors into a single tensor
        self.data = torch.stack(tensors)
        self.length = len(filenames)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index]
