import glob
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class TestDataset(Dataset):

    def __init__(self, test_dir):
        # get all filenames in numerical order
        filenames = glob.glob(test_dir + '/*.jpg')
        filenames.sort(key=lambda e: int(os.path.basename(e).split('.')[0]))
        self.imgs = filenames
        self.length = len(filenames)

        # for each in self.imgs[:50]:
        #     print(each)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # if index % 256 == 0:
        #     print('getitem',index)
        f = self.imgs[index]

        # open image
        img = Image.open(f) #.convert('RGB') #?? should I not be converting to RGB?

        # convert into a Tensor
        transform = transforms.ToTensor()
        return transform(img)
