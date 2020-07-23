import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
from skimage import io, transform
import numpy
import torchvision



class PatchDataset(Dataset):
    def __init__(self, x_set, y_set, color_jitter=False):

        """
        TODO Add transformations
        Args:
            x_set (string): List of paths to images
            y_set (int): Labels of each image in x_set
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x_set = x_set
        self.y_set = y_set
        # self.input_size = input_size
        # self.is_eval = is_eval

        self.length = len(x_set)
        # self.transform = transform
        self.t = transforms.Compose([transforms.ToTensor()])
        if len(x_set) != len(y_set):
            raise ValueError('x set length does not match y set length')

        self.color_jitter = color_jitter
        self.COLOR_JITTER = torchvision.transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)

        # self.data_transforms = {
        #     'train': transforms.Compose([
        #         transforms.RandomResizedCrop(input_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ]),
        #     'val': transforms.Compose([
        #         transforms.Resize(input_size),
        #         transforms.CenterCrop(input_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ]),
        # }

    def __len__(self):
        return self.length        

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = Image.open(self.x_set[idx]).convert('RGB')
        #x = x.transpose(2, 0, 1) #if x is np array and it has format Width * Height * Channel
        y = self.y_set[idx]


        if self.color_jitter:
        # color jitter requires a pillow imageng
        # so you need to convert to pillow like Image.fromarray
        # needs integer
            x = self.COLOR_JITTER(x)

        '''TODO Add transformations'''
        # if True:
        #     x = self.data_transforms['val' if self.is_eval else 'train'](x)





        x = numpy.asarray(x).copy().transpose(2, 0, 1)
        x = (x - 128.) / 128.
        x = torch.from_numpy(x).type(torch.float)
        return x, torch.tensor(y), self.x_set[idx]
