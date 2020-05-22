import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
from skimage import io, transform
import numpy
import torchvision

import submodule_cv.preprocess as preprocess

class SlidePatchExtractor(object):
    def __init__(self, os_slide, patch_size, resize_sizes=None):
        '''
        Parameters
        ----------
        os_slide : OpenSlide
            OpenSlide slide to extract patches from

        patch_size : int
            The size of the patch to extract
        
        resize_sizes : list of int
            A list of multiple sizes to resize
        '''
        self.os_slide = os_slide
        self.patch_size = patch_size
        self.resize_sizes = resize_sizes
        self.width, self.height = self.os_slide.dimensions
        self.tile_width = int(self.width / self.patch_size)
        self.tile_height = int(self.height / self.patch_size)

    def __len__(self):
        return self.tile_width * self.tile_height

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        tile_x = idx % self.tile_width
        tile_y = int(idx / self.tile_width)
        x = tile_x * self.patch_size
        y = tile_y * self.patch_size
        patch = preprocess.extract(self.os_slide, x, y, self.patch_size)
        # patch = preprocess.extract_and_resize(self.os_slide,
        #         x, y, self.patch_size, self.resize_size)
        if self.resize_sizes:
            resized_patches = { }
            for resize_size in self.resize_sizes:
                if resize_size == self.patch_size:
                    resized_patches[resize_size] = patch
                else:
                    resized_patches[resize_size] = preprocess.resize(patch, resize_size)
            return patch, (tile_x, tile_y,), resized_patches
        else:
            return patch, (tile_x, tile_y,)

class PatchDataset(Dataset):
    def __init__(self, x_set, y_set, transform=None, color_jitter=False):
        """
        Args:
            x_set (string): List of paths to images
            y_set (int): Labels of each image in x_set
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x_set = x_set
        self.y_set = y_set
        self.length = len(x_set)
        self.transform = transform
        self.t = transforms.Compose([transforms.ToTensor()])
        if len(x_set) != len(y_set):
            raise ValueError('x set length does not match y set length')

        self.color_jitter = color_jitter
        self.COLOR_JITTER = torchvision.transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)

    def __len__(self):
        return self.length        

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = Image.open(self.x_set[idx]).convert('RGB')
        #x = x.transpose(2, 0, 1) #if x is np array and it has format Width * Height * Channel
        y = self.y_set[idx]

        #if self.transform:
        #   sample = self.transform(sample)

        if self.color_jitter:
        # color jitter requires a pillow imageng
        # so you need to convert to pillow like Image.fromarray
        # needs integer
            x = self.COLOR_JITTER(x)

        x = numpy.asarray(x).copy().transpose(2, 0, 1)
        x = (x - 128.) / 128.
        x = torch.from_numpy(x).type(torch.float)


        return x, torch.tensor(y)
