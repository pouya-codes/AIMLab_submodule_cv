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
    def __init__(self, x_set, y_set, model_config=None, training_set=False):

        """
        Args:
            x_set (string): List of paths to images
            y_set (int): Labels of each image in x_set
            model_config (dict): Dict of model config.
        """
        self.x_set = x_set
        self.y_set = y_set

        self.training_set = training_set
        if model_config :
            self.normalize = True if 'normalize' in model_config and model_config['normalize']['normalize'] else False
            self.augmentation = True if 'augmentation' in model_config and model_config['augmentation']['augmentation'] else False
        else:
            self.normalize= False
            self.augmentation= False

        transforms_array = []
        if self.augmentation:
            if self.training_set:
                transforms_array.append(transforms.RandomHorizontalFlip())
                transforms_array.append(transforms.RandomVerticalFlip())
                transforms_array.append(transforms.ColorJitter(hue=.05, saturation=.05))
                if 'resize' in model_config['augmentation']:
                    transforms_array.append(transforms.Resize(model_config['augmentation']['resize']))
                if 'crop' in model_config['augmentation']:
                    transforms_array.append(transforms.RandomCrop(model_config['augmentation']['crop']))
                transforms_array.append(transforms.RandomRotation(20, resample=Image.BILINEAR))
            else:
                if 'resize' in model_config['augmentation']:
                    transforms_array.append(transforms.Resize(model_config['augmentation']['resize']))
                if 'crop' in model_config['augmentation']:
                    transforms_array.append(transforms.RandomCrop(model_config['augmentation']['crop']))
        transforms_array.append(ToTensor.ToTensor())
        if (self.normalize):
            transforms_array.append(Normalize.Normalize(mean=model_config['normalize']['mean'], std=model_config['normalize']['std']))
        else:
            transforms_array.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))

        # if (self.normalize or self.augmentation) :
        self.transform = transforms.Compose(transforms_array)
        print(self.transform)


        self.length = len(x_set)
        if len(x_set) != len(y_set):
            raise ValueError('x set length does not match y set length')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = Image.open(self.x_set[idx]).convert('RGB')
        #x = x.transpose(2, 0, 1) #if x is np array and it has format Width * Height * Channel
        y = self.y_set[idx]

        x = self.transform(x)
        # else :
            # x = numpy.asarray(x).copy().transpose(2, 0, 1)
            # x = (x - 128.) / 128. # must be in [-1, 1] range
            # x = torch.from_numpy(x).type(torch.float)
        return x, torch.tensor(y), self.x_set[idx]
