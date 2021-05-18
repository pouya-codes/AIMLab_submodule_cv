import torch
from torchvision import transforms
import random

class SizeJitter(object):
    """
    Cutting out random places in image.
    """

    def __init__(self, size_perc, prob=0.5, color="black"):
        assert isinstance(size_perc, float)
        assert isinstance(prob, float)

        self.size_perc = size_perc
        self.prob = prob

        if color=="white":
            self.color = 1.0
        elif color=="black":
            self.color = 0.0
        else:
            raise NotImplementedError(f"{color} is not implemented!")

    def __call__(self, tensor_img):
        assert isinstance(tensor_img, torch.Tensor)

        if random.random() >= self.prob:

            _, H, W = tensor_img.shape
            resize_size = (int(H*self.size_perc), int(W*self.size_perc))
            resized_img = transforms.Resize(resize_size)(tensor_img)

            if self.size_perc < 1:
                pad_H = int((H-resize_size[0])/2)
                pad_W = int((W-resize_size[1])/2)
                pad_size = (pad_H, pad_W, H-pad_H-resize_size[0], W-pad_W-resize_size[1])
                out_tensor_img = transforms.Pad(pad_size, fill=self.color)(resized_img)

            elif self.size_perc > 1:
                crop_size = (H,W)
                out_tensor_img = transforms.RandomCrop(crop_size)(resized_img)

            else:
                out_tensor_img = tensor_img
            # ToTensor Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
            # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
            return out_tensor_img
        return tensor_img
