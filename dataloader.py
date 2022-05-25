import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional

# from utils import full_mask
# full_masks = full_mask()
# y_list, x_list = full_masks.nonzero()


class RandomCrop(torch.nn.Module):

    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int], idx:int, x_list:List, y_list:List) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.size
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger then input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w
        
#         idx = np.random.choice(len(x_list))
        cx, cy = x_list[idx], y_list[idx]
        h_half = int(th / 2.)
        w_half = int(tw / 2.)
        
        i = cx - w_half
        j = cy - h_half
        return i, j, th, tw

    def __init__(self, size, idx, padding=None, x_list=None, y_list=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.x_list = x_list
        self.y_list = y_list
        self.padding_mode = padding_mode
        self.idx = idx
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        h, w = img.size
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size, self.idx, x_list=self.x_list, y_list= self.y_list)
       
        return img.crop((i, j, i+h, j+w))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"
        

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy.random as nr
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import random

class TorchvisionDataset(Dataset):
    def __init__(self, df, masks, img_size, forecast_steps, crops, x_list, y_list, add_loss=None, transform=None):
        self.file_paths = df
        self.masks = masks
        self.transform = transform
        self.cus_crop = crops
        self.x_list = x_list
        self.y_list = y_list
        self.img_size = img_size
        self.forecast_steps = forecast_steps
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image_list = [Image.open(path) for path in img_path]
        idx = np.random.choice(len(self.x_list))
        images = [self.cus_crop(self.img_size, idx, self.x_list, self.y_list)(img) for img in image_list]
        masked = self.cus_crop(self.img_size, idx, self.x_list, self.y_list)(self.masks)
        images = [self.transform(np.array(img) * np.array(masked)) for img in images]
        images = torch.stack(images, dim=0)
        return (images[0:4, :, :], images[4 : 4 + self.forecast_steps, :, :])