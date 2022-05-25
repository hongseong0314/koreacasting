from PIL import Image
import os
import sys

sys.path.append("skillful_nowcasting")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional
from glob import glob
from utils import files_detail
from utils import make_windowed_dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy.random as nr
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import random
from dataloader import TorchvisionDataset, RandomCrop
from utils import full_mask

# train data 구성
files = glob(os.path.join(os.getcwd(), 'data/train/*.png'))
file_df = files_detail(files)

# step 구성된 데이터로 변환
time_step = 24
df = make_windowed_dataset(file_df, start_index=0, time_step=time_step, end_index=None, step=1)

full_masks = full_mask()

img_size = (256,256)
forecast_steps = 8
train_load = TorchvisionDataset(df=df,
                                masks=Image.fromarray(full_masks),
                                img_size=img_size,
                                forecast_steps = forecast_steps,
                                crops=RandomCrop,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize([0.5], [0.5]),]))

train_loader = torch.utils.data.DataLoader(train_load, batch_size=1)
val_loader = torch.utils.data.DataLoader(train_load, batch_size=1)


import torch
import torch.nn.functional as F
from dgmr import (
    DGMR,
    Generator,
    Discriminator,
    TemporalDiscriminator,
    SpatialDiscriminator,  
    Sampler,
    LatentConditioningStack,
    ContextConditioningStack,
)
from dgmr.layers import ConvGRU
from dgmr.layers.ConvGRU import ConvGRUCell
from dgmr.common import DBlock, GBlock
import einops
from pytorch_lightning import Trainer

trainer = Trainer(gpus=0, max_epochs=10)
model = DGMR(forecast_steps=forecast_steps)

if __name__ == '__main__':
    trainer.fit(model, train_loader, val_loader)