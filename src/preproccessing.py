import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets, transforms

import cv2
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
%matplotlib inline

#PARAMS
#Use cuda if available as device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Use all available cores for dataloading
num_workers = os.cpu_count()
batch_size = 32
input_size  = 224
data_dir = "./finetuning"

augs = A.Compose([A.Resize(height  = input_size, 
                           width   = input_size),
                  A.Normalize(mean = (0), 
                              std  = (1)),
                  ToTensorV2()])

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), augs) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}

##### COMPUTE PIXEL SUM AND SQUARED SUM
for x in ['train', 'val']:
    # placeholders
    psum    = torch.tensor([0.0])
    psum_sq = torch.tensor([0.0])

    # loop through images
    for inputs in tqdm(dataloaders_dict[x]):
        psum    += inputs.sum(axis        = [0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])

    ##### FINAL CALCULATIONS

    # pixel count: all images in the training set multiplied by the image size twice
    count = len(image_datasets[x]) * input_size * input_size

    # mean and STD
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    if x == 'train':
        print('Training data stats:')
    else:
        print('Validation data stats:')
    print('- mean: {:.4f}'.format(total_mean.item()))
    print('- std:  {:.4f}'.format(total_std.item()))

