import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision.transforms as transforms
import constants
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self, X, y, transform):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        image = self.X[item]
        mask = self.y[item]

        if mask is None:
            mask = np.zeros_like(image)
        return self.transform(image=image, mask=mask)