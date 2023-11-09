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
    def __init__(self, imgsDir, imgNames, transforms, masksDir=None):
        self.imgsDir = imgsDir
        self.masksDir = masksDir

        self.imageNames = imgNames
        self.transform = transforms

    def __len__(self):
        return len(self.imageNames)

    def __getitem__(self, item):
        name = self.imageNames[item]

        image = Image.open(os.path.join(self.imgsDir, name)).convert("RGB")

        width, height = image.size
        if width > 400:
            image = image.crop((0, 0, 400, 400))

        mask = None
        if self.masksDir != None:
            mask = Image.open(os.path.join(self.masksDir, name)).convert("L")

        image = np.array(image)
        if mask != None:
            mask = np.array(mask)
            mask[mask > 0] = 1
            return self.transform(image=image, mask=mask)
        else:
            return self.transform(image=image, mask=np.zeros_like(image))['image']
