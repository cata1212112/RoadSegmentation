import numpy as np

import preprocessing
from torch.utils.data import DataLoader
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import constants
import dataset

testTransforms = A.Compose([
    A.Normalize(mean=constants.DATA_MEAN,std=constants.DATA_STD,max_pixel_value=255.0),
    ToTensorV2()
])
def predict(model, imagePath, device):
    model.eval()

    one, two, three, four = dataset.loadTestImage(imagePath)
    # image = dataset.loadTestImage(imagePath)

    # one = image[:400, :400, :]
    # two = image[208:608, :400, :]
    # three = image[:400, 208:608, :]
    # four =image[208:608, 208:608, :]

    testDataset = preprocessing.CustomDataset([one, two, three, four], [np.zeros_like(one), np.zeros_like(one), np.zeros_like(one), np.zeros_like(one)], testTransforms)
    # testDataset = preprocessing.CustomDataset([image], [np.zeros_like(image)], testTransforms)
    testLoader = DataLoader(testDataset, 4)
    # testLoader = DataLoader(testDataset, len(image))


    with torch.no_grad():
        for imgBatch in testLoader:
            imgBatch = imgBatch['image'].to(device=device, dtype=torch.float32)

            pred = torch.sigmoid(model(imgBatch))

            pred = (pred > 0.5).float().detach().cpu().numpy()
            pred = np.array(pred, dtype=np.uint8)
            pred = pred * 255
            return pred[:, :]