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

    patches = dataset.loadTestImage(imagePath)

    testDataset = preprocessing.CustomDataset(patches, np.zeros_like(patches), testTransforms)
    testLoader = DataLoader(testDataset, len(patches))

    segmentation = np.zeros((96 * int(np.sqrt(len(patches))), 96 * int(np.sqrt(len(patches)))))

    with torch.no_grad():
        for imgBatch in testLoader:
            imgBatch = imgBatch['image'].to(device)

            pred = torch.sigmoid(model(imgBatch))

            pred = (pred > 0.5).float().detach().cpu().numpy()
            pred = np.array(pred, dtype=np.uint8)
            pred = pred * 255

            index = 0
            for i in range(0, segmentation.shape[1], 96):
                for j in range(0, segmentation.shape[1], 96):
                    segmentation[i:i+96, j:j+96] += pred[index][0]
                    index += 1
            return np.array(segmentation, dtype=np.uint8)