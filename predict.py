import numpy as np

import preprocessing
from torch.utils.data import DataLoader
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import constants
import dataset

testTransforms = A.Compose([
    # A.Normalize(mean=constants.DATA_MEAN,std=constants.DATA_STD,max_pixel_value=255.0),
    ToTensorV2()
])
def predict(model, imagePath, device):
    model.eval()

    image = dataset.loadTestImage(imagePath)


    testDataset = preprocessing.CustomDataset([image], [np.zeros_like(image)], testTransforms)
    testLoader = DataLoader(testDataset, len(image))

    segmentation = np.zeros_like(image)

    with torch.no_grad():
        for imgBatch in testLoader:
            imgBatch = imgBatch['image'].to(device=device, dtype=torch.float32)

            # imgBatch = imgBatch[:, :, :400, :400]

            pred = torch.sigmoid(model(imgBatch))

            pred = (pred > 0.5).float().detach().cpu().numpy()
            pred = np.array(pred, dtype=np.uint8)
            pred = pred * 255
            return pred[0][0]