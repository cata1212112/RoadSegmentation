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
    # A.Resize(224, 224),
    ToTensorV2()
])
def predict(model, imagePath, device):
    model.eval()

    patches = dataset.loadTestImage(imagePath)
    testDataset = preprocessing.CustomDataset(patches, [np.zeros_like(patch) for patch in patches], testTransforms)
    testLoader = DataLoader(testDataset, len(patches))

    with torch.no_grad():
        for imgBatch in testLoader:
            imgBatch = imgBatch['image'].to(device=device, dtype=torch.float32)

            pred = torch.sigmoid(model(imgBatch))

            pred = (pred > 0.5).float().detach().cpu().numpy()
            pred = np.array(pred, dtype=np.uint8)
            pred = pred * 255


            # pred = model(imgBatch)
            # pred = pred.float().detach().cpu().numpy()
            return pred[:, :]