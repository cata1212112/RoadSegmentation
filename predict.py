import preprocessing
from torch.utils.data import DataLoader
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import constants

testTransforms = A.Compose([
    A.Normalize(mean=constants.DATA_MEAN,std=constants.DATA_STD,max_pixel_value=255.0,),
    ToTensorV2()
])
def predict(model, imageDir, imgName, device):
    model.eval()
    testDataset = preprocessing.CustomDataset(imageDir, [imgName], testTransforms, None)
    testLoader = DataLoader(testDataset, 1)

    with torch.no_grad():
        for imgBatch in testLoader:
            imgBatch = imgBatch.to(device)

            pred = torch.sigmoid(model(imgBatch))

            pred = (pred > 0.5).float().detach().cpu().numpy()
            return pred[0][0]