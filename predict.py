import preprocessing
from torch.utils.data import DataLoader
import torch
def predict(model, imageDir, device):
    model.eval()
    testDataset = preprocessing.CustomDataset(imageDir, None)
    testLoader = DataLoader(testDataset, 1)

    with torch.no_grad():
        for imgBatch in testLoader:
            imgBatch = imgBatch.to(device)

            pred = torch.sigmoid(model(imgBatch))

            pred = (pred > 0.5).float().detach().cpu().numpy()
            return pred[0][0]