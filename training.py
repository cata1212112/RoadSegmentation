import os

import constants
import preprocessing
from torch.utils.data import DataLoader, random_split
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import  torchmetrics
import albumentations as A
from albumentations.pytorch import ToTensorV2

trainTransforms = A.Compose([
    A.Normalize(mean=constants.DATA_MEAN,std=constants.DATA_STD,max_pixel_value=255.0,),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2()
])

validationTransforms = A.Compose([
    A.Normalize(mean=constants.DATA_MEAN,std=constants.DATA_STD,max_pixel_value=255.0,),
    ToTensorV2()
])

def trainMethod(model, epochs, batchSize, learningRate, weightDecay, device):
    imgNames = os.listdir(constants.trainingImages)

    trainNames = imgNames[:67]
    valNames = imgNames[67:]

    nTrain = len(trainNames)
    nVal = len(valNames)

    trainDataset = preprocessing.CustomDataset(constants.trainingImages, trainNames, trainTransforms,  constants.trainingMasks)
    validationDataset = preprocessing.CustomDataset(constants.trainingImages, valNames, validationTransforms,  constants.trainingMasks)

    trainLoader = DataLoader(trainDataset, batchSize, shuffle=True)
    valLoader = DataLoader(validationDataset, batchSize, shuffle=True)

    writer = SummaryWriter(comment=f'LR_{learningRate}_BS_{batchSize}')

    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
    lossFunction = nn.BCEWithLogitsLoss()
    globalStep = 0
    for epoch in range(epochs):
        model.train()

        epochLoss = 0

        with tqdm(total=nTrain, desc=f"Epoch {epoch + 1}/{epochs}", unit='image') as progessBar:
            for batch in trainLoader:
                X = batch['image']
                y = batch['mask']

                X = X.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32).squeeze(1)

                pred = model(X).squeeze(1)
                loss = lossFunction(pred, y)
                epochLoss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), globalStep)

                progessBar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progessBar.update(X.shape[0])
                globalStep += 1

                if globalStep % 5 == 0:
                    validationScore = validationMethod(model, valLoader, device)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], globalStep)

                    writer.add_scalar('Dice/test', validationScore, globalStep)

                    writer.add_images('images', X, globalStep)
                    writer.add_images('masks/true', pred.unsqueeze(1), globalStep)
                    writer.add_images('mask/pred', (torch.sigmoid(pred) > 0.5).unsqueeze(1), globalStep)



def validationMethod(model, loader, device):
    model.eval()

    nVal = len(loader)
    total = 0
    dice = torchmetrics.Dice(average='micro').to(device)

    for batch in loader:
        X = batch['image']
        y = batch['mask']

        X = X.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.uint8).squeeze(1)

        with torch.no_grad():
            pred = model(X).squeeze(1)

        total += dice(pred, y)

    model.train()
    return total / nVal