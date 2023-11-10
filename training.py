import os

import cv2

import constants
import preprocessing
from torch.utils.data import DataLoader, random_split
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import  torchmetrics
import albumentations as A
from albumentations.pytorch import ToTensorV2
import dataset
from segmentation_models_pytorch import losses as L

trainTransforms = A.Compose([

    A.HorizontalFlip(),
    A.VerticalFlip(),

    A.RandomRotate90(),
    A.Transpose(),

    A.Rotate(limit=90, p=0.7, border_mode=cv2.BORDER_REFLECT),

    A.Normalize(mean=constants.DATA_MEAN,std=constants.DATA_STD,max_pixel_value=255.0,),
    ToTensorV2()
])

validationTransforms = A.Compose([
    A.Normalize(mean=constants.DATA_MEAN,std=constants.DATA_STD,max_pixel_value=255.0,),
    ToTensorV2()
])

def trainMethod(model, epochs, batchSize, learningRate, weightDecay, device):
    Xtrain, Xval, ytrain, yval = dataset.getTrainValSplit()


    nTrain = len(Xtrain)
    nVal = len(Xval)

    trainDataset = preprocessing.CustomDataset(Xtrain, ytrain, trainTransforms)
    validationDataset = preprocessing.CustomDataset(Xval, yval, validationTransforms)

    trainLoader = DataLoader(trainDataset, batchSize, shuffle=True)
    valLoader = DataLoader(validationDataset, batchSize, shuffle=True)

    writer = SummaryWriter(comment=f'LR_{learningRate}_BS_{batchSize}')

    optimizer = torch.optim.Adamax(model.parameters(), lr=learningRate)
    # lossFunction = nn.BCEWithLogitsLoss()
    # lossFunction = L.JaccardLoss(mode='binary', from_logits=True)
    lossFunction = L.FocalLoss(mode='binary')
    # lossFunction = L.TverskyLoss(mode='binary', from_logits=True, alpha=0.4, beta=0.6)

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
                    validationScoreDice, validationScoreJaccard = validationMethod(model, valLoader, device)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], globalStep)

                    writer.add_scalar('Dice/validation', validationScoreDice, globalStep)
                    writer.add_scalar('Jaccard/validation', validationScoreJaccard, globalStep)

                    writer.add_images('images', X, globalStep)
                    writer.add_images('masks/true', (y.unsqueeze(1)) * 255, globalStep)
                    writer.add_images('mask/predicted', (torch.sigmoid(pred) > 0.5).unsqueeze(1), globalStep)



def validationMethod(model, loader, device):
    model.eval()

    nVal = len(loader)
    totalDice = 0
    dice = torchmetrics.Dice(average='micro').to(device)
    jaccard = torchmetrics.JaccardIndex(task='binary', average='micro').to(device)

    totalJaccard = 0

    for batch in loader:
        X = batch['image']
        y = batch['mask']

        X = X.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.uint8).squeeze(1)

        with torch.no_grad():
            pred = model(X).squeeze(1)

        totalDice += dice(pred, y)
        totalJaccard += jaccard(pred, y)

    model.train()
    return totalDice / nVal, totalJaccard / nVal