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
import numpy as np
from segmentation_models_pytorch import losses as L

trainTransforms = A.Compose([
    A.RandomRotate90(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.Rotate(p=0.5, border_mode=cv2.BORDER_REFLECT),
    A.Normalize(mean=constants.DATA_MEAN,std=constants.DATA_STD,max_pixel_value=255.0,),
    # A.Resize(224, 224),
    ToTensorV2()
])

validationTransforms = A.Compose([
    A.Normalize(mean=constants.DATA_MEAN,std=constants.DATA_STD,max_pixel_value=255.0,),
    # A.Resize(224, 224),
    ToTensorV2()
])

def trainMethod(model, epochs, batchSize, learningRate, weightDecay, device):
    Xtrain, Xval, ytrain, yval = dataset.getTrainValSplit()

    nTrain = len(Xtrain)
    nVal = len(Xval)

    trainDataset = preprocessing.CustomDataset(Xtrain, ytrain, trainTransforms)
    validationDataset = preprocessing.CustomDataset(Xval, yval, validationTransforms)

    trainLoader = DataLoader(trainDataset, batchSize)
    valLoader = DataLoader(validationDataset, batchSize)

    writer = SummaryWriter(comment=f'LR_{learningRate}_BS_{batchSize}')

    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    # lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    lossFunction = nn.BCEWithLogitsLoss()
    # lossFunction = nn.CrossEntropyLoss()
    # lossFunction = L.JaccardLoss(mode='binary', from_logits=True)
    # lossFunction = L.FocalLoss(mode='binary')
    # lossFunction = L.TverskyLoss(mode='binary', from_logits=True, alpha=0.4, beta=0.6)
    # lossFunction = L.DiceLoss(mode='binary', from_logits=True)

    for epoch in range(epochs):
        model.train()

        epochLoss = 0

        with tqdm(total=nTrain, desc=f"Epoch {epoch + 1}/{epochs}", unit='image') as progessBar:
            for batch in trainLoader:
                X = batch['image']
                y = batch['mask']
                X = X.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32).squeeze(1)

                optimizer.zero_grad()
                pred = model(X).squeeze(1)

                opposite_mask_tensor = 1 - y
                original_mask_tensor = y.unsqueeze(1)
                opposite_mask_tensor = opposite_mask_tensor.unsqueeze(1)

                combined_mask_tensor = torch.cat((original_mask_tensor, opposite_mask_tensor), dim=1)

                loss = lossFunction(pred, y)
                epochLoss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), epoch)

                progessBar.set_postfix(**{'loss (batch)': loss.item()})

                loss.backward()
                optimizer.step()
                # lrScheduler.step(loss)

                progessBar.update(X.shape[0])

                if epoch % 1 == 0:
                    validationScoreDice, validationScoreJaccard, lossValidation = validationMethod(model, valLoader, device, lossFunction)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                    writer.add_scalar('Dice/validation', validationScoreDice, epoch)
                    writer.add_scalar('F1Score/validation', validationScoreJaccard, epoch)
                    writer.add_scalar('Loss/validation', lossValidation, epoch)

                    writer.add_images('images', X, epoch)
                    writer.add_images('masks/true', (y.unsqueeze(1)) * 255, epoch)
                    writer.add_images('mask/predicted', (torch.sigmoid(pred) > 0.5).unsqueeze(1) * 255, epoch)
                    writer.add_scalar('Loss/train', loss.item(), epoch)

        torch.save(model.state_dict(), "best/segnet.pth")



def validationMethod(model, loader, device, lossVal):
    model.eval()

    nVal = len(loader)
    totalDice = 0
    totalLoss = 0
    dice = torchmetrics.Dice(average='micro').to(device)
    jaccard = torchmetrics.F1Score(task='binary', average='micro').to(device)

    totalJaccard = 0

    for batch in loader:
        X = batch['image']
        y = batch['mask']

        X = X.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.uint8).squeeze(1)
        y_true = y.to(device=device, dtype=torch.float32).squeeze(1)

        with torch.no_grad():
            pred = model(X).squeeze(1)

        opposite_mask_tensor = 1 - y_true
        original_mask_tensor = y_true.unsqueeze(1)
        opposite_mask_tensor = opposite_mask_tensor.unsqueeze(1)

        combined_mask_tensor = torch.cat((original_mask_tensor, opposite_mask_tensor), dim=1)

        totalDice += dice(pred, y)
        totalJaccard += jaccard(pred, y)

        totalLoss += lossVal(pred, y_true).item()
        y_true.detach()

    model.train()
    return totalDice / nVal, totalJaccard / nVal, totalLoss / nVal