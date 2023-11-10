import cv2
from PIL import Image
import os
import numpy as np
import constants
import albumentations as A
from sklearn.model_selection import train_test_split

def extractPatches(images, masks, patchSize=112):
    patchesImg = []
    patchesMask = []

    for image, mask in zip(images, masks):
        height = image.shape[0]
        width = image.shape[1]

        heightPad = height % patchSize
        widthPad = width % patchSize

        newImage = np.array(image)
        newMask = np.array(mask)

        newHeight = height
        newWidth = width

        if heightPad != 0 or widthPad != 0:
            newHeight = height + patchSize - heightPad
            newWidth = width + patchSize - widthPad
            padded = A.PadIfNeeded(min_height=newHeight, min_width=newWidth, p=1)(image=image, mask=mask)
            newImage = np.array(padded['image'])
            newMask = np.array(padded['mask'])


        for i in range(0, newHeight, patchSize):
            for j in range(0, newWidth, patchSize):
                patchImg = newImage[i:i + patchSize, j:j + patchSize, :]
                patchMask = newMask[i:i + patchSize, j:j + patchSize]

                patchesImg.append(newImage[i:i + patchSize, j:j + patchSize, :])
                patchesMask.append(newMask[i:i + patchSize, j:j + patchSize])


    return np.array(patchesImg), np.array(patchesMask)

def extractPatchesTest(image, patchSize=96):
    height = image.shape[0]
    width = image.shape[1]

    patchesImg = []

    heightPad = height % patchSize
    widthPad = width % patchSize

    newImage = np.array(image)

    newHeight = height
    newWidth = width

    if heightPad != 0 or widthPad != 0:
        newHeight = height + patchSize - heightPad
        newWidth = width + patchSize - widthPad
        padded = A.PadIfNeeded(min_height=newHeight, min_width=newWidth, p=1)(image=image, mask=np.zeros_like(image))
        newImage = np.array(padded['image'])

    for i in range(0, newHeight, patchSize):
        for j in range(0, newWidth, patchSize):
            patchesImg.append(newImage[i:i + patchSize, j:j + patchSize, :])

    return np.array(patchesImg)

def loadDataset():
    imgNames = os.listdir(constants.trainingImages)

    images = []
    masks = []

    for img in imgNames:
        image = Image.open(os.path.join(constants.trainingImages, img)).convert("RGB")
        mask = Image.open(os.path.join(constants.trainingMasks, img)).convert("L")

        images.append(np.array(image))
        masks.append(np.array(mask))

    images, masks = extractPatches(images, masks)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks)
    masks[masks > 0] = 1.

    shuffle = np.random.permutation(images.shape[0])
    images = images[shuffle]
    masks = masks[shuffle]

    return images, masks

def getTrainValSplit():
    X, y = loadDataset()
    return train_test_split(X, y, test_size=0.33, random_state=42)

def loadTestImage(path):
    image = Image.open(path).convert("RGB")
    image = np.array(image)

    return extractPatchesTest(image)