import cv2
from PIL import Image
import os
import numpy as np
import constants
import albumentations as A
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

# albumentation does not have this feature
def rotateWithReflectMode(image, mask, degree):
    segmap = SegmentationMapsOnImage(mask, shape=(400, 400))

    augmenter = iaa.Affine(rotate=[degree], mode='reflect')

    augmenter._mode_segmentation_maps = 'reflect'

    aug = augmenter(image=image, segmentation_maps=segmap)

    augImg = aug[0]
    augMask = aug[1].get_arr()

    return augImg, augMask


# return augImg, augMask.get_arr()


def extractPatches(images, masks, patchSize=96):
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

def generateAugmentations(images, masks):
    # rotate the images by 15, 45, 60, 75, 90
    # horizontal flip, vertical flip

    augmentedImages = []
    augmentedMasks = []

    flipVertical = A.VerticalFlip(p=1)
    flipHorizontal = A.HorizontalFlip(p=1)


    for image, mask in zip(images, masks):
        augmentedImages.append(image)
        augmentedMasks.append(mask)

        aug = flipVertical(image=image, mask=mask)
        imageAug = aug['image']
        imageMask = aug['mask']
        augmentedImages.append(imageAug)
        augmentedMasks.append(imageMask)

        aug = flipHorizontal(image=image, mask=mask)
        imageAug = aug['image']
        imageMask = aug['mask']
        augmentedImages.append(imageAug)
        augmentedMasks.append(imageMask)

        imageAug, imageMask = rotateWithReflectMode(image, mask, 15)
        augmentedImages.append(imageAug)
        augmentedMasks.append(imageMask)

        imageAug, imageMask = rotateWithReflectMode(image, mask, 30)
        augmentedImages.append(imageAug)
        augmentedMasks.append(imageMask)

        imageAug, imageMask = rotateWithReflectMode(image, mask, 45)
        augmentedImages.append(imageAug)
        augmentedMasks.append(imageMask)

        imageAug, imageMask = rotateWithReflectMode(image, mask, 60)
        augmentedImages.append(imageAug)
        augmentedMasks.append(imageMask)

        imageAug, imageMask = rotateWithReflectMode(image, mask, 75)
        augmentedImages.append(imageAug)
        augmentedMasks.append(imageMask)

        imageAug, imageMask = rotateWithReflectMode(image, mask, 90)
        augmentedImages.append(imageAug)
        augmentedMasks.append(imageMask)

    return np.array(augmentedImages), np.array(augmentedMasks)

def loadDataset():
    imgNames = os.listdir(constants.trainingImages)

    images = []
    masks = []

    for img in imgNames:
        image = Image.open(os.path.join(constants.trainingImages, img)).convert("RGB")
        mask = Image.open(os.path.join(constants.trainingMasks, img)).convert("L")

        images.append(np.array(image))
        masks.append(np.array(mask))

    images, masks = generateAugmentations(images, masks)
    images, masks = extractPatches(images, masks)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks)

    masks[masks <= 127] = 0
    masks[masks > 127] = 1

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

    return image
    # return extractPatchesTest(image)