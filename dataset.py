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
    resize = A.Resize(352, 352, interpolation=cv2.INTER_AREA, p=1)
    flipHorizontal = A.HorizontalFlip(p=1)

    cnt = 0
    for image, mask in zip(images, masks):
        aug = resize(image=image, mask=mask)
        image = aug['image']
        mask = aug['mask']
        augmentedImages.append(image)
        augmentedMasks.append(mask)

        # one_image = image[:256, :256, :]
        # one_mask = mask[:256, :256]
        #
        # augmentedImages.append(one_image)
        # augmentedMasks.append(one_mask)
        #
        # two_image = image[144:400, :256, :]
        # two_mask = mask[144:400, :256]
        #
        # augmentedImages.append(two_image)
        # augmentedMasks.append(two_mask)
        #
        # three_image = image[:256, 144:400, :]
        # three_mask = mask[:256, 144:400]
        #
        # augmentedImages.append(three_image)
        # augmentedMasks.append(three_mask)
        #
        # four_image = image[144:400, 144:400, :]
        # four_mask = mask[144:400, 144:400]
        #
        # augmentedImages.append(four_image)
        # augmentedMasks.append(four_mask)
        #
        # Image.fromarray(image).save(f"imagini/sat_{cnt}.jpg")
        # Image.fromarray(mask).save(f"imagini/gt_{cnt}.jpg")
        # cnt += 1
        # aug = flipVertical(image=image, mask=mask)
        # imageAug = aug['image']
        # imageMask = aug['mask']
        # augmentedImages.append(imageAug)
        # augmentedMasks.append(imageMask)
        #
        # aug = flipHorizontal(image=image, mask=mask)
        # imageAug = aug['image']
        # imageMask = aug['mask']
        # augmentedImages.append(imageAug)
        # augmentedMasks.append(imageMask)
        # #
        # imageAug, imageMask = rotateWithReflectMode(image, mask, 225)
        # augmentedImages.append(imageAug)
        # augmentedMasks.append(imageMask)
        #
        # imageAug, imageMask = rotateWithReflectMode(image, mask, 45)
        # augmentedImages.append(imageAug)
        # augmentedMasks.append(imageMask)
        #
        # Image.fromarray(imageAug).save(f"imagini/sat_{cnt}.jpg")
        # Image.fromarray(imageMask).save(f"imagini/gt_{cnt}.jpg")
        #
        # cnt += 1
        #
        # imageAug, imageMask = rotateWithReflectMode(image, mask, 90)
        # augmentedImages.append(imageAug)
        # augmentedMasks.append(imageMask)
        #
        # Image.fromarray(imageAug).save(f"imagini/sat_{cnt}.jpg")
        # Image.fromarray(imageMask).save(f"imagini/gt_{cnt}.jpg")
        #
        # cnt += 1
        #
        # imageAug, imageMask = rotateWithReflectMode(image, mask, 135)
        # augmentedImages.append(imageAug)
        # augmentedMasks.append(imageMask)
        #
        # Image.fromarray(imageAug).save(f"imagini/sat_{cnt}.jpg")
        # Image.fromarray(imageMask).save(f"imagini/gt_{cnt}.jpg")
        #
        # imageAug, imageMask = rotateWithReflectMode(image, mask, 270)
        # augmentedImages.append(imageAug)
        # augmentedMasks.append(imageMask)
        #
        # cnt += 1

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
    # images, masks = extractPatches(images, masks)

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
    return train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

def loadTestImage(path):
    image = Image.open(path).convert("RGB")
    image = np.array(image)

    one, two, three, four = image[:400, :400, :], image[208:608, :400, :], image[:400, 208:608, :], image[208:608, 208:608, :]

    one = cv2.resize(one, (352, 352), cv2.INTER_AREA)
    two = cv2.resize(two, (352, 352), cv2.INTER_AREA)
    three = cv2.resize(three, (352, 352), cv2.INTER_AREA)
    four = cv2.resize(four, (352, 352), cv2.INTER_AREA)

    return one, two, three, four
    # return image