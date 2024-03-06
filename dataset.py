import cv2
from PIL import Image
import os
import numpy as np
import constants
import albumentations as A
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import copy
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import exposure

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
    augmentedImages = []
    augmentedMasks = []

    for image, mask in zip(images, masks):
        # augmentedImages.append(image)
        # augmentedMasks.append(mask)

        # for i in [0, 100, 200, 300]:
        #     for j in [0, 100, 200, 300]:
        #         patch_image = image[i:i + 100, j:j + 100]
        #         patch_mask = mask[i:i + 100, j:j + 100]
        #
        #         augmentedImages.append(copy.deepcopy(patch_image))
        #         augmentedMasks.append(copy.deepcopy(patch_mask))

        for i in [0, 160]:
            for j in [0, 160]:
                patch_image = image[i:i + 240, j:j + 240]
                patch_mask = mask[i:i + 240, j:j + 240]

                augmentedImages.append(copy.deepcopy(patch_image))
                augmentedMasks.append(copy.deepcopy(patch_mask))

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
    patches = []
    # for i in [0, 100, 200, 300, 400, 500]:
    #     for j in [0, 100, 200, 300, 400, 500]:
    #         patches.append(image[i:i + 100, j:j + 100])

    for i in [0, 240, 368]:
        for j in [0, 240, 368]:
            patches.append(image[i:i + 240, j:j + 240])

    # one, two, three, four = image[:400, :400, :], image[208:608, :400, :], image[:400, 208:608, :], image[208:608,
    #                                                                                                 208:608, :]
    #
    # one = cv2.resize(one, (352, 352), cv2.INTER_AREA)
    # two = cv2.resize(two, (352, 352), cv2.INTER_AREA)
    # three = cv2.resize(three, (352, 352), cv2.INTER_AREA)
    # four = cv2.resize(four, (352, 352), cv2.INTER_AREA)

    return patches
    # return image

def showHog(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()