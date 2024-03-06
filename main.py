import cv2

import predict
import unet
import resunet
import torch
import training
import numpy as np
from PIL import Image
import albumentations as A
import dataset
from torchsummary import summary
from VisionTransformer.Segmenter import Segmenter
from SegNet.segnet import SegNet
from AttentionUNET.AttentionUNET import AttentionUNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainNewModel():
    # model = unet.UNet()
    # model.to(device)
    # model = Segmenter()
    # model = SegNet()
    model = AttentionUNet()
    # model = resunet.ResUNet(features=[64, 128, 256, 512])
    # model.load_state_dict(torch.load("best/modelmartie52.pth"))
    model.to(device)

    training.trainMethod(model, 200, 16, 1e-4, 1e-8, device)


def loadModelAndSubmit():
    foreground_threshold = 0.2

    def patch_to_label(patch):
        df = np.mean(patch) / 255

        if df > foreground_threshold:
            return 1
        else:
            return 0

    model = AttentionUNet()
    model.load_state_dict(torch.load("best/segnet.pth"))
    model.to(device)

    file = 'submission.csv'
    with open(file, 'w') as f:
        f.write('id,prediction\n')

        for index in range(1, 51):
            prediction = predict.predict(model, f"data/test_set_images/test_set_images/test_{index}/test_{index}.png", device)

            # prediction = np.array(prediction, dtype=np.uint8)
            # prediction = np.argmax(prediction, axis=1) * 255
            # prediction = prediction.squeeze(axis=1)

            segmentation = np.zeros((608, 608))

            index_patch = 0
            for i in [0, 240, 368]:
                for j in [0, 240, 368]:
                    segmentation[i:i+240, j:j+240] = np.maximum(segmentation[i:i+240, j:j+240], prediction[index_patch])
                    index_patch += 1

            segmentation = np.array(segmentation, dtype=np.uint8)
            img = Image.fromarray(segmentation)
            img.save(f"segmentations/test_{index}.png")

            forSubmissions = np.zeros_like(segmentation, dtype=np.uint8)

            for i in range(0, segmentation.shape[1], 16):
                for j in range(0, segmentation.shape[0], 16):
                    label = patch_to_label(segmentation[j:j+16, i:i+16])
                    f.write("{:03d}_{}_{},{}\n".format(index, i, j, label))
                    forSubmissions[i:i+16, j:j+16] += 255 * label

            Image.fromarray(forSubmissions).save(f"pathes/pred_{index}.jpg")

# trainNewModel()
loadModelAndSubmit()

# dataset.showHog(cv2.imread("data/training/training/images/satImage_002.png"))