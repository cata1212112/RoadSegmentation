import cv2

import predict
import unet
import torch
import training
import numpy as np
from PIL import Image
import albumentations as A
import dataset
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def trainNewModel():
    # model = unet.UNet()
    # model.to(device)

    model = unet.UNet(features=[64, 128, 256, 512])
    # model.load_state_dict(torch.load("best/model6.pth"))
    model.to(device)

    print(summary(model, (3, 256, 256)))

    training.trainMethod(model, 200, 16, 1e-4, 1e-8, device)

    torch.save(model.state_dict(), "0")


def loadModelAndSubmit():
    foreground_threshold = 0.25

    def patch_to_label(patch):
        df = np.mean(patch) / 256

        if df > foreground_threshold:
            return 1
        else:
            return 0

    model = unet.UNet(features=[64, 128, 256, 512])
    model.load_state_dict(torch.load("best/model6.pth"))
    model.to(device)

    file = 'submission.csv'
    with open(file, 'w') as f:
        f.write('id,prediction\n')

        for index in range(1, 51):
            prediction = predict.predict(model, f"data/test_set_images/test_set_images/test_{index}/test_{index}.png", device)


            # aug = A.CenterCrop(p=1, height=608, width=608)
            # augmented = aug(image=prediction, mask=np.zeros_like(prediction))
            # prediction = augmented['image']
            prediction = np.array(prediction, dtype=np.uint8)
            # prediction = prediction.squeeze(axis=0)
            prediction = prediction.squeeze(axis=1)
            # print(prediction.shape)

            one = cv2.resize(prediction[0, :, :].copy(), (400, 400), cv2.INTER_CUBIC)
            two = cv2.resize(prediction[1, :, :].copy(), (400, 400), cv2.INTER_CUBIC)
            three = cv2.resize(prediction[2, :, :].copy(), (400, 400), cv2.INTER_CUBIC)
            four = cv2.resize(prediction[3, :, :].copy(), (400, 400), cv2.INTER_CUBIC)

            # one = prediction[0, :, :]
            # two = prediction[1, :, :]
            # three = prediction[2, :, :]
            # four = prediction[3, :, :]
            #
            prediction = np.zeros((608, 608))
            prediction[:400, :400] = one
            prediction[208:608, :400] = np.maximum(prediction[208:608, :400], two)
            prediction[:400, 208:608] = np.maximum(prediction[:400, 208:608], three)
            prediction[208:608, 208:608] = np.maximum(prediction[208:608, 208:608], four)

            prediction = np.array(prediction, dtype=np.uint8)
            img = Image.fromarray(prediction)
            img.save(f"segmentations/test_{index}.png")

            forSubmissions = np.zeros_like(prediction, dtype=np.uint8)

            for i in range(0, prediction.shape[1], 16):
                for j in range(0, prediction.shape[0], 16):
                    label = patch_to_label(prediction[j:j+16, i:i+16])
                    f.write("{:03d}_{}_{},{}\n".format(index, i, j, label))
                    forSubmissions[i:i+16, j:j+16] += 255 * label

            Image.fromarray(forSubmissions).save(f"pathes/pred_{index}.jpg")



trainNewModel()
# loadModelAndSubmit()