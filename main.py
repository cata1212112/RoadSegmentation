import cv2

import predict
import unet
import torch
import training
import numpy as np
from PIL import Image
import albumentations as A
import dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def trainNewModel():
    # model = unet.UNet()
    # model.to(device)

    model = unet.UNet()
    # model.load_state_dict(torch.load("15"))
    model.to(device)

    training.trainMethod(model, 10, 128, 1e-3, 1e-8, device)

    torch.save(model.state_dict(), "17")


def loadModelAndSubmit():
    foreground_threshold = 0

    def patch_to_label(patch):
        df = np.mean(patch) / 255

        if df > foreground_threshold:
            return 1
        else:
            return 0

    model = unet.UNet()
    model.load_state_dict(torch.load("16"))
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


            img = Image.fromarray(prediction)
            img.save(f"segmentations/test_{index}.png")

            # forSubmissions = np.zeros_like(prediction, dtype=np.uint8)

            for i in range(0, prediction.shape[1], 16):
                for j in range(0, prediction.shape[0], 16):
                    label = patch_to_label(prediction[i:i+16, j:j+16])
                    f.write("{:03d}_{}_{},{}\n".format(index, i, j, label))
                    # forSubmissions[i:i+16, j:j+16] += 255 * label

            # Image.fromarray(forSubmissions).show()



trainNewModel()
# loadModelAndSubmit()