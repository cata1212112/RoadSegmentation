import predict
import unet
import torch
import training
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def trainNewModel():
    model = unet.UNet()
    model.to(device)
    training.trainMethod(model, 50, 8, 1e-3, 1e-8, device)

    torch.save(model.state_dict(), "1")

    prediction = predict.predict(model, "data/test_set_images/test_set_images/test_1", "test_1.png", device)
    prediction = np.array(prediction * 255, dtype=np.uint8)
    Image.fromarray(prediction).show()

def loadModel():
    model = unet.UNet()
    model.load_state_dict(torch.load("1"))