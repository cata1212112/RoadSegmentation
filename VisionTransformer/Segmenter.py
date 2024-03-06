import torch
import torch.nn as nn
import torch.nn.functional as F
from VisionTransformer.VisionTransformer import VisionTransformer, height, width, patch_size, EMBEDDING_SIZE, \
    NUMBER_OF_HEADS, HIDDEN_SIZE, TE_NUM_LAYERS


class LinearDecoder(nn.Module):
    def __init__(self, d_encoder):
        super().__init__()

        self.linear1 = nn.Linear(d_encoder, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, height // patch_size, width // patch_size)
        return x


class Segmenter(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = VisionTransformer()
        self.decoder = LinearDecoder(EMBEDDING_SIZE)

    def forward(self, x):
        x = self.encoder(x)
        x = x[:, 2:]
        masks = self.decoder(x)
        masks = F.interpolate(masks, (100, 100), mode='bilinear')
        return masks
