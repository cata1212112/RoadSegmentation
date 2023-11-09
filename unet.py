import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms, datasets, models


class ConvBlock(nn.Module):
    def __init__(self, channelsIn, channelsOut):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channelsIn, channelsOut, 3, padding=1, bias=False),
            nn.BatchNorm2d(channelsOut),
            nn.ReLU(inplace=True),

            nn.Conv2d(channelsOut, channelsOut, 3, padding=1, bias=False),
            nn.BatchNorm2d(channelsOut),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, channelsIn=3, channelsOut=1, features=None):
        super(UNet, self).__init__()
        if features is None:
            self.features = [64, 128, 256, 512]
        else:
            self.features = features

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.pool = nn.MaxPool2d(2, 2)

        for feature in self.features:
            self.encoder.append(ConvBlock(channelsIn, feature))
            channelsIn = feature

        for feature in reversed(self.features):
            self.decoder.append(nn.ConvTranspose2d(2 * feature, feature, 2, 2))
            self.decoder.append(ConvBlock(2 * feature, feature))

        self.bottleneck = ConvBlock(self.features[-1], self.features[-1] * 2)

        self.final = nn.Conv2d(self.features[0], channelsOut, 1)

    def forward(self, x):
        skipConnections = []

        for encode in self.encoder:
            x = encode(x)

            skipConnections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skipConnections = skipConnections[::-1]

        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)

            if x.shape != skipConnections[i // 2].shape:
                x = TF.resize(x, size=skipConnections[i // 2].shape[2:])  ## sau padding or mirror de testat

            concatenateSkipConn = torch.cat((skipConnections[i // 2], x), dim=1)
            x = self.decoder[i + 1](concatenateSkipConn)

        return self.final(x)