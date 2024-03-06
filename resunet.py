import torch.nn as nn
import torch

class ResConvBlock(nn.Module):
    def __init__(self, channelsIn, channelsOut):
        super(ResConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channelsIn, channelsOut, 3, padding=1, bias=False),
            nn.BatchNorm2d(channelsOut),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv2d(channelsOut, channelsOut, 3, padding=1, bias=False),
            nn.BatchNorm2d(channelsOut),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(channelsIn, channelsOut, kernel_size=3, padding=1),
            nn.BatchNorm2d(channelsOut),
        )

    def forward(self, x):
        return self.conv(x) + self.conv_skip(x)


class ResUNet(nn.Module):
    def __init__(self, channelsIn=3, channelsOut=1, features=None):
        super(ResUNet, self).__init__()
        if features is None:
            self.features = [64, 128, 256, 512]
        else:
            self.features = features

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)

        for feature in self.features:
            self.encoder.append(ResConvBlock(channelsIn, feature))
            channelsIn = feature

        for feature in reversed(self.features):
            self.decoder.append(nn.Sequential(nn.ConvTranspose2d(2 * feature, feature, kernel_size=2, stride=2)))
            self.decoder.append(nn.Sequential(ResConvBlock(2 * feature, feature)))

        self.bottleneck = ResConvBlock(self.features[-1], self.features[-1] * 2)

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
            concatenateSkipConn = torch.cat((skipConnections[i // 2], x), dim=1)
            x = self.decoder[i + 1](concatenateSkipConn)

        return self.final(x)