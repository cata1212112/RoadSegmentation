import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, channelsIn, channelsOut):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channelsIn, channelsOut, 3, padding=1, bias=False),
            nn.BatchNorm2d(channelsOut),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv2d(channelsOut, channelsOut, 3, padding=1, bias=False),
            nn.BatchNorm2d(channelsOut),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(output_channels))

        self.W_x = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(output_channels))

        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(nn.Conv2d(output_channels, 1, kernel_size=1, stride=1, padding=0),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid())


    def forward(self, g, s):
        g = self.W_g(g)
        s = self.W_x(s)

        psi = g + s
        psi = self.relu(psi)
        psi = self.psi(psi)
        return s * psi

class DecoderBlock(nn.Module):
    def __init__(self, features):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.attention = AttentionGate([2*features, features], features)


class AttentionUNet(nn.Module):
    def __init__(self, channelsIn=3, channelsOut=1, features=None):
        super(AttentionUNet, self).__init__()
        if features is None:
            self.features = [64, 128, 256, 512]
        else:
            self.features = features

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)

        for feature in self.features:
            self.encoder.append(ConvBlock(channelsIn, feature))
            channelsIn = feature

        for feature in reversed(self.features):
            self.decoder.append(nn.Sequential(nn.ConvTranspose2d(2 * feature, feature, kernel_size=2, stride=2)))
            self.decoder.append(AttentionGate(feature, feature))
            self.decoder.append(nn.Sequential(ConvBlock(2 * feature, feature)))

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

        for i in range(0, len(self.decoder), 3):
            x = self.decoder[i](x)
            s = self.decoder[i+1](x, skipConnections[i // 3])
            x = torch.cat((x, s), dim=1)
            x = self.decoder[i + 2](x)

        return self.final(x)