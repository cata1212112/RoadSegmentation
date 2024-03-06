import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 64, 3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU())

        self.encoder2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, 3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU())

        self.encoder3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU())

        self.encoder4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU())

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        self.decoder1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(),
                                      nn.Conv2d(512, 256, 3, 1, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU())

        self.decoder2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 128, 3, 1, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU())

        self.decoder3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 64, 3, 1, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU())

        self.decoder4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 3, 3, 1, padding=1),
                                      nn.BatchNorm2d(3),
                                      nn.ReLU(),
                                      nn.Conv2d(3, 1, 3, 1, padding=1))

        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, x):
        x = self.encoder1(x)
        x1_size = x.size()
        x, indices1 = self.pool(x)

        x = self.encoder2(x)
        x2_size = x.size()
        x, indices2 = self.pool(x)

        x = self.encoder3(x)
        x3_size = x.size()
        x, indices3 = self.pool(x)

        x = self.encoder4(x)
        x4_size = x.size()
        x, indices4 = self.pool(x)

        x = self.unpool(x, indices4, output_size=x4_size)
        x = self.decoder1(x)

        x = self.unpool(x, indices3, output_size=x3_size)
        x = self.decoder2(x)

        x = self.unpool(x, indices2, output_size=x2_size)
        x = self.decoder3(x)

        x = self.unpool(x, indices1, output_size=x1_size)
        x = self.decoder4(x)

        return x
