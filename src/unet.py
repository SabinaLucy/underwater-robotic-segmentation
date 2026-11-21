import torch
import torch.nn as nn


def double_conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    def __init__(self, num_classes=8, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc1 = double_conv(3, c[0])
        self.enc2 = double_conv(c[0], c[1])
        self.enc3 = double_conv(c[1], c[2])
        self.enc4 = double_conv(c[2], c[3])
        self.bottleneck = double_conv(c[3], c[4])
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.dec4 = double_conv(c[3] * 2, c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = double_conv(c[2] * 2, c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = double_conv(c[1] * 2, c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = double_conv(c[0] * 2, c[0])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)