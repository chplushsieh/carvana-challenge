from functools import partial

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

__all__ = [ 'SmallUnet', 'OriginalUnet', 'BetterUnet', 'UpsamplingUnet' ]

class BaseNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, dropout=0.0, bn=1, activation='relu', filters_base=32):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.filters_base = filters_base
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        # TODO assign hyperparameters to self

        if dropout:
            self.dropout2d = nn.Dropout2d(p=dropout)
        else:
            self.dropout2d = lambda x: x

class SmallUnet(BaseNet):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(self.n_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(32, self.n_classes, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = self.pool(x)
        x1 = F.relu(self.conv3(x1))
        x1 = F.relu(self.conv4(x1))
        x1 = F.relu(self.conv5(x1))
        x1 = self.upsample(x1)
        x = torch.cat([x, x1], 1)
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        return x


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn, activation):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = getattr(F, activation)
        self.bn = nn.BatchNorm2d(out) if bn else None

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x, inplace=True)
        if self.bn is not None:
            x = self.bn(x)
        return x


class UNetDownBlock(nn.Module):
    def __init__(self, in_: int, out: int, *, bn=True, activation='relu'):
        super().__init__()
        self.l1 = Conv3BN(in_, out, bn, activation)
        self.l2 = Conv3BN(out, out, bn, activation)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_: int, out: int, *, bn=True, activation='relu', up='upconv'):
        super().__init__()
        self.l1 = Conv3BN(in_, out, bn, activation)
        self.l2 = Conv3BN(out, out, bn, activation)

        if up == 'upconv':
            self.up = nn.ConvTranspose2d(in_, out, 2, stride=2)
        elif up == 'upsample':
            self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, skip, x):
        up = self.up(x)
        x = torch.cat([up, skip], 1)

        x = self.l1(x)
        x = self.l2(x)

        return x

class OriginalUnet(BaseNet):
    def __init__(self):
        super().__init__()

        self.down1 = UNetDownBlock(self.n_channels,  64)
        self.down2 = UNetDownBlock(             64, 128)
        self.down3 = UNetDownBlock(            128, 256)
        self.down4 = UNetDownBlock(            256, 512)
        self.down5 = UNetDownBlock(            512,1024)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.up4 = UNetUpBlock(1024, 512)
        self.up3 = UNetUpBlock( 512, 256)
        self.up2 = UNetUpBlock( 256, 128)
        self.up1 = UNetUpBlock( 128,  64)

        self.classify = nn.Conv2d(64, self.n_classes, 1)
        return

    def forward(self, x):

        down1 = self.down1(x)
        x = self.pool1(down1)

        down2 = self.down2(x)
        x = self.pool2(down2)

        down3 = self.down3(x)
        x = self.pool3(down3)

        down4 = self.down4(x)
        x = self.pool4(down4)

        down5 = self.down5(x)

        up4 = self.up4(down4, down5)
        up3 = self.up3(down3, up4)
        up2 = self.up2(down2, up3)
        up1 = self.up1(down1, up2)

        out =  self.classify(up1)
        return out # No Sigmoid layer

class BetterUnet(BaseNet): # Improved: add the last Sigmoid layer
    def __init__(self):
        super().__init__()

        self.down1 = UNetDownBlock(self.n_channels,  64)
        self.down2 = UNetDownBlock(             64, 128)
        self.down3 = UNetDownBlock(            128, 256)
        self.down4 = UNetDownBlock(            256, 512)
        self.down5 = UNetDownBlock(            512,1024)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.up4 = UNetUpBlock(1024, 512)
        self.up3 = UNetUpBlock( 512, 256)
        self.up2 = UNetUpBlock( 256, 128)
        self.up1 = UNetUpBlock( 128,  64)

        self.classify = nn.Conv2d(64, self.n_classes, 1)
        return

    def forward(self, x):

        down1 = self.down1(x)
        x = self.pool1(down1)

        down2 = self.down2(x)
        x = self.pool2(down2)

        down3 = self.down3(x)
        x = self.pool3(down3)

        down4 = self.down4(x)
        x = self.pool4(down4)

        down5 = self.down5(x)

        up4 = self.up4(down4, down5)
        up3 = self.up3(down3, up4)
        up2 = self.up2(down2, up3)
        up1 = self.up1(down1, up2)

        out =  self.classify(up1)
        return F.sigmoid(out)

class UpsamplingUnet(BaseNet):
    def __init__(self):
        super().__init__()

        self.down1 = UNetDownBlock(self.n_channels,  64)
        self.down2 = UNetDownBlock(             64, 128)
        self.down3 = UNetDownBlock(            128, 256)
        self.down4 = UNetDownBlock(            256, 512)
        self.down5 = UNetDownBlock(            512,1024)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.up4 = UNetUpBlock(512+1024, 512, up='upsample')
        self.up3 = UNetUpBlock( 256+512, 256, up='upsample')
        self.up2 = UNetUpBlock( 128+256, 128, up='upsample')
        self.up1 = UNetUpBlock(  64+128,  64, up='upsample')

        self.classify = nn.Conv2d(64, self.n_classes, 1)
        return

    def forward(self, x):

        down1 = self.down1(x)
        x = self.pool1(down1)

        down2 = self.down2(x)
        x = self.pool2(down2)

        down3 = self.down3(x)
        x = self.pool3(down3)

        down4 = self.down4(x)
        x = self.pool4(down4)

        down5 = self.down5(x)

        up4 = self.up4(down4, down5)
        up3 = self.up3(down3, up4)
        up2 = self.up2(down2, up3)
        up1 = self.up1(down1, up2)

        out =  self.classify(up1)
        return F.sigmoid(out)
