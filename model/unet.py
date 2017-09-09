'''
Modified from:
https://github.com/lopuhin/kaggle-dstl/blob/master/models.py
'''

from functools import partial

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F




class BaseNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, dropout=0.0, bn=1, activation='relu'):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bn = bn
        self.activation = activation
        self.dropout = dropout

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
        return F.sigmoid(x)


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

#---------------

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_: int, out: int, bn, activation):
        super().__init__()
        out_1 = out * 3 // 8
        out_2 = out * 2 // 8
        self.conv1x1 = BasicConv2d(in_, out_1, kernel_size=1)
        self.conv3x3_pre = BasicConv2d(in_, in_ // 2, kernel_size=1)
        self.conv3x3 = BasicConv2d(in_ // 2, out_1, kernel_size=3, padding=1)
        self.conv5x5_pre = BasicConv2d(in_, in_ // 2, kernel_size=1)
        self.conv5x5 = BasicConv2d(in_ // 2, out_2, kernel_size=5, padding=2)
        #assert hps.bn
        #assert hps.activation == 'relu'

    def forward(self, x):
        return torch.cat([
            self.conv1x1(x),
            self.conv3x3(self.conv3x3_pre(x)),
            self.conv5x5(self.conv5x5_pre(x)),
        ], 1)


#---------------

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
            self.up = nn.Upsample(scale_factor=2)

    def forward(self, skip, x):
        up = self.up(x)
        x = torch.cat([up, skip], 1)

        x = self.l1(x)
        x = self.l2(x)

        return x


class UNetDownBlock3(nn.Module):
    def __init__(self, in_: int, out: int, *, bn=True, activation='relu'):
        super().__init__()
        self.l1 = Conv3BN(in_, out, bn, activation)
        self.l2 = Conv3BN(out, out, bn, activation)
        self.l3 = Conv3BN(out, out, bn, activation)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

class UNetUpBlock3(nn.Module):
    def __init__(self, in_: int, out: int, *, bn=True, activation='relu', up='upconv'):
        super().__init__()
        self.l1 = Conv3BN(in_, out, bn, activation)
        self.l2 = Conv3BN(out, out, bn, activation)
        self.l3 = Conv3BN(out, out, bn, activation)

        if up == 'upconv':
            self.up = nn.ConvTranspose2d(in_, out, 2, stride=2)
        elif up == 'upsample':
            self.up = nn.Upsample(scale_factor=2)

    def forward(self, skip, x):
        up = self.up(x)
        x = torch.cat([up, skip], 1)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x


class UNetDownBlock4(nn.Module):
    def __init__(self, in_: int, out: int, *, bn=True, activation='relu'):
        super().__init__()
        self.l1 = Conv3BN(in_, out, bn, activation)
        self.l2 = Conv3BN(out, out, bn, activation)
        self.l3 = Conv3BN(out, out, bn, activation)
        self.l4 = Conv3BN(out, out, bn, activation)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x

class UNetUpBlock4(nn.Module):
    def __init__(self, in_: int, out: int, *, bn=True, activation='relu', up='upconv'):
        super().__init__()
        self.l1 = Conv3BN(in_, out, bn, activation)
        self.l2 = Conv3BN(out, out, bn, activation)
        self.l3 = Conv3BN(out, out, bn, activation)
        self.l4 = Conv3BN(out, out, bn, activation)

        if up == 'upconv':
            self.up = nn.ConvTranspose2d(in_, out, 2, stride=2)
        elif up == 'upsample':
            self.up = nn.Upsample(scale_factor=2)

    def forward(self, skip, x):
        up = self.up(x)
        x = torch.cat([up, skip], 1)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        return x

class InceptiondDownModule(nn.Module):
    def __init__(self, in_: int, out: int, *, bn=True, activation='relu'):
        super().__init__()
        self.l1 = InceptionModule(in_, out, bn, activation)

    def forward(self, x):
        x = self.l1(x)

        return x


class InceptiondUpModule(nn.Module):
    def __init__(self, in_: int, out: int, *, bn=True, activation='relu', up='upconv'):
        super().__init__()
        self.l1 = InceptionModule(in_, out, bn, activation)

        if up == 'upconv':
            self.up = nn.ConvTranspose2d(in_, out, 2, stride=2)
        elif up == 'upsample':
            self.up = nn.Upsample(scale_factor=2)


    def forward(self, skip, x):
        up = self.up(x)
        x = torch.cat([up, skip], 1)

        x = self.l1(x)

        return x







class InceptiondDownModule2(nn.Module):
    def __init__(self, in_: int, out: int, *, bn=True, activation='relu'):
        super().__init__()
        self.l1 = InceptionModule(in_, out, bn, activation)
        self.l2 = InceptionModule(out, out, bn, activation)

    # self.l3 = InceptionModule(out, out, bn, activation)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        # x = self.l3(x)
        return x


class InceptiondUpModule2(nn.Module):
    def __init__(self, in_: int, out: int, *, bn=True, activation='relu', up='upconv'):
        super().__init__()
        self.l1 = InceptionModule(in_, out, bn, activation)
        self.l2 = InceptionModule(out, out, bn, activation)
        # self.l3 = InceptionModule(out, out, bn, activation)

        if up == 'upconv':
            self.up = nn.ConvTranspose2d(in_, out, 2, stride=2)
        elif up == 'upsample':
            self.up = nn.Upsample(scale_factor=2)


    def forward(self, skip, x):
        up = self.up(x)
        x = torch.cat([up, skip], 1)

        x = self.l1(x)
        x = self.l2(x)
        # x = self.l3(x)

        return x



class Unet(BaseNet): # Improved: add the last Sigmoid layer
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

class DynamicUnet(BaseNet):
    def __init__(self, DownBlock=UNetDownBlock, UpBlock=UNetUpBlock, nums_filters = [64, 128, 256, 512, 1024], dropout=0.0):
        super().__init__(dropout=dropout)

        self.down = nn.ModuleList([ DownBlock(self.n_channels,  nums_filters[0]) ])
        for i in range(len(nums_filters)-1):
            self.down.append(DownBlock(nums_filters[i],  nums_filters[i+1]))

        self.pool = nn.ModuleList([ nn.MaxPool2d(2) for i in range(len(nums_filters) - 1) ])

        self.up = nn.ModuleList([])
        for i in range(len(nums_filters)-1):
            self.up.append(UpBlock(nums_filters[i] + nums_filters[i+1], nums_filters[i],  up='upsample'))

        self.classify = nn.Conv2d(nums_filters[0], self.n_classes, 1)
        return

    def forward(self, x):

        down_outputs = []
        for i in range(len(self.down)):
            down_output = self.down[i](x)
            down_output = self.dropout2d(down_output)
            down_outputs.append(down_output)

            if i < len(self.pool):
                x = self.pool[i](down_output)

        x = down_outputs[-1]
        for i in reversed(range(len(self.up))):
            x = self.up[i](down_outputs[i], x)
            x = self.dropout2d(x)

        out =  self.classify(x)
        return F.sigmoid(out)

def AndresUnet():
    return DynamicUnet(nums_filters = [32, 64, 128, 256, 512])

def AndresUnet_without_bn():
    UNetDownBlock_without_bn = lambda x, y: UNetDownBlock(x, y, bn=False)
    UNetUpBlock_without_bn   = lambda x, y: UNetUpBlock(x, y, bn=False)
    return DynamicUnet(DownBlock=UNetDownBlock_without_bn, UpBlock=UNetUpBlock_without_bn, nums_filters = [32, 64, 128, 256, 512])

def PeterUnet():
    '''
    https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/blob/master/model/u_net.py#L404
    '''
    return DynamicUnet(nums_filters = [8, 16, 32, 64, 128, 256, 512, 1024])

def PeterUnet3():
    return DynamicUnet(DownBlock=UNetDownBlock3, UpBlock=UNetUpBlock3, nums_filters = [8, 16, 32, 64, 128, 256, 512, 1024])

def PeterUnet3_dropout():
    return DynamicUnet(DownBlock=UNetDownBlock3, UpBlock=UNetUpBlock3, nums_filters = [8, 16, 32, 64, 128, 256, 512, 1024], dropout=0.5)

def PeterUnet4():
    return DynamicUnet(DownBlock=UNetDownBlock4, UpBlock=UNetUpBlock4, nums_filters = [8, 16, 32, 64, 128, 256, 512, 1024])

def PeterUnet34():
    return DynamicUnet(DownBlock=UNetDownBlock3, UpBlock=UNetUpBlock4, nums_filters = [8, 16, 32, 64, 128, 256, 512, 1024])

def PeterUnetInception2():
	return DynamicUnet(DownBlock=InceptiondDownModule2, UpBlock=InceptiondUpModule2, nums_filters = [8, 16, 32, 64, 128, 256, 512, 1024])

def PeterUnetInception():
	return DynamicUnet(DownBlock=InceptiondDownModule, UpBlock=InceptiondUpModule, nums_filters = [8, 16, 32, 64, 128, 256, 512, 1024])

class SmallerUpsamplingUnet(BaseNet):
    def __init__(self):
        super().__init__()

        self.down1 = UNetDownBlock(self.n_channels,  64)
        self.down2 = UNetDownBlock(             64, 128)
        self.down3 = UNetDownBlock(            128, 256)
        self.down4 = UNetDownBlock(            256, 512)
        self.down5 = UNetDownBlock(            512, 512)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.up4 = UNetUpBlock(1024, 256, up='upsample')
        self.up3 = UNetUpBlock( 512, 128, up='upsample')
        self.up2 = UNetUpBlock( 256,  64, up='upsample')
        self.up1 = UNetUpBlock(  128, 32, up='upsample')

        self.classify = nn.Conv2d(32, self.n_classes, 1)
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


class DenseLayer(nn.Module):
    def __init__(self, in_, out, *, bn, activation):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_) if bn else None
        self.activation = getattr(F, activation)
        self.conv = conv3x3(in_, out)

    def forward(self, x):
        x = self.activation(x, inplace=True)
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_, k, n_layers, bn=False, activation='relu'):
        super().__init__()
        self.out = k * n_layers
        layer_in = in_
        self.layers = []
        for i in range(n_layers):
            layer = DenseLayer(layer_in, k, bn=bn, activation=activation)
            self.layers.append(layer)
            setattr(self, 'layer_{}'.format(i), layer)
            layer_in += k

    def forward(self, x):
        inputs = [x]
        outputs = []
        for i, layer in enumerate(self.layers[:-1]):
            outputs.append(layer(inputs[i]))
            inputs.append(torch.cat([outputs[i], inputs[i]], 1))
        return torch.cat([self.layers[-1](inputs[-1])] + outputs, 1)


class DenseUNetModule(DenseBlock):
    def __init__(self, in_: int, out: int, *, bn=True, activation='relu'):
        n_layers = 4
        super().__init__(in_, out // n_layers, n_layers,
                         bn, activation)

class DenseDownBlock(nn.Module):
    def __init__(self, in_: int, out: int, *, bn=True, activation='relu'):
        super().__init__()
        self.l1 = DenseUNetModule(in_, out, bn=bn, activation=activation)

    def forward(self, x):
        x = self.l1(x)
        return x

class DenseUpBlock(nn.Module):
    def __init__(self, in_: int, out: int, *, bn=True, activation='relu', up='upsample'):
        super().__init__()
        self.l1 = DenseUNetModule(in_, out, bn=bn, activation=activation)

        if up == 'upconv':
            self.up = nn.ConvTranspose2d(in_, out, 2, stride=2)
        elif up == 'upsample':
            self.up = nn.Upsample(scale_factor=2)

    def forward(self, skip, x):
        up = self.up(x)
        x = torch.cat([up, skip], 1)

        x = self.l1(x)
        return x


def DenseUnet():
    return DynamicUnet(DownBlock=DenseDownBlock, UpBlock=DenseUpBlock, nums_filters = [8, 16, 32, 64, 128, 256, 512, 1024])
