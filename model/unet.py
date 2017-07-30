from functools import partial

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

__all__ = [ 'SmallUNet', 'SimpleSegNet', 'UNet', 'UNet3l', 'UNet2', 'InceptionUNet', 'Inception2UNet', 'DenseUNet', 'DenseNet']

class BaseNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, dropout=0.0, bn=1, activation='relu', filters_base=32):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.filters_base = filters_base
        self.bn = bn
        self.activation = activation
        self.dropout=dropout
        # TODO assign hyperparameters to self

        if dropout:
            self.dropout2d = nn.Dropout2d(p=dropout)
        else:
            self.dropout2d = lambda x: x


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


def concat(xs):
    return torch.cat(xs, 1)


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

class SmallUNet(BaseNet):
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


class SimpleSegNet(BaseNet):
    def __init__(self):
        super().__init__()

        s = self.filters_base

        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.input_conv = BasicConv2d(self.n_channels, s, 1)
        self.enc_1 = BasicConv2d(s * 1, s * 2, 3, padding=1)
        self.enc_2 = BasicConv2d(s * 2, s * 4, 3, padding=1)
        self.enc_3 = BasicConv2d(s * 4, s * 8, 3, padding=1)
        self.enc_4 = BasicConv2d(s * 8, s * 8, 3, padding=1)
        # https://github.com/pradyu1993/segnet - decoder lacks relu (???)
        self.dec_4 = BasicConv2d(s * 8, s * 8, 3, padding=1)
        self.dec_3 = BasicConv2d(s * 8, s * 4, 3, padding=1)
        self.dec_2 = BasicConv2d(s * 4, s * 2, 3, padding=1)
        self.dec_1 = BasicConv2d(s * 2, s * 1, 3, padding=1)
        self.conv_final = nn.Conv2d(s, self.n_classes, 1)

    def forward(self, x):
        # Input
        x = self.input_conv(x)
        # Encoder
        x = self.enc_1(x)
        x = self.pool(x)
        x = self.enc_2(x)
        x = self.pool(x)
        x = self.enc_3(x)
        x = self.pool(x)
        x = self.enc_4(x)

        # Decoder
        x = self.dec_4(x)
        x = self.upsample(x)
        x = self.dec_3(x)
        x = self.upsample(x)
        x = self.dec_2(x)
        x = self.upsample(x)
        x = self.dec_1(x)

        # Output
        x = self.conv_final(x)
        return x


class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int, *, dropout, bn, activation): # TODO dropout not used
        super().__init__()
        self.conv1 = conv3x3(in_, out)
        self.conv2 = conv3x3(out, out)
        self.bn = bn
        self.activation = getattr(F, activation)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(out)
            self.bn2 = nn.BatchNorm2d(out)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        x = self.activation(x)
        return x


class UNet(BaseNet):
    module = UNetModule

    def __init__(self, top_scale=2, filter_factors=[1, 2, 4, 8, 16]):
        super().__init__()
        self.top_scale = top_scale
        self.filter_factors = filter_factors

        self.pool = nn.MaxPool2d(2, 2)
        self.pool_top = nn.MaxPool2d(self.top_scale, self.top_scale)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample_top = nn.UpsamplingNearest2d(scale_factor=self.top_scale)

        filter_sizes = [self.filters_base * s for s in self.filter_factors]
        self.down, self.up = [], []
        for i, nf in enumerate(filter_sizes):
            low_nf = self.n_channels if i == 0 else filter_sizes[i - 1]
            self.down.append(self.module(low_nf, nf, dropout=self.dropout, bn=self.bn, activation=self.activation))
            setattr(self, 'down_{}'.format(i), self.down[-1])
            if i != 0:
                self.up.append(self.module(low_nf + nf, low_nf, dropout=self.dropout, bn=self.bn, activation=self.activation))
                setattr(self, 'conv_up_{}'.format(i), self.up[-1])
        self.conv_final = nn.Conv2d(filter_sizes[0], self.n_classes, 1)

    def forward(self, x):
        xs = []
        for i, down in enumerate(self.down):
            if i == 0:
                x_in = x
            elif i == 1:
                x_in = self.pool_top(xs[-1])
            else:
                x_in = self.pool(xs[-1])
            x_out = down(x_in)
            x_out = self.dropout2d(x_out)
            xs.append(x_out)

        x_out = xs[-1]
        for i, (x_skip, up) in reversed(list(enumerate(zip(xs[:-1], self.up)))):
            upsample = self.upsample_top if i == 0 else self.upsample
            x_out = up(torch.cat([upsample(x_out), x_skip], 1))
            x_out = self.dropout2d(x_out)

        x_out = self.conv_final(x_out)
        return x_out


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn, activation):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = getattr(F, activation)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x, inplace=True)
        return x


class UNet3lModule(nn.Module):
    def __init__(self, in_: int, out: int, *, dropout, bn, activation):
        super().__init__()
        self.l1 = Conv3BN(in_, out, bn, activation)
        self.l2 = Conv3BN(out, out, bn, activation)
        self.l3 = Conv3BN(out, out, bn, activation)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


class UNet3l(UNet):
    module = UNet3lModule

class UNet2Module(nn.Module):
    def __init__(self, in_: int, out: int, *, dropout, bn, activation):  # TODO dropout not used
        super().__init__()
        self.l1 = Conv3BN(in_, out, bn, activation)
        self.l2 = Conv3BN(out, out, bn, activation)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class UNet2(BaseNet):
    def __init__(self):
        super().__init__()
        b = self.filters_base
        self.filters = [b * 2, b * 2, b * 4, b * 8, b * 16]
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.down, self.down_pool, self.mid, self.up = [[] for _ in range(4)]
        for i, nf in enumerate(self.filters):
            low_nf = self.n_channels if i == 0 else self.filters[i - 1]
            self.down_pool.append(
                nn.Conv2d(low_nf, low_nf, 3, padding=1, stride=2))
            setattr(self, 'down_pool_{}'.format(i), self.down_pool[-1])
            self.down.append(UNet2Module(low_nf, nf, bn=self.bn, activation=self.activation))
            setattr(self, 'down_{}'.format(i), self.down[-1])
            if i != 0:
                self.mid.append(Conv3BN(low_nf, low_nf, self.bn, self.activation))
                setattr(self, 'mid_{}'.format(i), self.mid[-1])
                self.up.append(UNet2Module(low_nf + nf, low_nf, bn=self.bn, activation=self.activation))
                setattr(self, 'up_{}'.format(i), self.up[-1])
        self.conv_final = nn.Conv2d(self.filters[0], self.n_classes, 1)

    def forward(self, x):
        xs = []
        for i, (down, down_pool) in enumerate(zip(self.down, self.down_pool)):
            x_out = down(down_pool(xs[-1]) if xs else x)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, up, mid in reversed(list(zip(xs[:-1], self.up, self.mid))):
            x_out = up(torch.cat([self.upsample(x_out), mid(x_skip)], 1))

        x_out = self.conv_final(x_out)
        return x_out


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
    def __init__(self, in_: int, out: int, dropout, bn, activation):  # TODO dropout, bn, activation not used
        super().__init__()
        out_1 = out * 3 // 8
        out_2 = out * 2 // 8
        self.conv1x1 = BasicConv2d(in_, out_1, kernel_size=1)
        self.conv3x3_pre = BasicConv2d(in_, in_ // 2, kernel_size=1)
        self.conv3x3 = BasicConv2d(in_ // 2, out_1, kernel_size=3, padding=1)
        self.conv5x5_pre = BasicConv2d(in_, in_ // 2, kernel_size=1)           # Since n_channel // 4 == 0
        self.conv5x5 = BasicConv2d(in_ // 2, out_2, kernel_size=5, padding=2)  # I have to change it to in_// 2

    def forward(self, x):
        return torch.cat([
            self.conv1x1(x),
            self.conv3x3(self.conv3x3_pre(x)),
            self.conv5x5(self.conv5x5_pre(x)),
        ], 1)


class Inception2Module(nn.Module):
    def __init__(self, in_: int, out: int, dropout, bn, activation): # TODO dropout, bn, activation not used
        super().__init__()
        self.l1 = InceptionModule(in_, out, dropout, bn, activation)
        self.l2 = InceptionModule(out, out, dropout, bn, activation)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class InceptionUNet(UNet):
    module = InceptionModule


class Inception2UNet(UNet):
    module = Inception2Module


class DenseLayer(nn.Module):
    def __init__(self, in_, out, *, dropout, bn):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_) if bn else None
        self.activation = nn.ReLU(inplace=True)
        self.conv = conv3x3(in_, out)
        self.dropout = nn.Dropout2d(p=dropout) if dropout else None

    def forward(self, x):
        x = self.activation(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_, k, n_layers, dropout, bn):
        super().__init__()
        self.out = k * n_layers
        layer_in = in_
        self.layers = []
        for i in range(n_layers):
            layer = DenseLayer(layer_in, k, dropout=dropout, bn=bn)
            self.layers.append(layer)
            setattr(self, 'layer_{}'.format(i), layer)
            layer_in += k

    def forward(self, x):
        inputs = [x]
        outputs = []
        for i, layer in enumerate(self.layers[:-1]):
            outputs.append(layer(inputs[i]))
            inputs.append(concat([outputs[i], inputs[i]]))
        return torch.cat([self.layers[-1](inputs[-1])] + outputs, 1)


class DenseUNetModule(DenseBlock):
    def __init__(self, in_: int, out: int, *, dropout, bn, activation):  # TODO activation not used
        n_layers = 4
        super().__init__(in_, out // n_layers, n_layers,
                         dropout=dropout, bn=bn)


class DenseUNet(UNet):
    module = DenseUNetModule


class DownBlock(nn.Module):
    def __init__(self, in_, out, scale, *, dropout, bn):
        super().__init__()
        self.in_ = in_
        self.bn = nn.BatchNorm2d(in_) if bn else None
        self.activation = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_, out, 1)
        self.dropout = nn.Dropout2d(p=dropout) if dropout else None
        self.pool = nn.MaxPool2d(scale, scale)

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_, out, scale):
        super().__init__()
        self.up_conv = nn.Conv2d(in_, out, 1)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=scale)

    def forward(self, x):
        return self.upsample(self.up_conv(x))


class DenseNet(BaseNet):
    """ https://arxiv.org/pdf/1611.09326v2.pdf
    """
    def __init__(self):
        super().__init__()
        k = self.filters_base
        block_layers = [3, 5, 7, 5, 3]
        block_in = [n * k for n in [3, 8, 16, 8, 4]]
        scale_factors = [4, 2]
        dense = partial(DenseBlock, dropout=self.dropout, bn=self.bn)
        self.input_conv = nn.Conv2d(self.n_channels, block_in[0], 3, padding=1)
        self.blocks = []
        self.scales = []
        self.n_layers = len(block_layers) // 2
        for i, (in_, l) in enumerate(zip(block_in, block_layers)):
            if i < self.n_layers:
                block = dense(in_, k, l)
                scale = DownBlock(block.out + in_, block_in[i + 1],
                                  scale_factors[i],
                                  dropout=self.dropout, bn=self.bn)
            elif i == self.n_layers:
                block = dense(in_, k, l)
                scale = None
            else:
                block = dense(in_ + self.scales[2 * self.n_layers - i].in_,
                              k, l)
                scale = UpBlock(self.blocks[-1].out, in_,
                                scale_factors[2 * self.n_layers - i])
            setattr(self, 'block_{}'.format(i), block)
            setattr(self, 'scale_{}'.format(i), scale)
            self.blocks.append(block)
            self.scales.append(scale)
        self.output_conv = nn.Conv2d(self.blocks[-1].out, self.n_classes, 1)

    def forward(self, x):
        # Input
        x = self.input_conv(x)
        # Network
        skips = []
        for i, (block, scale) in enumerate(zip(self.blocks, self.scales)):
            if i < self.n_layers:
                x = concat([block(x), x])
                skips.append(x)
                x = scale(x)
            elif i == self.n_layers:
                x = block(x)
            else:
                x = block(concat([scale(x), skips[2 * self.n_layers - i]]))
        # Output
        x = self.output_conv(x)
        return x
