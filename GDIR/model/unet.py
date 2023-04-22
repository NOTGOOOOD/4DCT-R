import torch
from torch import nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, bias=False, batchnorm=False)
        self.ec1 = self.encoder(32, 64, bias=False, batchnorm=False)
        self.ec2 = self.encoder(64, 64, bias=False, batchnorm=False)
        self.ec3 = self.encoder(64, 128, bias=False, batchnorm=False)
        self.ec4 = self.encoder(128, 128, bias=False, batchnorm=False)
        self.ec5 = self.encoder(128, 256, bias=False, batchnorm=False)
        self.ec6 = self.encoder(256, 256, bias=False, batchnorm=False)
        self.ec7 = self.encoder(256, 512, bias=False, batchnorm=False)

        self.pool0 = nn.AvgPool3d(2)
        self.pool1 = nn.AvgPool3d(2)
        self.pool2 = nn.AvgPool3d(2)

        self.dc9 = self.decoder(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.decoder(64, n_classes, kernel_size=3, stride=1, padding=1, bias=False)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.LeakyReLU(0.2))
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        d9 = self.dc9(e7)
        if d9.shape[2:] != syn2.shape[2:]:
            d9 = F.interpolate(d9, syn2.shape[2:], mode='trilinear', align_corners=True, recompute_scale_factor=False)

        d9 = torch.cat((d9, syn2), 1)
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = self.dc6(d7)
        if d6.shape[2:] != syn1.shape[2:]:
            d6 = F.interpolate(d6, syn1.shape[2:], mode='trilinear', align_corners=True, recompute_scale_factor=False)

        d6 = torch.cat((d6, syn1), 1)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = self.dc3(d4)
        if d3.shape[2:] != syn0.shape[2:]:
            d3 = F.interpolate(d3, syn0.shape[2:], mode='trilinear', align_corners=True, recompute_scale_factor=False)

        d3 = torch.cat((d3, syn0), 1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0


class UNet(nn.Module):
    '''
    U-net implementation with modifications.
        1. Works for input of 2D or 3D
        2. Change batch normalization to instance normalization

    Adapted from https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

    Parameters
    ----------
    in_channels : int
        number of input channels.
    out_channels : int
        number of output channels.
    dim : (2 or 3), optional
        The dimention of input data. The default is 2.
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : TYPE, optional
        Number of initial channels. The default is 32.
    normalization : bool, optional
        Whether to add instance normalization after activation. The default is False.

        encoder
DownBlock(
  (block): Sequential(
    (0): Conv3d(2, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (1): InstanceNorm3d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2)
  )
)
通过Conv3d提取特征 增加通道
利用插值方式下采样 0.5


decoder
UpBlock(
  (conv): Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (conv_block): ConvBlock(
    (block): Sequential(
      (0): Conv3d(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (1): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (2): LeakyReLU(negative_slope=0.2)
    )
  )
)
插值恢复到下采样最后一层的维度
利用nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)) 恢复通道数
跳跃连接
再通过ConvBlock提取特征

    '''

    def __init__(self, in_channels, out_channels, dim=2, depth=5, initial_channels=32, normalization=True):

        super().__init__()
        assert dim in (2, 3)
        self.dim = dim

        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(self.depth):
            current_channels = 2 ** i * initial_channels
            self.down_path.append(ConvBlock(prev_channels, current_channels, dim, normalization))
            prev_channels = current_channels

        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            current_channels = 2 ** i * initial_channels
            # print(prev_channels, current_channels)
            self.up_path.append(UpBlock(prev_channels, current_channels, dim, normalization))
            prev_channels = current_channels

        if dim == 2:
            self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=(1, 1))
        elif dim == 3:
            self.last = nn.Conv3d(prev_channels, out_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i < self.depth - 1:
                blocks.append(x)
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear' if self.dim == 2 else 'trilinear',
                                  align_corners=True, recompute_scale_factor=False)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim, normalization, LeakyReLU_slope=0.2):
        super().__init__()
        block = []
        if dim == 2:
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))
            if normalization:
                block.append(nn.InstanceNorm2d(out_channels))
            block.append(nn.LeakyReLU(LeakyReLU_slope))
        elif dim == 3:
            block.append(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1))
            if normalization:
                block.append(nn.InstanceNorm3d(out_channels))
            block.append(nn.LeakyReLU(LeakyReLU_slope))
        else:
            raise (f'dim should be 2 or 3, got {dim}')
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim, normalization):
        super().__init__()
        self.dim = dim
        if dim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        elif dim == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        self.conv_block = ConvBlock(in_channels, out_channels, dim, normalization)

    def forward(self, x, skip):
        x_up = F.interpolate(x, skip.shape[2:], mode='bilinear' if self.dim == 2 else 'trilinear', align_corners=True)
        x_up_conv = self.conv(x_up)
        out = torch.cat([x_up_conv, skip], 1)
        out = self.conv_block(out)
        return out
