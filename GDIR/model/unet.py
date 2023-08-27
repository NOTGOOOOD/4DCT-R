import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes, norm=False, flag_512=False):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.flag_512 = flag_512
        super(UNet3D, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, bias=False, batchnorm=norm)
        self.ec1 = self.encoder(32, 64, bias=False, batchnorm=norm)
        self.ec2 = self.encoder(64, 64, bias=False, batchnorm=norm)
        self.ec3 = self.encoder(64, 128, bias=False, batchnorm=norm)
        self.ec4 = self.encoder(128, 128, bias=False, batchnorm=norm)
        self.ec5 = self.encoder(128, 256, bias=False, batchnorm=norm)

        if self.flag_512:
            self.ec6 = self.encoder(256, 256, bias=False, batchnorm=norm)
            self.ec7 = self.encoder(256, 512, bias=False, batchnorm=norm)
            self.dc7 = self.decoder(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
            self.dc6 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.dc5 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.decoder(64, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.flow = nn.Conv3d(16, 3, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels),
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
        syn0 = self.ec1(self.ec0(x))
        syn1 = self.ec3(self.ec2(F.interpolate(syn0, scale_factor=0.5, mode='trilinear',
                           align_corners=True, recompute_scale_factor=False)))
        syn2 = self.ec5(self.ec4(F.interpolate(syn1, scale_factor=0.5, mode='trilinear',
                           align_corners=True, recompute_scale_factor=False)))

        if self.flag_512:
            syn3 = self.ec7(self.ec6(F.interpolate(syn2, scale_factor=0.5, mode='trilinear',
                               align_corners=True, recompute_scale_factor=False)))

            d7 = self.dc7(syn3)
            if d7.shape[2:] != syn2.shape[2:]:
                d7 = F.interpolate(d7, syn2.shape[2:], mode='trilinear', align_corners=True, recompute_scale_factor=False)

            d7 = torch.cat((d7, syn2), 1)
            d6 = self.dc6(d7)
            del d7
            d5 = F.interpolate(d6, syn1.shape[2:], mode='trilinear', align_corners=True, recompute_scale_factor=False)
        else:
            d5 = F.interpolate(syn2, syn1.shape[2:], mode='trilinear', align_corners=True, recompute_scale_factor=False)

        d5 = self.dc5(d5)
        d5 = torch.cat((d5, syn1), 1)
        del syn1

        d4 = self.dc4(d5)
        del d5

        if d4.shape[2:] != syn0.shape[2:]:
            d4 = F.interpolate(d4, syn0.shape[2:], mode='trilinear', align_corners=True, recompute_scale_factor=False)

        d3 = torch.cat((self.dc3(d4), syn0), 1)
        del syn0, d4
        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return self.flow(d0)


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
