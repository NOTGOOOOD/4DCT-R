import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.Functions import AdaptiveSpatialTransformer
from utils.correlation_layer import CorrTorch


def resblock_seq(in_channels, bias_opt=False):
    layer = nn.Sequential(
        PreActBlock(in_channels, in_channels, bias=bias_opt),
        nn.LeakyReLU(0.2),
        PreActBlock(in_channels, in_channels, bias=bias_opt),
        nn.LeakyReLU(0.2),
        PreActBlock(in_channels, in_channels, bias=bias_opt),
        nn.LeakyReLU(0.2),
        PreActBlock(in_channels, in_channels, bias=bias_opt),
        nn.LeakyReLU(0.2),
        PreActBlock(in_channels, in_channels, bias=bias_opt),
        nn.LeakyReLU(0.2)
    )
    return layer


def input_feature_extract(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                          bias=False, batchnorm=False):
    if batchnorm:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
    else:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
    return layer


def outputs(in_channels, out_channels, kernel_size=3, stride=1, padding=0,
            bias=False, batchnorm=False):
    if batchnorm:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.Tanh())
    else:
        layer = nn.Sequential(
            # nn.Conv3d(in_channels, int(in_channels / 2), kernel_size, stride=stride, padding=padding, bias=bias),
            # nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.Softsign())
    return layer


class Model_lv1_signle(nn.Module):

    def __init__(self, in_channel, n_classes, start_channel, is_train=True, range_flow=0.4, grid=None, corr=False):
        super(Model_lv1_signle, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.grid_1 = grid
        self.transform = AdaptiveSpatialTransformer()
        self.correlation = CorrTorch() if corr else None

        bias_opt = False
        self.dialation_num = 3
        self.dialation_conv0 = nn.ModuleList([nn.Conv3d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=2**i,  dilation=2**i) for i in range(self.dialation_num)])

        self.dialation_conv1 = nn.ModuleList([nn.Conv3d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=2**i, dilation=2**i) for i in range(self.dialation_num)])

        self.input_encoder = input_feature_extract(self.in_channel, self.in_channel * self.dialation_num, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.in_channel * self.dialation_num * 2+27, self.start_channel * 8, kernel_size=3, stride=2,padding=1,bias=bias_opt) if corr else nn.Conv3d(self.in_channel * self.dialation_num * 2, self.start_channel * 8, kernel_size=3, stride=2,padding=1,bias=bias_opt)

        self.resblock_group = resblock_seq(self.start_channel * 8, bias_opt=bias_opt)

        self.up = nn.ConvTranspose3d(self.start_channel * 8, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.decoder = nn.Sequential(
            nn.Conv3d(self.start_channel * 4 + self.in_channel * self.dialation_num * 2+27, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1)) if corr else nn.Sequential(
            nn.Conv3d(self.start_channel * 4 + self.in_channel * self.dialation_num * 2, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1))

        self.conv_block = nn.Sequential(
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2))

        self.output = outputs(self.start_channel * 4, self.n_classes, kernel_size=3, stride=1, padding=1,
                              bias=False)

    def path_one(self, x):
        dialation_res = []
        for dialation in self.dialation_conv0:
            res = dialation(x)
            dialation_res.append(res)

        dialation_out1 = torch.sum(torch.stack(dialation_res),0)
        return self.input_encoder(dialation_out1)

    def path_two(self, x):
        dialation_res = []
        for dialation in self.dialation_conv1:
            res = dialation(x)
            dialation_res.append(res)

        dialation_out2 = torch.sum(torch.stack(dialation_res),0)
        return self.input_encoder(dialation_out2)

    def ds_encoder_path(self, moving, fixed):
        fea_moving = self.path_one(moving)
        fea_fixed = self.path_two(fixed)
        if self.correlation is not None:
            fea_correlation = self.correlation(fea_moving, fea_fixed)
            return torch.cat((fea_moving, fea_fixed, fea_correlation), 1)
        else:
            return torch.cat((fea_moving, fea_fixed), 1)

    def decoder_path(self, fea_encoder):
        e0 = self.down_conv(fea_encoder)
        e0 = self.resblock_group(e0)
        e0 = self.up(e0)

        if e0.shape[2:] != fea_encoder.shape[2:]:
            e0 = F.interpolate(e0, size=fea_encoder.shape[2:],
                               mode='trilinear',
                               align_corners=True)

        embeding = torch.cat([e0, fea_encoder], dim=1)

        decoder = self.decoder(embeding)
        x1 = self.conv_block(decoder)
        x2 = self.conv_block(x1 + decoder)

        return (decoder + x1 + x2), e0

    def forward(self, moving, fixed):
        # x: moving y:fixed  b,c,d,h,w
        fea_encoder = self.ds_encoder_path(moving, fixed)
        decoder, fea = self.decoder_path(fea_encoder)
        disp = self.output(decoder) * self.range_flow
        warped_img = self.transform(moving, disp.permute(0, 2, 3, 4, 1),
                                                 self.grid_1.get_grid(moving.shape[2:], True))

        if self.is_train is True:
            return {'disp': disp, 'warped_img': warped_img, 'fea_pre': fea}
        else:
            return {'disp': disp, 'warped_img': warped_img}


class Model_lv1_signle(nn.Module):

    def __init__(self, in_channel, n_classes, start_channel, is_train=True, range_flow=0.4, grid=None, corr=False):
        super(Model_lv1_signle, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.grid_1 = grid
        self.transform = AdaptiveSpatialTransformer()
        self.correlation = CorrTorch() if corr else None

        bias_opt = False
        self.dialation_num = 3
        self.dialation_conv0 = nn.ModuleList([nn.Conv3d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=2**i,  dilation=2**i) for i in range(self.dialation_num)])

        self.dialation_conv1 = nn.ModuleList([nn.Conv3d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=2**i, dilation=2**i) for i in range(self.dialation_num)])

        self.input_encoder = input_feature_extract(self.in_channel, self.in_channel * self.dialation_num, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.in_channel * self.dialation_num * 2+27, self.start_channel * 8, kernel_size=3, stride=2,padding=1,bias=bias_opt) if corr else nn.Conv3d(self.in_channel * self.dialation_num * 2, self.start_channel * 8, kernel_size=3, stride=2,padding=1,bias=bias_opt)

        self.resblock_group = resblock_seq(self.start_channel * 8, bias_opt=bias_opt)

        self.up = nn.ConvTranspose3d(self.start_channel * 8, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.decoder = nn.Sequential(
            nn.Conv3d(self.start_channel * 4 + self.in_channel * self.dialation_num * 2+27, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1)) if corr else nn.Sequential(
            nn.Conv3d(self.start_channel * 4 + self.in_channel * self.dialation_num * 2, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1))

        self.conv_block = nn.Sequential(
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2))

        self.output = outputs(self.start_channel * 4, self.n_classes, kernel_size=3, stride=1, padding=1,
                              bias=False)

    def path_one(self, x):
        dialation_res = []
        for dialation in self.dialation_conv0:
            res = dialation(x)
            dialation_res.append(res)

        dialation_out1 = torch.sum(torch.stack(dialation_res),0)
        return self.input_encoder(dialation_out1)

    def path_two(self, x):
        dialation_res = []
        for dialation in self.dialation_conv1:
            res = dialation(x)
            dialation_res.append(res)

        dialation_out2 = torch.sum(torch.stack(dialation_res),0)
        return self.input_encoder(dialation_out2)

    def ds_encoder_path(self, moving, fixed):
        fea_moving = self.path_one(moving)
        fea_fixed = self.path_two(fixed)
        if self.correlation is not None:
            fea_correlation = self.correlation(fea_moving, fea_fixed)
            return torch.cat((fea_moving, fea_fixed, fea_correlation), 1)
        else:
            return torch.cat((fea_moving, fea_fixed), 1)

    def decoder_path(self, fea_encoder):
        e0 = self.down_conv(fea_encoder)
        e0 = self.resblock_group(e0)
        e0 = self.up(e0)

        if e0.shape[2:] != fea_encoder.shape[2:]:
            e0 = F.interpolate(e0, size=fea_encoder.shape[2:],
                               mode='trilinear',
                               align_corners=True)

        embeding = torch.cat([e0, fea_encoder], dim=1)

        decoder = self.decoder(embeding)
        x1 = self.conv_block(decoder)
        x2 = self.conv_block(x1 + decoder)

        return (decoder + x1 + x2), e0

    def forward(self, moving, fixed):
        # x: moving y:fixed  b,c,d,h,w
        fea_encoder = self.ds_encoder_path(moving, fixed)
        decoder, fea = self.decoder_path(fea_encoder)
        disp = self.output(decoder) * self.range_flow
        warped_img = self.transform(moving, disp.permute(0, 2, 3, 4, 1),
                                                 self.grid_1.get_grid(moving.shape[2:], True))

        if self.is_train is True:
            return {'disp': disp, 'warped_img': warped_img, 'fea_pre': fea}
        else:
            return {'disp': disp, 'warped_img': warped_img}

class Model_lv2_signle(nn.Module):

    def __init__(self, in_channel, n_classes, start_channel, is_train=True, range_flow=0.4, dialation_num = 3, grid=None, corr=False):
        super(Model_lv2_signle, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.grid_1 = grid
        self.transform = AdaptiveSpatialTransformer()
        self.correlation = CorrTorch() if corr else None

        bias_opt = False
        self.dialation_num = dialation_num
        self.dialation_conv0 = nn.ModuleList([nn.Conv3d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=2**i,  dilation=2**i) for i in range(self.dialation_num)])

        self.dialation_conv1 = nn.ModuleList([nn.Conv3d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=2**i, dilation=2**i) for i in range(self.dialation_num)])

        self.input_encoder = input_feature_extract(self.in_channel, self.in_channel * self.dialation_num, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.in_channel * self.dialation_num * 2+27, self.start_channel * 8, kernel_size=3, stride=2,padding=1,bias=bias_opt) if corr else nn.Conv3d(self.in_channel * self.dialation_num * 2, self.start_channel * 8, kernel_size=3, stride=2,padding=1,bias=bias_opt)

        self.resblock_group = resblock_seq(self.start_channel * 8, bias_opt=bias_opt)

        self.up = nn.ConvTranspose3d(self.start_channel * 8, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.decoder = nn.Sequential(
            nn.Conv3d(self.start_channel * 4 + self.in_channel * self.dialation_num * 2+27, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1)) if corr else nn.Sequential(
            nn.Conv3d(self.start_channel * 4 + self.in_channel * self.dialation_num * 2, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1))

        self.conv_block = nn.Sequential(
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2))

        self.output = outputs(self.start_channel * 4, self.n_classes, kernel_size=3, stride=1, padding=1,
                              bias=False)

    def path_one(self, x):
        dialation_res = []
        for dialation in self.dialation_conv0:
            res = dialation(x)
            dialation_res.append(res)

        dialation_out1 = torch.sum(torch.stack(dialation_res),0)
        return self.input_encoder(dialation_out1)

    def path_two(self, x):
        dialation_res = []
        for dialation in self.dialation_conv1:
            res = dialation(x)
            dialation_res.append(res)

        dialation_out2 = torch.sum(torch.stack(dialation_res),0)
        return self.input_encoder(dialation_out2)

    def ds_encoder_path(self, moving, fixed):
        fea_moving = self.path_one(moving)
        fea_fixed = self.path_two(fixed)
        if self.correlation is not None:
            fea_correlation = self.correlation(fea_moving, fea_fixed)
            return torch.cat((fea_moving, fea_fixed, fea_correlation), 1)
        else:
            return torch.cat((fea_moving, fea_fixed), 1)

    def decoder_path(self, fea_encoder):
        e0 = self.down_conv(fea_encoder)
        e0 = self.resblock_group(e0)
        e0 = self.up(e0)

        if e0.shape[2:] != fea_encoder.shape[2:]:
            e0 = F.interpolate(e0, size=fea_encoder.shape[2:],
                               mode='trilinear',
                               align_corners=True)

        embeding = torch.cat([e0, fea_encoder], dim=1)

        decoder = self.decoder(embeding)
        x1 = self.conv_block(decoder)
        x2 = self.conv_block(x1 + decoder)

        return (decoder + x1 + x2), e0

    def forward(self, moving, fixed):
        # x: moving y:fixed  b,c,d,h,w
        fea_encoder = self.ds_encoder_path(moving, fixed)
        decoder, fea = self.decoder_path(fea_encoder)
        disp = self.output(decoder) * self.range_flow
        warped_img = self.transform(moving, disp.permute(0, 2, 3, 4, 1),
                                                 self.grid_1.get_grid(moving.shape[2:], True))

        if self.is_train is True:
            return {'disp': disp, 'warped_img': warped_img, 'fea_pre': fea}
        else:
            return {'disp': disp, 'warped_img': warped_img}

class Model_lv1(nn.Module):

    def __init__(self, in_channel, n_classes, start_channel, is_train=True, range_flow=0.4, grid=None):
        super(Model_lv1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.grid_1 = grid
        self.transform = AdaptiveSpatialTransformer()

        bias_opt = False

        self.dialation_conv0 = nn.Conv3d(self.in_channel, self.start_channel, kernel_size=3, stride=1, padding=1,
                                         dilation=1)
        self.dialation_conv1 = nn.Conv3d(self.in_channel, self.start_channel, kernel_size=3, stride=1, padding=2,
                                         dilation=2)

        self.input_encoder_lvl1 = input_feature_extract(self.start_channel, self.start_channel,
                                                             bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel, self.start_channel * 4, kernel_size=3, stride=2,
                                   padding=1,
                                   bias=bias_opt)

        self.resblock_group_lvl1 = resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.decoder = nn.Sequential(
            nn.Conv3d(self.start_channel * 5, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1))

        self.conv_block = nn.Sequential(
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2))

        self.output_lvl1 = outputs(self.start_channel * 4, self.n_classes, kernel_size=3, stride=1, padding=1,
                                   bias=False)

    def forward(self, x, y):
        # x: moving y:fixed  b,c,d,h,w
        cat_input = torch.cat((x, y), 1)
        cat_input = self.down_avg(cat_input)
        cat_input_lvl1 = self.down_avg(cat_input)

        down_x = cat_input_lvl1[:, 0:1, :, :, :]
        down_y = cat_input_lvl1[:, 1:2, :, :, :]

        dialation_out1 = self.dialation_conv0(cat_input_lvl1)
        dialation_out2 = self.dialation_conv1(cat_input_lvl1)

        fea_e0 = self.input_encoder_lvl1(dialation_out1 + dialation_out2)

        e0 = self.down_conv(fea_e0)
        e0 = self.resblock_group_lvl1(e0)

        e0 = self.up(e0)

        if e0.shape[2:] != fea_e0.shape[2:]:
            e0 = F.interpolate(e0, size=fea_e0.shape[2:],
                               mode='trilinear',
                               align_corners=True)

        # att = self.ca_module(e0, fea_e0)
        # embeding = torch.cat([e0, fea_e0], dim=1) + att
        # embeding = att
        # show_slice(att.detach().cpu().numpy(), embeding.detach().cpu().numpy())
        embeding = torch.cat([e0, fea_e0], dim=1)

        decoder = self.decoder(embeding)
        x1 = self.conv_block(decoder)
        x2 = self.conv_block(x1 + decoder)

        decoder = decoder + x1 + x2  # B, C, D, H, W

        # att = self.ca_module(e0, fea_e0)  # B,C, DHW/d, DHW/d
        # decoder_plus = self.cross_att(decoder, att).reshape(decoder.shape[0],-1,decoder.shape[2],decoder.shape[3],decoder.shape[4])

        output_disp_e0_v = self.output_lvl1(decoder) * self.range_flow
        warpped_inputx_lvl1_out = self.transform(down_x, output_disp_e0_v.permute(0, 2, 3, 4, 1),
                                                 self.grid_1.get_grid(down_x.shape[2:], True))

        if self.is_train is True:
            return {'disp': output_disp_e0_v, 'warped_img': warpped_inputx_lvl1_out, 'down_y': down_y, 'embedding': e0}
        else:
            return {'disp': output_disp_e0_v, 'warped_img': warpped_inputx_lvl1_out}

class Model_lv2(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, range_flow=0.4,
                 model_lvl1=None, grid=None):
        super(Model_lv2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.model_lvl1 = model_lvl1

        self.grid_1 = grid

        self.transform = AdaptiveSpatialTransformer()

        bias_opt = False
        self.dialation_conv0 = nn.Conv3d(self.in_channel + 3, self.start_channel, kernel_size=3, stride=1, padding=1,
                                         dilation=1)
        self.dialation_conv1 = nn.Conv3d(self.in_channel + 3, self.start_channel, kernel_size=3, stride=1, padding=2,
                                         dilation=2)
        self.dialation_conv2 = nn.Conv3d(self.in_channel + 3, self.start_channel, kernel_size=3, stride=1, padding=4,
                                         dilation=4)

        self.input_encoder_lvl1 = input_feature_extract(self.start_channel, self.start_channel,
                                                        bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel, self.start_channel * 4, 3, stride=2, padding=1,
                                   bias=bias_opt)

        self.resblock_group_lvl1 = resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.decoder = nn.Sequential(
            nn.Conv3d(self.start_channel * 5, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1))

        self.conv_block = nn.Sequential(
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2))

        self.output_lvl2 = outputs(self.start_channel * 4, self.n_classes, kernel_size=3, stride=1, padding=1,
                                   bias=False)

        # self.cor_conv = nn.Sequential(nn.Conv3d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1),
        #                               nn.LeakyReLU(0.2))

    def unfreeze_modellvl1(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model_lvl1.parameters():
            param.requires_grad = True

    def forward(self, x, y):
        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        pred = self.model_lvl1(x, y)
        lvl1_disp, warpped_inputx_lvl1_out, lvl1_embedding = pred['flow'], pred['warped_img'], pred['embedding']

        x_down = self.down_avg(x)
        y_down = self.down_avg(y)

        lvl1_disp_up = F.interpolate(lvl1_disp, size=x_down.shape[2:],
                                     mode='trilinear',
                                     align_corners=True)

        warpped_x = self.transform(x_down, lvl1_disp_up.permute(0, 2, 3, 4, 1),
                                   self.grid_1.get_grid(x_down.shape[2:], True))

        fea_e0 = torch.cat((warpped_x, lvl1_disp_up, y_down), 1)

        dialation_out0 = self.dialation_conv0(fea_e0)
        dialation_out1 = self.dialation_conv1(fea_e0)
        dialation_out2 = self.dialation_conv2(fea_e0)

        fea_e0 = self.input_encoder_lvl1(dialation_out0 + dialation_out1 + dialation_out2)

        e0 = self.down_conv(fea_e0)
        e0 = e0 + lvl1_embedding
        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)

        # decoder = self.decoder(torch.cat([e0, fea_e0], dim=1))
        if e0.shape[2:] != fea_e0.shape[2:]:
            e0 = F.interpolate(e0, size=fea_e0.shape[2:],
                               mode='trilinear',
                               align_corners=True)

        embeding = torch.cat((e0, fea_e0), 1)
        decoder = self.decoder(embeding)

        x1 = self.conv_block(decoder)
        x2 = self.conv_block(x1 + decoder)
        decoder = decoder + x1 + x2

        disp_lv2 = self.output_lvl2(decoder)

        output_disp_e0_v = disp_lv2 * self.range_flow
        compose_field_e0_lvl2 = lvl1_disp_up + output_disp_e0_v
        warpped_inputx_lvl2_out = self.transform(x_down, compose_field_e0_lvl2.permute(0, 2, 3, 4, 1),
                                                 self.grid_1.get_grid(x_down.shape[2:], True))

        if self.is_train is True:
            return {'disp': compose_field_e0_lvl2, 'warped_img': warpped_inputx_lvl2_out, 'down_y': y_down, 'embedding': e0}
        else:
            return {'disp': compose_field_e0_lvl2, 'warped_img': warpped_inputx_lvl2_out}


class Model_lv3(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, range_flow=0.4,
                 model_lvl2=None, grid=None):
        super(Model_lv3, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.model_lvl2 = model_lvl2

        self.grid_1 = grid

        self.transform = AdaptiveSpatialTransformer()

        bias_opt = False
        self.correlation_layer = CorrTorch()
        self.dialation_conv0 = nn.Conv3d(self.in_channel + 3, self.start_channel, kernel_size=3, stride=1, padding=1,
                                         dilation=1)
        self.dialation_conv1 = nn.Conv3d(self.in_channel + 3, self.start_channel, kernel_size=3, stride=1, padding=2,
                                         dilation=2)
        self.dialation_conv2 = nn.Conv3d(self.in_channel + 3, self.start_channel, kernel_size=3, stride=1, padding=4,
                                         dilation=4)
        self.dialation_conv3 = nn.Conv3d(self.in_channel + 3, self.start_channel, kernel_size=3, stride=1, padding=6,
                                         dilation=6)

        self.input_encoder_lvl1 = input_feature_extract(self.start_channel, self.start_channel,
                                                        bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel, self.start_channel * 4, 3, stride=2, padding=1,
                                   bias=bias_opt)

        self.resblock_group_lvl1 = resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        # self.sa_module = Self_Attn(self.start_channel * 8, self.start_channel * 8)

        # self.ca_module = Cross_attention(self.start_channel * 4, self.start_channel * 4)

        # self.activate_att = nn.LeakyReLU(0.2)

        self.decoder = nn.Sequential(
            nn.Conv3d(self.start_channel * 5, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1))

        self.conv_block = nn.Sequential(
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2))

        self.output_lvl3 = outputs(self.start_channel * 4, self.n_classes, kernel_size=3, stride=1, padding=1,
                                   bias=False)

        # self.cor_conv = nn.Sequential(nn.Conv3d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1),
        #                               nn.LeakyReLU(0.2))

    def unfreeze_modellvl2(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl2 parameter")
        for param in self.model_lvl2.parameters():
            param.requires_grad = True

    def forward(self, x, y):
        # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
        pred = self.model_lvl2(x, y)
        lvl2_disp, warpped_inputx_lvl2_out, lvl2_embedding = pred['flow'], pred['warped_img'], pred['embedding']

        lvl2_disp_up = F.interpolate(lvl2_disp, size=x.shape[2:],
                                     mode='trilinear',
                                     align_corners=True)

        warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1.get_grid(x.shape[2:], True))

        fea = torch.cat((warpped_x, lvl2_disp_up, y), 1)

        dialation_out0 = self.dialation_conv0(fea)
        dialation_out1 = self.dialation_conv1(fea)
        dialation_out2 = self.dialation_conv2(fea)
        dialation_out3 = self.dialation_conv3(fea)

        fea_e0 = self.input_encoder_lvl1(dialation_out0 + dialation_out1 + dialation_out2 + dialation_out3)

        e0 = self.down_conv(fea_e0)
        e0 = e0 + lvl2_embedding
        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)

        if e0.shape[2:] != fea_e0.shape[2:]:
            e0 = F.interpolate(e0, size=fea_e0.shape[2:],
                               mode='trilinear',
                               align_corners=True)

        # decoder = self.decoder(torch.cat([e0, fea_e0], dim=1))
        # att = self.ca_module(e0, fea_e0)
        # embeding = torch.cat([e0, fea_e0], dim=1)
        # att = self.ca_module(e0, fea_e0)
        # embeding = torch.cat([e0, fea_e0], dim=1) + att

        embeding = torch.cat([e0, fea_e0], dim=1)
        decoder = self.decoder(embeding)
        x1 = self.conv_block(decoder)
        x2 = self.conv_block(x1 + decoder)

        decoder = x1 + x2
        # decoder_plus = self.cross_att(decoder, att).reshape(decoder.shape[0], -1, decoder.shape[2],
        #                                                     decoder.shape[3], decoder.shape[4])

        disp_lv3 = self.output_lvl3(decoder)
        output_disp_e0_v = disp_lv3 * self.range_flow

        compose_field_e0_lvl1 = output_disp_e0_v + lvl2_disp_up

        warpped_inputx_lvl3_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1),
                                                 self.grid_1.get_grid(x.shape[2:], True))

        return {'disp':compose_field_e0_lvl1, 'warped_img': warpped_inputx_lvl3_out}


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, num_group=4, stride=1, bias=False):
        super(PreActBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=0.2)

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.conv2(F.leaky_relu(out, negative_slope=0.2))

        out += shortcut
        return out


class NCC_bak(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=5, eps=1e-8):
        super(NCC_bak, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device,
                            requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size / 2))
        J_sum = conv_fn(J, weight, padding=int(win_size / 2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size / 2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size / 2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size / 2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)
