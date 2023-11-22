import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.Functions import generate_grid_unit, Grid, AdaptiveSpatialTransformer
from utils.losses import NCC
from utils.Attention import across_attention_multi as across_atn


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


class CRegNet_lv0(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, range_flow=0.4, grid=None):
        super(CRegNet_lv0, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.grid_1 = grid
        self.transform = AdaptiveSpatialTransformer()

        bias_opt = False

        self.input_encoder = input_feature_extract(self.in_channel, self.start_channel * 2, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 2, self.start_channel * 4, 3, stride=2, padding=1,
                                   bias=bias_opt)

        self.resblock = resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        # self.sa_module = Self_Attn(self.start_channel * 8, self.start_channel * 8)
        # self.ca_module = Cross_attention(self.start_channel * 4, self.start_channel * 4)

        self.decoder = nn.Sequential(
            nn.Conv3d(self.start_channel * 6, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, padding=1))

        self.conv_block = nn.Sequential(
            nn.Conv3d(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2))

        self.output = outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1,
                              bias=False)

    def forward(self, x, y):
        # x: moving y:fixed  b,c,d,h,w
        cat_input = torch.cat((x, y), 1)
        cat_input_lvl0 = self.down_avg(self.down_avg(self.down_avg(cat_input)))
        down_x = cat_input_lvl0[:, 0:1, :, :, :]
        down_y = cat_input_lvl0[:, 1:2, :, :, :]

        fea_e0 = self.input_encoder(cat_input_lvl0)  # b,start_channel * 2, d/8,h/8,w/8
        e0 = self.down_conv(fea_e0)  # b,start_channel * 4, d/16,h/16,w/16
        e0 = self.resblock(e0)
        e0 = self.up(e0)  # b,start_channel * 2, d/8,h/8,w/8

        if e0.shape[2:] != fea_e0.shape[2:]:
            e0 = F.interpolate(e0, size=fea_e0.shape[2:], mode='trilinear', align_corners=True)

        decoder = self.decoder(torch.cat([e0, fea_e0], dim=1))  # b,start_channel * 2, d/8,h/8,w/8
        x1 = self.conv_block(decoder)
        x2 = self.conv_block(x1 + decoder)

        decoder = x1 + x2

        output_disp = self.output(decoder) * self.range_flow
        warpped_input = self.transform(down_x, output_disp.permute(0, 2, 3, 4, 1),
                                       self.grid_1.get_grid(down_x.shape[2:], True))

        if self.is_train is True:
            return {'flow': output_disp, 'warped_img': warpped_input, 'down_y': down_y, 'embedding': e0}
        else:
            return {'flow': output_disp, 'warped_img': warpped_input}


class CRegNet_lv1(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, range_flow=0.4, grid=None, model_lvl0=None, att=False):
        super(CRegNet_lv1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.model_lv0 = model_lvl0
        self.range_flow = range_flow
        self.is_train = is_train
        self.att = att
        self.grid_1 = grid
        self.transform = AdaptiveSpatialTransformer()

        bias_opt = False

        self.input_encoder_lvl1 = input_feature_extract(self.in_channel + 3, self.start_channel * 4,
                                                        bias=bias_opt) if model_lvl0 is not None else input_feature_extract(
            self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                   bias=bias_opt)

        self.resblock_group_lvl1 = resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        # self.sa_module = Self_Attn(self.start_channel * 8, self.start_channel * 8)
        if model_lvl0 is not None and att:
            self.ca_module = across_atn(self.start_channel * 4, self.start_channel * 4)

        self.decoder = nn.Sequential(
            nn.Conv3d(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
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

        if self.model_lv0 is not None:
            pred = self.model_lv0(x, y)
            lvl0_disp, warpped_inputx_lvl0_out, lvl0_embedding = pred['flow'], pred['warped_img'], pred['embedding']
            lvl0_disp_up = F.interpolate(lvl0_disp, size=down_x.shape[2:], mode='trilinear', align_corners=True)
            warpped_x = self.transform(down_x, lvl0_disp_up.permute(0, 2, 3, 4, 1),
                                       self.grid_1.get_grid(down_x.shape[2:], True))
            cat_input_lvl1 = torch.cat((warpped_x, down_y, lvl0_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl1)
        e0 = self.down_conv(fea_e0)

        if self.model_lv0 is not None:
            e0 = e0 + lvl0_embedding

        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)

        if e0.shape[2:] != fea_e0.shape[2:]:
            e0 = F.interpolate(e0, size=fea_e0.shape[2:], mode='trilinear', align_corners=True)

        decoder = self.decoder(torch.cat([e0, fea_e0], dim=1))
        x1 = self.conv_block(decoder)
        x2 = self.conv_block(x1 + decoder)

        decoder = x1 + x2

        output_disp_e0_v = self.output_lvl1(decoder) * self.range_flow
        if self.model_lv0 is not None:
            output_disp_e0_v += lvl0_disp_up
            if self.att:
                output_disp_e0_v += self.ca_module(fea_e0, lvl0_embedding, output_disp_e0_v)
        warpped_inputx_lvl1_out = self.transform(down_x, output_disp_e0_v.permute(0, 2, 3, 4, 1),
                                                 self.grid_1.get_grid(down_x.shape[2:], True))

        if self.is_train is True:
            return {'flow': output_disp_e0_v, 'warped_img': warpped_inputx_lvl1_out, 'down_y': down_y, 'embedding': e0}
        else:
            return {'flow': output_disp_e0_v, 'warped_img': warpped_inputx_lvl1_out}


class CRegNet_lv2(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, range_flow=0.4,
                 model_lvl1=None, grid=None, att=False):
        super(CRegNet_lv2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.att = att
        self.range_flow = range_flow
        self.is_train = is_train

        self.model_lvl1 = model_lvl1

        self.grid_1 = grid

        self.transform = AdaptiveSpatialTransformer()

        bias_opt = False

        self.input_encoder_lvl1 = input_feature_extract(self.in_channel + 3, self.start_channel * 4,
                                                        bias=bias_opt) if model_lvl1 is not None else input_feature_extract(
            self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                   bias=bias_opt)

        self.resblock_group_lvl1 = resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        # self.sa_module = Self_Attn(self.start_channel * 8, self.start_channel * 8)
        if model_lvl1 is not None and att:
            self.ca_module = across_atn(self.start_channel * 4, self.start_channel * 4)

        self.decoder = nn.Sequential(
            nn.Conv3d(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
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
        x_down = self.down_avg(x)
        y_down = self.down_avg(y)
        cat_input_lvl2 = torch.cat((x_down, y_down), 1)
        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        if self.model_lvl1 is not None:
            pred = self.model_lvl1(x, y)
            lvl1_disp, warpped_inputx_lvl1_out, lvl1_embedding = pred['flow'], pred['warped_img'], pred['embedding']
            # lvl1_disp_up = self.up_tri(lvl1_disp)
            lvl1_disp_up = F.interpolate(lvl1_disp, size=x_down.shape[2:],
                                         mode='trilinear',
                                         align_corners=True)
            warpped_x = self.transform(x_down, lvl1_disp_up.permute(0, 2, 3, 4, 1),
                                       self.grid_1.get_grid(x_down.shape[2:], True))

            cat_input_lvl2 = torch.cat((warpped_x, y_down, lvl1_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl2)
        e0 = self.down_conv(fea_e0)

        if self.model_lvl1 is not None:
            e0 = e0 + lvl1_embedding

        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        if e0.shape[2:] != fea_e0.shape[2:]:
            e0 = F.interpolate(e0, size=fea_e0.shape[2:], mode='trilinear', align_corners=True)

        decoder = self.decoder(torch.cat([e0, fea_e0], dim=1))
        x1 = self.conv_block(decoder)
        x2 = self.conv_block(x1 + decoder)

        decoder = x1 + x2

        output_disp_e0_v = self.output_lvl2(decoder) * self.range_flow
        if self.model_lvl1 is not None:
            compose_field_e0_lvl2 = output_disp_e0_v + lvl1_disp_up
            if self.att:
                compose_field_e0_lvl2 += self.ca_module(fea_e0, lvl1_embedding, output_disp_e0_v)
        else:
            compose_field_e0_lvl2 = output_disp_e0_v

        warpped_inputx_lvl2_out = self.transform(x_down, compose_field_e0_lvl2.permute(0, 2, 3, 4, 1),
                                                 self.grid_1.get_grid(x_down.shape[2:], True))

        if self.is_train is True:
            # return compose_field_e0_lvl2, warpped_inputx_lvl1_out, warpped_inputx_lvl2_out, y_down, output_disp_e0_v, lvl1_v, e0
            return {'flow': compose_field_e0_lvl2, 'warped_img': warpped_inputx_lvl2_out, 'down_y': y_down,
                    'embedding': e0}
        else:
            return {'flow': compose_field_e0_lvl2, 'warped_img': warpped_inputx_lvl2_out}


class CRegNet_lv3(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, range_flow=0.4,
                 model_lvl2=None, grid=None, att=False):
        super(CRegNet_lv3, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.att = att
        self.range_flow = range_flow
        self.is_train = is_train

        self.model_lvl2 = model_lvl2

        self.grid_1 = grid

        self.transform = AdaptiveSpatialTransformer()

        bias_opt = False

        self.input_encoder_lvl1 = input_feature_extract(self.in_channel + 3, self.start_channel * 4,
                                                        bias=bias_opt) if model_lvl2 is not None else input_feature_extract(
            self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                   bias=bias_opt)

        self.resblock_group_lvl1 = resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        # self.sa_module = Self_Attn(self.start_channel * 8, self.start_channel * 8)
        if model_lvl2 is not None and att:
            self.ca_module = across_atn(self.start_channel * 4, self.start_channel * 4)

        self.decoder = nn.Sequential(
            nn.Conv3d(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
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
        cat_input = torch.cat((x, y), 1)
        if self.model_lvl2 is not None:
            # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
            pred = self.model_lvl2(x, y)
            lvl2_disp, warpped_inputx_lvl2_out, lvl2_embedding = pred['flow'], pred['warped_img'], pred['embedding']
            # lvl2_disp_up = self.up_tri(lvl2_disp)
            lvl2_disp_up = F.interpolate(lvl2_disp, size=x.shape[2:],
                                         mode='trilinear',
                                         align_corners=True)

            warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1.get_grid(x.shape[2:], True))

            cat_input = torch.cat((warpped_x, y, lvl2_disp_up), 1)
        # cat_input = torch.cat((y, lvl2_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input)
        e0 = self.down_conv(fea_e0)

        if self.model_lvl2 is not None:
            e0 = e0 + lvl2_embedding

        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        if e0.shape[2:] != fea_e0.shape[2:]:
            e0 = F.interpolate(e0, size=fea_e0.shape[2:], mode='trilinear', align_corners=True)

        decoder = self.decoder(torch.cat([e0, fea_e0], dim=1))
        x1 = self.conv_block(decoder)
        x2 = self.conv_block(x1 + decoder)

        decoder = x1 + x2

        output_disp_e0_v = self.output_lvl3(decoder) * self.range_flow
        if self.model_lvl2 is not None:
            compose_field_e0_lvl1 = output_disp_e0_v + lvl2_disp_up
            if self.att:
                compose_field_e0_lvl1 += self.ca_module(fea_e0, lvl2_embedding, output_disp_e0_v)
        else:
            compose_field_e0_lvl1 = output_disp_e0_v

        warpped_inputx_lvl3_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1),
                                                 self.grid_1.get_grid(x.shape[2:], True))

        return {'flow': compose_field_e0_lvl1, 'warped_img': warpped_inputx_lvl3_out}


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


class CRegNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, range_flow=0.4, grid=None):
        super(CRegNet, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.grid_1 = grid

        self.transform = AdaptiveSpatialTransformer()

        bias_opt = False

        self.input_encoder_lvl1 = input_feature_extract(self.in_channel, self.start_channel * 2, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 2, self.start_channel * 4, 3, stride=2, padding=1,
                                   bias=bias_opt)

        self.resblock_group_lvl1 = resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        # self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        # self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 2, 2, stride=2,
        #                              padding=0, output_padding=0, bias=bias_opt)
        self.conv_up = nn.Conv3d(self.start_channel * 4, self.start_channel * 2, kernel_size=1)

        # self.sa_module = Self_Attn(self.start_channel * 8, self.start_channel * 8)
        # self.ca_module = Cross_attention(self.start_channel * 4, self.start_channel * 4)

        self.decoder = nn.Sequential(
            nn.Conv3d(self.start_channel * 4, self.start_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, padding=1))

        self.conv_block = nn.Sequential(
            nn.Conv3d(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2))

        self.output_lvl3 = outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1,
                                   bias=False)

    def forward(self, x, y):
        # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
        # pred = self.model_lvl2(x, y)
        # lvl2_disp, warpped_inputx_lvl2_out, lvl2_embedding = pred['flow'], pred['warped_img'], pred['embedding']
        # # lvl2_disp_up = self.up_tri(lvl2_disp)
        #
        # lvl2_disp_up = F.interpolate(lvl2_disp, size=x.shape[2:],
        #                              mode='trilinear',
        #                              align_corners=True)
        #
        # warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1.get_grid(x.shape[2:], True))

        # cat_input = torch.cat((warpped_x, y, lvl2_disp_up), 1)
        cat_input = torch.cat((x, y), 1)
        fea_e0 = self.input_encoder_lvl1(cat_input)
        e0 = self.down_conv(fea_e0)
        e0 = self.resblock_group_lvl1(e0)
        # e0 = self.up(e0)
        e0 = self.conv_up(e0)
        if e0.shape[2:] != fea_e0.shape[2:]:
            e0 = F.interpolate(e0, size=fea_e0.shape[2:], mode='trilinear', align_corners=True)

        decoder = self.decoder(torch.cat([e0, fea_e0], dim=1))
        x1 = self.conv_block(decoder)
        x2 = self.conv_block(x1 + decoder)

        decoder = x1 + x2

        output_disp_e0_v = self.output_lvl3(decoder) * self.range_flow

        warpped_inputx_lvl3_out = self.transform(x, output_disp_e0_v.permute(0, 2, 3, 4, 1),
                                                 self.grid_1.get_grid(x.shape[2:], True))

        return {'flow': output_disp_e0_v, 'warped_img': warpped_inputx_lvl3_out}
