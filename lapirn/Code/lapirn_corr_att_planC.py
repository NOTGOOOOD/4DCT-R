import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.Functions import generate_grid_unit
from utils.losses import NCC
from utils.Attention import Cross_attention_2


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


class Miccai2020_LDR_laplacian_unit_disp_add_lvl1(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4):
        super(Miccai2020_LDR_laplacian_unit_disp_add_lvl1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.transform = SpatialTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                   bias=bias_opt)

        self.resblock_group_lvl1 = resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        # self.sa_module = Self_Attn(self.start_channel * 8, self.start_channel * 8)
        self.ca_module = Cross_attention_2(self.start_channel * 4, self.start_channel * 4)

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

        fea_x = self.input_encoder_lvl1(down_x)
        fea_y = self.input_encoder_lvl1(down_y)

        # Dual stream
        fea_x_e0 = self.up(self.resblock_group_lvl1(self.down_conv(fea_x)))
        fea_y_e0 = self.up(self.resblock_group_lvl1(self.down_conv(fea_y)))

        if fea_x_e0.shape != fea_x.shape:
            fea_x_e0 = F.interpolate(fea_x_e0, size=fea_x.shape[2:],
                               mode='trilinear',
                               align_corners=True)

            fea_y_e0 = F.interpolate(fea_y_e0, size=fea_y.shape[2:],
                                     mode='trilinear',
                                     align_corners=True)

        fea_xy_e0 = torch.cat([fea_x_e0, fea_y_e0], dim=1)
        embeding = self.ca_module(fea_x_e0, fea_y_e0)

        # correlation layer
        decoder = self.decoder(embeding)
        x1 = self.conv_block(decoder)
        x2 = self.conv_block(x1 + decoder)

        decoder = x1 + x2  # B, C, D, H, W

        output_disp_e0_v = self.output_lvl1(decoder) * self.range_flow
        warpped_inputx_lvl1_out = self.transform(down_x, output_disp_e0_v.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return output_disp_e0_v, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, fea_xy_e0
        else:
            return output_disp_e0_v, warpped_inputx_lvl1_out


class Miccai2020_LDR_laplacian_unit_disp_add_lvl2(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4,
                 model_lvl1=None):
        super(Miccai2020_LDR_laplacian_unit_disp_add_lvl2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl1 = model_lvl1

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.transform = SpatialTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = input_feature_extract(self.in_channel + 3, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                   bias=bias_opt)

        self.resblock_group_lvl1 = resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.ca_module = Cross_attention_2(self.start_channel * 4, self.start_channel * 4)

        self.activate_att = nn.LeakyReLU(0.2)

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
        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        lvl1_disp, warpped_inputx_lvl1_out, _, lvl1_v, lvl1_embedding = self.model_lvl1(x, y)

        x_down = self.down_avg(x)
        y_down = self.down_avg(y)

        lvl1_disp_up = F.interpolate(lvl1_disp, size=x_down.shape[2:],
                                     mode='trilinear',
                                     align_corners=True)

        warpped_x = self.transform(x_down, lvl1_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        # cat_input_lvl2 = torch.cat((warpped_x, y_down, lvl1_disp_up), 1)

        fea_x = self.input_encoder_lvl1(torch.cat((warpped_x, lvl1_disp_up), 1))
        fea_y = self.input_encoder_lvl1(torch.cat((y_down, lvl1_disp_up), 1))

        x_embedding = lvl1_embedding[:, 0:28, ...]
        y_embedding = lvl1_embedding[:, 28:, ...]
        fea_x_e0 = self.up(self.resblock_group_lvl1(self.down_conv(fea_x) + x_embedding))
        fea_y_e0 = self.up(self.resblock_group_lvl1(self.down_conv(fea_y) + y_embedding))

        if fea_x_e0.shape != fea_x.shape:
            fea_x_e0 = F.interpolate(fea_x_e0, size=fea_x.shape[2:],
                               mode='trilinear',
                               align_corners=True)

            fea_y_e0 = F.interpolate(fea_y_e0, size=fea_y.shape[2:],
                                     mode='trilinear',
                                     align_corners=True)

        fea_xy_e0 = torch.cat([fea_x_e0, fea_y_e0], dim=1)
        embeding = self.ca_module(fea_x_e0, fea_y_e0)

        # embeding = torch.cat([e0, fea_e0], dim=1)
        decoder = self.decoder(embeding)
        x1 = self.conv_block(decoder)
        x2 = self.conv_block(x1 + decoder)
        decoder = x1 + x2

        # att = self.ca_module(e0, fea_e0)
        # decoder_plus = self.cross_att(decoder, att).reshape(decoder.shape[0], -1, decoder.shape[2],
        #                                                     decoder.shape[3], decoder.shape[4])

        disp_lv2 = self.output_lvl2(decoder)
        output_disp_e0_v = disp_lv2 * self.range_flow
        compose_field_e0_lvl2 = lvl1_disp_up + output_disp_e0_v
        warpped_inputx_lvl2_out = self.transform(x_down, compose_field_e0_lvl2.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl2, warpped_inputx_lvl1_out, warpped_inputx_lvl2_out, y_down, output_disp_e0_v, lvl1_v, fea_xy_e0
        else:
            return compose_field_e0_lvl2, warpped_inputx_lvl1_out, warpped_inputx_lvl2_out


class Miccai2020_LDR_laplacian_unit_disp_add_lvl3(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4,
                 model_lvl2=None):
        super(Miccai2020_LDR_laplacian_unit_disp_add_lvl3, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl2 = model_lvl2

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.transform = SpatialTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = input_feature_extract(self.in_channel + 3, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                   bias=bias_opt)

        self.resblock_group_lvl1 = resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        # self.sa_module = Self_Attn(self.start_channel * 8, self.start_channel * 8)

        self.ca_module = Cross_attention_2(self.start_channel * 4, self.start_channel * 4)

        self.activate_att = nn.LeakyReLU(0.2)

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
        # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
        lvl2_disp, warpped_inputx_lvl1_out, warpped_inputx_lvl2_out, _, lvl2_v, lvl1_v, lvl2_embedding = self.model_lvl2(
            x, y)

        lvl2_disp_up = F.interpolate(lvl2_disp, size=x.shape[2:],
                                     mode='trilinear',
                                     align_corners=True)
        warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        fea_x = self.input_encoder_lvl1(torch.cat((warpped_x, lvl2_disp_up), 1))
        fea_y = self.input_encoder_lvl1(torch.cat((y, lvl2_disp_up), 1))

        x_embedding = lvl2_embedding[:, 0:28, ...]
        y_embedding = lvl2_embedding[:, 28:, ...]

        fea_x_e0 = self.up(self.resblock_group_lvl1(self.down_conv(fea_x) + x_embedding))
        fea_y_e0 = self.up(self.resblock_group_lvl1(self.down_conv(fea_y) + y_embedding))

        if fea_x_e0.shape != fea_x.shape:
            fea_x_e0 = F.interpolate(fea_x_e0, size=fea_x.shape[2:],
                               mode='trilinear',
                               align_corners=True)

            fea_y_e0 = F.interpolate(fea_y_e0, size=fea_y.shape[2:],
                                     mode='trilinear',
                                     align_corners=True)

        fea_xy_e0 = torch.cat([fea_x_e0, fea_y_e0], dim=1)
        embeding = self.ca_module(fea_x_e0, fea_y_e0)

        decoder = self.decoder(embeding)
        x1 = self.conv_block(decoder)
        x2 = self.conv_block(x1 + decoder)

        decoder = x1 + x2
        # decoder_plus = self.cross_att(decoder, att).reshape(decoder.shape[0], -1, decoder.shape[2],
        #                                                     decoder.shape[3], decoder.shape[4])

        disp_lv3 = self.output_lvl3(decoder)
        output_disp_e0_v = disp_lv3 * self.range_flow

        compose_field_e0_lvl1 = output_disp_e0_v + lvl2_disp_up

        warpped_inputx_lvl3_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, warpped_inputx_lvl2_out, warpped_inputx_lvl3_out, y, output_disp_e0_v, lvl1_v, lvl2_v, fea_xy_e0
        else:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, warpped_inputx_lvl2_out, warpped_inputx_lvl3_out


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


class SpatialTransform_unit(nn.Module):
    def __init__(self):
        super(SpatialTransform_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear', padding_mode="border",
                                               align_corners=True)
        return flow


class SpatialTransformNearest_unit(nn.Module):
    def __init__(self):
        super(SpatialTransformNearest_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest', padding_mode="border",
                                               align_corners=True)
        return flow


class DiffeomorphicTransform_unit(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform_unit, self).__init__()
        self.time_step = time_step

    def forward(self, velocity, sample_grid):
        flow = velocity / (2.0 ** self.time_step)
        for _ in range(self.time_step):
            grid = sample_grid + flow.permute(0, 2, 3, 4, 1)
            flow = flow + F.grid_sample(flow, grid, mode='bilinear', padding_mode="border", align_corners=True)
        return flow


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
