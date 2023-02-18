import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from Functions import generate_grid_unit


class SpatialTransform_unit(nn.Module):
    def __init__(self):
        super(SpatialTransform_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear', padding_mode="border",
                                               align_corners=True)
        return flow


class UNet_lv1(nn.Module):
    def __init__(self, in_channels, out_channels, dim=3, is_train=True, depth=2, imgshape=(144, 144, 144),
                 initial_channels=7, normalization=True):

        super().__init__()
        assert dim in (2, 3)
        self.dim = dim

        self.depth = depth
        prev_channels = in_channels

        self.is_train = is_train

        self.imgshape = imgshape

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.transform = SpatialTransform_unit().cuda()
        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.down_path = nn.ModuleList()

        for i in range(self.depth):
            current_channels = 4 ** i * initial_channels
            self.down_path.append(ConvBlock(prev_channels, current_channels, dim, normalization))
            prev_channels = current_channels

        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            current_channels = 4 ** i * initial_channels
            # print(prev_channels, current_channels)
            self.up_path.append(UpBlock(prev_channels, current_channels, dim, normalization))
            prev_channels = current_channels

        if dim == 2:
            self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=(1, 1))
        elif dim == 3:
            self.last = nn.Conv3d(prev_channels, out_channels, kernel_size=(1, 1, 1))

    def forward(self, x, y):
        # x: moving y:fixed  b,c,d,h,w
        cat_input = torch.cat((x, y), 1)

        cat_input = self.down_avg(cat_input)
        x = self.down_avg(cat_input)
        down_x = x[:, 0:1, :, :, :]
        down_y = x[:, 1:2, :, :, :]

        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i < self.depth - 1:
                blocks.append(x)
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear' if self.dim == 2 else 'trilinear',
                                  align_corners=True, recompute_scale_factor=False)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        output_disp_e0_v = self.last(x) * 0.4
        warpped_inputx_lvl1_out = self.transform(down_x, output_disp_e0_v.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return output_disp_e0_v, warpped_inputx_lvl1_out, down_y, output_disp_e0_v
        else:
            return output_disp_e0_v


class UNet_lv2(nn.Module):

    def __init__(self, in_channels, out_channels, dim=3, is_train=True, depth=2, imgshape=(144, 144, 144),
                 initial_channels=7, normalization=True, model_lvl1=None):

        super().__init__()
        assert dim in (2, 3)
        self.dim = dim

        self.depth = depth
        prev_channels = in_channels

        self.is_train = is_train

        self.imgshape = imgshape
        self.model_lvl1 = model_lvl1
        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.transform = SpatialTransform_unit().cuda()
        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.down_path = nn.ModuleList()

        for i in range(self.depth):
            current_channels = 4 ** i * initial_channels
            self.down_path.append(ConvBlock(prev_channels, current_channels, dim, normalization))
            prev_channels = current_channels

        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            current_channels = 4 ** i * initial_channels
            # print(prev_channels, current_channels)
            self.up_path.append(UpBlock(prev_channels, current_channels, dim, normalization))
            prev_channels = current_channels

        if dim == 2:
            self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=(1, 1))
        elif dim == 3:
            self.last = nn.Conv3d(prev_channels, out_channels, kernel_size=(1, 1, 1))

    def unfreeze_modellvl1(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model_lvl1.parameters():
            param.requires_grad = True

    def forward(self, x, y):
        # x: moving y:fixed  b,c,d,h,w
        lvl1_disp, _, _, lvl1_v = self.model_lvl1(x, y)
        lvl1_disp_up = self.up_tri(lvl1_disp)

        x_down = self.down_avg(x)
        y_down = self.down_avg(y)

        warpped_x = self.transform(x_down, lvl1_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)
        x = torch.cat((warpped_x, y_down), 1)

        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i < self.depth - 1:
                blocks.append(x)
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear' if self.dim == 2 else 'trilinear',
                                  align_corners=True, recompute_scale_factor=False)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        output_disp_e0_v = self.last(x) * 0.4
        compose_field_e0_lvl1 = lvl1_disp_up + output_disp_e0_v
        warpped_inputx_lvl1_out = self.transform(x_down, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y_down, output_disp_e0_v, lvl1_v
        else:
            return compose_field_e0_lvl1


class UNet_lv3(nn.Module):

    def __init__(self, in_channels, out_channels, dim=3, is_train=True, depth=2, imgshape=(144, 144, 144),
                 initial_channels=7, normalization=True, model_lvl2=None):

        super().__init__()
        assert dim in (2, 3)
        self.dim = dim

        self.depth = depth
        prev_channels = in_channels

        self.is_train = is_train
        self.model_lvl2 = model_lvl2
        self.imgshape = imgshape

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.transform = SpatialTransform_unit().cuda()
        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.down_path = nn.ModuleList()

        for i in range(self.depth):
            current_channels = 4 ** i * initial_channels
            self.down_path.append(ConvBlock(prev_channels, current_channels, dim, normalization))
            prev_channels = current_channels

        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            current_channels = 4 ** i * initial_channels
            # print(prev_channels, current_channels)
            self.up_path.append(UpBlock(prev_channels, current_channels, dim, normalization))
            prev_channels = current_channels

        if dim == 2:
            self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=(1, 1))
        elif dim == 3:
            self.last = nn.Conv3d(prev_channels, out_channels, kernel_size=(1, 1, 1))

    def unfreeze_modellvl2(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model_lvl1.parameters():
            param.requires_grad = True

    def forward(self, x, y):
        # x: moving y:fixed  b,c,d,h,w
        lvl2_disp, _, _, lvl2_v, lvl1_v = self.model_lvl2(x, y)
        lvl2_disp_up = self.up_tri(lvl2_disp)
        x_ori = x
        warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)
        x = torch.cat((warpped_x, y), 1)

        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i < self.depth - 1:
                blocks.append(x)
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear' if self.dim == 2 else 'trilinear',
                                  align_corners=True, recompute_scale_factor=False)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        output_disp_e0_v = self.last(x) * 0.4
        compose_field_e0_lvl1 = lvl2_disp_up + output_disp_e0_v
        warpped_inputx_lvl1_out = self.transform(x_ori, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v
        else:
            return compose_field_e0_lvl1


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
