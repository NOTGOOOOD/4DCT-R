import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.correlation_layer import CorrTorch as Correlation

def apply_offset(offset):
    '''
        convert offset grid to location grid
        offset: [N, 2, H, W] for 2D or [N, 3, D, H, W] for 3D
        output: [N, H, W, 2] for 2D or [N, D, H, W, 3] for 3D
    '''
    sizes = list(offset.size()[2:]) # [D, H, W] or [H, W]
    grid_list = torch.meshgrid([ torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)

    # apply offset
    grid_list = [ grid.float().unsqueeze(0) + offset[:, dim, ...]
        for dim, grid in enumerate(grid_list) ]

    # normalize
    grid_list = [ grid / ((size-1.0)/2.0) - 1.0
        for grid, size in zip(grid_list, reversed(sizes))]
    return torch.stack(grid_list, dim=-1)

def permute_channel_last(x):
    dims = [0] + list(range(2, x.dim())) + [1]
    return x.permute(*dims)

def permute_channel_first(x):
    dims = [0, x.dim()-1] + list(range(1, x.dim()-1))
    return x.permute(*dims)

class DownSample(nn.Module):

    def __init__(self, ftr1, ftr2):
        super(DownSample, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ftr1, ftr1, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm3d(ftr1),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool3d(2),
            nn.Conv3d(ftr1, ftr1, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(ftr1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(ftr1, ftr2, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(ftr2),
            nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)

class Bottle(nn.Module):

    def __init__(self, ftr1, ftr2):
        super(Bottle, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ftr1, ftr2, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(ftr2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(ftr2, ftr2, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(ftr2),
            nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)


class _CascadeFusionNet(nn.Module):
    
    def __init__(self, backend, conv_size, fpn_dim, keep_delta=True):
        super(_CascadeFusionNet, self).__init__()
        self.num_pyramid = len(conv_size)
        self.keep_delta = keep_delta

    def forward(self, source, target, train=False):
        x = torch.cat((source, target), dim=0)
        batch_size = 1
        pyramid = self.pyramid(x)
        feature_warp = []
        feature_fix = []
        shape_list = []
        last_flow = None
        delta_list = []

        for feature in pyramid:
            feature_warp.append(feature[:batch_size, ...])
            feature_fix.append(feature[batch_size:, ...])
            shape_list.append(feature.shape[2:])

        feature_warp = reversed(feature_warp)
        feature_fix = reversed(feature_fix)

        # delta_last_list = []
        for lvl, (x_warp, x_fix) in enumerate(zip(feature_warp, feature_fix)):
            # apply flow
            if last_flow is not None:
                x_warp = F.grid_sample(x_warp, permute_channel_last(last_flow.detach()),
                     mode='bilinear', padding_mode="border",align_corners=True,)
            # fusion
            flow = self.offset[lvl](x_warp, x_fix)
            # cascade
            # flow = self.offset[lvl](torch.cat([x_warp, x_fix], dim=1))
            if self.keep_delta:
                delta_list.append(flow)
            flow = apply_offset(flow)
            if last_flow is not None:
                flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode="border",align_corners=True,)
            else:
                flow = permute_channel_first(flow)
            # last_flow = F.interpolate(flow, scale_factor=2, mode=self.interp_mode)
            if lvl < len(pyramid)-2:
                # last_flow = F.interpolate(flow, scale_factor=2, mode=self.interp_mode)
                last_flow = F.interpolate(flow, size=shape_list[-2-lvl], mode='trilinear', align_corners=True)
            else:
                last_flow = flow
            # if self.keep_delta:
            #     delta_last_list.append(last_flow)
        x_warp = x[:batch_size, ...]
        x_warp = F.grid_sample(x_warp, permute_channel_last(last_flow),
                     mode='bilinear', padding_mode="border", align_corners=True,)

        return {'warped_img': x_warp, 'flow': last_flow, 'delta_list': delta_list}


class CascadeFusionNet3d(_CascadeFusionNet):
    
    def __init__(self, backend, conv_size, fpn_dim, keep_delta=True):
        super(CascadeFusionNet3d, self).__init__(backend, conv_size, fpn_dim, keep_delta)
        self.pyramid = backend
        self.interp_mode = "trilinear"

        # adaptive 
        self.fuse = []
        self.resblock = []
        self.offset = []
        for i in range(len(conv_size)):
            # fusion
            offset_layer = FlowNet(conv_size[-i-1]*2)
            # cascade
            # offset_layer = nn.Conv3d(conv_size[-i-1]*2, 3, kernel_size=3, padding=1)
            self.offset.append(offset_layer)
        self.offset = nn.ModuleList(self.offset)
        


class FlowNet_bk(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(FlowNet_bk, self).__init__()
        med_channels = int(in_channels/2)
        self.fuse_layer = nn.Sequential(
                nn.Conv3d(in_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.resblock_layer_1 = nn.Sequential(
                nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.offset_layer = nn.Conv3d(med_channels, out_channels, kernel_size=3, padding=1)
        # self.offset_layer = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.fuse_layer(x)
        x = x + self.resblock_layer_1(x)
        x = self.offset_layer(x)
        return x
    
class FlowNet(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(FlowNet, self).__init__()
        med_channels = int(in_channels/2)
        self.corr_layer = Correlation()
        self.conv_layer = nn.Sequential(
                # nn.Conv3d(27+in_channels, med_channels, kernel_size=3, padding=1),
                nn.Conv3d(in_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.resblock_layer_1 = nn.Sequential(
                nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        # self.resblock_layer_2 = nn.Sequential(
        #         nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
        #         nn.ReLU(inplace=True),
        #     )
        self.offset_layer = nn.Conv3d(med_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        # x3 = self.corr_layer(x1, x2)
        # x = torch.cat([x1, x2, x3], dim=1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv_layer(x)
        x = x + self.resblock_layer_1(x)
        x = self.offset_layer(x)
        return x

if __name__ == "__main__":
    in_channels = 1
    conv_size = [8, 16, 16, 32, 32]
    fpn_dim = [32, 32, 16, 16, 8]

    backend = FlowNet(in_channels=in_channels)
    model = CascadeFusionNet3d(backend=backend,conv_size=conv_size,fpn_dim=fpn_dim)
        

