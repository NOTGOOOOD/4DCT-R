from voxelmorph.model import unet
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from utils.utilize import count_parameters


class RegNet_pairwise(nn.Module):
    '''
    Pairwise CNN registration method.

    Parameters
    ----------
    dim : int
        Dimension of input image.
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : TYPE, optional
        Number of initial channels. The default is 64.
    normalization : TYPE, optional
        Whether to add instance normalization after activation. The default is True.
    '''

    def __init__(self, dim, scale=1, depth=5, initial_channels=64, normalization=True):

        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.scale = scale

        self.unet = unet.UNet(in_channels=2, out_channels=dim, dim=dim, depth=depth, initial_channels=initial_channels,
                              normalization=normalization)
        self.spatial_transform = SpatialTransformer(self.dim)

    def forward(self, fixed_image, moving_image):
        '''
        Parameters
        ----------
        fixed_image, moving_image : (h, w) or (d, h, w)
            Fixed and moving image to be registered

        Returns
        -------
        warped_moving_image : (h, w) or (d, h, w)
            Warped input image.
        disp : (2, h, w) or (3, d, h, w)
            Flow field from fixed image to moving image.
        scaled_disp
        '''

        original_image_shape = fixed_image.shape
        input_image = torch.cat((fixed_image, moving_image),dim=1)   # (1, 2, h, w) or (1, 2, d, h, w)

        if self.scale < 1:
            scaled_image = F.interpolate(input_image, scale_factor=self.scale, align_corners=True,
                                         mode='bilinear' if self.dim == 2 else 'trilinear',
                                         recompute_scale_factor=False)  # (1, 2, h, w) or (1, 2, d, h, w)
        else:
            scaled_image = input_image

        scaled_image_shape = scaled_image.shape[2:]
        scaled_disp = torch.squeeze(self.unet(scaled_image), 0).reshape(self.dim,
                                                                        *scaled_image_shape)  # (2, h, w) or (3, d, h, w)
        if self.scale < 1:
            disp = torch.nn.functional.interpolate(torch.unsqueeze(scaled_disp, 0), size=original_image_shape,
                                                   mode='bilinear' if self.dim == 2 else 'trilinear',
                                                   align_corners=True)
        else:
            disp = torch.unsqueeze(scaled_disp, 0)

        warped_moving_image = self.spatial_transform(input_image[:, 1:], disp)  # (h, w) or (d, h, w)

        return disp, warped_moving_image


class SpatialTransformer(nn.Module):
    # 2D or 3d spatial transformer network to calculate the warped moving image

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.grid_dict = {}
        self.norm_coeff_dict = {}

    def forward(self, input_image, flow):
        '''
        input_image: (n, 1, h, w) or (n, 1, d, h, w)
        flow: (n, 2, h, w) or (n, 3, d, h, w)

        return:
            warped moving image, (n, 1, h, w) or (n, 1, d, h, w)
        '''
        img_shape = input_image.shape[2:]
        if img_shape in self.grid_dict:
            grid = self.grid_dict[img_shape]
            norm_coeff = self.norm_coeff_dict[img_shape]
        else:
            grids = torch.meshgrid([torch.arange(0, s) for s in img_shape])
            # grid = torch.stack(grids)
            grid = torch.stack(grids[::-1],
                               dim=0)  # 2 x h x w or 3 x d x h x w, the data in second dimension is in the order of [w, h, d]
            grid = torch.unsqueeze(grid, 0)
            grid = grid.to(dtype=flow.dtype, device=flow.device)
            norm_coeff = 2. / (torch.tensor(img_shape[::-1], dtype=flow.dtype,
                                            device=flow.device) - 1.)  # the coefficients to map image coordinates to [-1, 1]
            self.grid_dict[img_shape] = grid
            self.norm_coeff_dict[img_shape] = norm_coeff

        new_grid = grid + flow

        if self.dim == 2:
            new_grid = new_grid.permute(0, 2, 3, 1)  # n x h x w x 2
        elif self.dim == 3:
            new_grid = new_grid.permute(0, 2, 3, 4, 1)  # n x d x h x w x 3
            # new_grid = new_grid[..., [2, 1, 0]]

        if len(input_image) != len(new_grid):
            # make the image shape compatable by broadcasting
            input_image += torch.zeros_like(new_grid)
            new_grid += torch.zeros_like(input_image)

        warped_input_img = F.grid_sample(input_image, new_grid * norm_coeff - 1., mode='bilinear', align_corners=True,
                                         padding_mode='border')
        return warped_input_img
