import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import scipy.ndimage
import numpy as np

torch.backends.cudnn.deterministic = True


class NCC(nn.Module):
    '''
    Calculate local normalized cross-correlation coefficient between tow images.

    Parameters
    ----------
    dim : int
        Dimension of the input images.
    windows_size : int
        Side length of the square window to calculate the local NCC.
    '''

    def __init__(self, dim, windows_size=11):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.num_stab_const = 1e-4  # numerical stability constant
        self.windows_size = windows_size
        self.pad = windows_size // 2
        self.window_volume = windows_size ** self.dim
        if self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d

    def forward(self, y_pred, y_true):
        '''
        Parameters
        ----------
        I and J : (n, 1, h, w) or (n, 1, d, h, w)
            Torch tensor of same shape. The number of image in the first dimension can be different, in which broadcasting will be used.
        windows_size : int
            Side length of the square window to calculate the local NCC.

        Returns
        -------
        NCC : scalar
            Average local normalized cross-correlation coefficient.
        '''

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.windows_size is None else self.windows_size

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = self.windows_size
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
        # try:
        #     I_sum = self.conv(I, self.sum_filter, padding=self.pad)
        # except:
        #     self.sum_filter = torch.ones([1, 1] + [self.windows_size, ] * self.dim, dtype=I.dtype, device=I.device)
        #     I_sum = self.conv(I, self.sum_filter, padding=self.pad)
        #
        # J_sum = self.conv(J, self.sum_filter, padding=self.pad)  # (n, 1, h, w) or (n, 1, d, h, w)
        # I2_sum = self.conv(I * I, self.sum_filter, padding=self.pad)
        # J2_sum = self.conv(J * J, self.sum_filter, padding=self.pad)
        # IJ_sum = self.conv(I * J, self.sum_filter, padding=self.pad)
        #
        # cross = torch.clamp(IJ_sum - I_sum * J_sum / self.window_volume, min=self.num_stab_const)
        # I_var = torch.clamp(I2_sum - I_sum ** 2 / self.window_volume, min=self.num_stab_const)
        # J_var = torch.clamp(J2_sum - J_sum ** 2 / self.window_volume, min=self.num_stab_const)
        #
        # # cc = cross / ((I_var * J_var) ** 0.5)
        # cc = cross * cross / (I_var * J_var + 1e-5)
        # return -torch.mean(cc)


def smooth_loss(disp, image):
    '''
    Calculate the smooth loss. Return mean of absolute or squared of the forward difference of  flow field.

    Parameters
    ----------
    disp : (n, 2, h, w) or (n, 3, d, h, w)
        displacement field

    image : (n, 1, d, h, w) or (1, 1, d, h, w)

    '''

    image_shape = disp.shape
    dim = len(image_shape[2:])

    d_disp = torch.zeros((image_shape[0], dim) + tuple(image_shape[1:]), dtype=disp.dtype, device=disp.device)
    d_image = torch.zeros((image_shape[0], dim) + tuple(image_shape[1:]), dtype=disp.dtype, device=disp.device)

    # forward difference
    if dim == 2:
        d_disp[:, 1, :, :-1, :] = (disp[:, :, 1:, :] - disp[:, :, :-1, :])
        d_disp[:, 0, :, :, :-1] = (disp[:, :, :, 1:] - disp[:, :, :, :-1])
        d_image[:, 1, :, :-1, :] = (image[:, :, 1:, :] - image[:, :, :-1, :])
        d_image[:, 0, :, :, :-1] = (image[:, :, :, 1:] - image[:, :, :, :-1])

    elif dim == 3:
        d_disp[:, 2, :, :-1, :, :] = (disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :])
        d_disp[:, 1, :, :, :-1, :] = (disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :])
        d_disp[:, 0, :, :, :, :-1] = (disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1])

        d_image[:, 2, :, :-1, :, :] = (image[:, :, 1:, :, :] - image[:, :, :-1, :, :])
        d_image[:, 1, :, :, :-1, :] = (image[:, :, :, 1:, :] - image[:, :, :, :-1, :])
        d_image[:, 0, :, :, :, :-1] = (image[:, :, :, :, 1:] - image[:, :, :, :, :-1])

    loss = torch.mean(torch.sum(torch.abs(d_disp), dim=2, keepdims=True) * torch.exp(-torch.abs(d_image)))

    return loss
