"""
*Preliminary* pytorch implementation.

Losses for VoxelMorph
"""

import math
import torch
import torch.nn as nn
import numpy as np
from utils.config import get_args
import torch.nn.functional as F

args = get_args()


class NCC_const(nn.Module):
    '''
    Calculate local normalized cross-correlation coefficient between tow images.
    Parameters
    ----------
    dim : int
        Dimension of the input images.
    win : int
        Side length of the square window to calculate the local NCC.
    '''

    def __init__(self, dim=3, win=11):
        super(NCC_const).__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.num_stab_const = 1e-4  # numerical stability constant

        self.win = win

        self.pad = win // 2
        self.window_volume = win ** self.dim
        if self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d

    def forward(self, I, J):
        '''
        Parameters
        ----------
        I and J : (n, 1, h, w) or (n, 1, d, h, w)
            Torch tensor of same shape. The number of image in the first dimension can be different, in which broadcasting will be used.
        win : int
            Side length of the square window to calculate the local NCC.

        Returns
        -------
        NCC : scalar
            Average local normalized cross-correlation coefficient.
        '''
        try:
            I_sum = self.conv(I, self.sum_filter, padding=self.pad)
        except:
            self.sum_filter = torch.ones([1, 1] + [self.win, ] * self.dim, dtype=I.dtype, device=I.device)
            I_sum = self.conv(I, self.sum_filter, padding=self.pad)

        J_sum = self.conv(J, self.sum_filter, padding=self.pad)  # (n, 1, h, w) or (n, 1, d, h, w)
        I2_sum = self.conv(I * I, self.sum_filter, padding=self.pad)
        J2_sum = self.conv(J * J, self.sum_filter, padding=self.pad)
        IJ_sum = self.conv(I * J, self.sum_filter, padding=self.pad)

        cross = torch.clamp(IJ_sum - I_sum * J_sum / self.window_volume, min=self.num_stab_const)
        I_var = torch.clamp(I2_sum - I_sum ** 2 / self.window_volume, min=self.num_stab_const)
        J_var = torch.clamp(J2_sum - J_sum ** 2 / self.window_volume, min=self.num_stab_const)

        cc = cross / ((I_var * J_var) ** 0.5)

        return -torch.mean(cc)


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=5, eps=1e-8):
        super(NCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win
        self.num_stab_const = 1e-4

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

        cross = torch.clamp(IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size, min=self.num_stab_const)
        I_var = torch.clamp(I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size,min=self.num_stab_const)
        J_var = torch.clamp(J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size,min=self.num_stab_const)

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


# 平滑正则损失
def gradient_loss(s, penalty='l2'):
    '''

    Parameters
    ----------
    s size of b*c*h*w*d 传对了吗
    penalty

    Returns
    -------

    '''
    dz = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dy = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dx = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    # 各个方向上每个像素上的平均梯度
    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def ncc_loss(I, J, win=None):
    '''
    输入大小是[B,C,D,W,H]格式的，在计算ncc时用卷积来实现指定窗口内求和
    '''
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]).to("cuda:{}".format(args.gpu))
    pad_no = math.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [pad_no] * ndims
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross


def cc_loss(x, y):
    # 根据互相关公式进行计算
    dim = [2, 3, 4]
    mean_x = torch.mean(x, dim, keepdim=True)
    mean_y = torch.mean(y, dim, keepdim=True)
    mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
    mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
    stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
    stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
    return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


def Get_Ja(flow):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3

    the expected input: displacement of shape(batch, H, W, D, channel)
    but the input dim: (batch, channel, D, H, W)
    '''
    displacement = np.transpose(flow, (0, 3, 4, 2, 1))  # b, c, D, H, W -> b, H, W, D, c
    D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    return D1 - D2 + D3


# def jacobian_determinant(disp):
#     """
#     jacobian determinant of a displacement field.
#     NB: to compute the spatial gradients, we use np.gradient.
#
#     Parameters:
#         disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
#               where vol_shape is of len nb_dims
#
#     Returns:
#         jacobian determinant (scalar)
#     """
#
#     # check inputs
#     volshape = disp.shape[:-1]
#     nb_dims = len(volshape)
#     assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
#
#     # compute grid
#     grid_lst = nd.volsize2ndgrid(volshape)
#     grid = np.stack(grid_lst, len(volshape))
#
#     # compute gradients
#     J = np.gradient(disp + grid)
#
#     # 3D glow
#     if nb_dims == 3:
#         dx = J[0]
#         dy = J[1]
#         dz = J[2]
#
#         # compute jacobian components
#         Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
#         Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
#         Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
#
#         return Jdet0 - Jdet1 + Jdet2
#
#     else:  # must be 2
#
#         dfdx = J[0]
#         dfdy = J[1]
#
#         return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def NJ_loss(ypred):
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    Neg_Jac = 0.5 * (torch.abs(Get_Ja(ypred)) - Get_Ja(ypred))
    return torch.sum(Neg_Jac)


def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    Jdet1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    Jdet2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def smoothloss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
    return (torch.mean(dx * dx) + torch.mean(dy * dy) + torch.mean(dz * dz)) / 3.0


def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)


class multi_resolution_NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None, eps=1e-5, scale=3):
        super(multi_resolution_NCC, self).__init__()
        self.num_scale = scale
        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC(win=win - (i * 2)))

    def forward(self, I, J):
        total_NCC = []

        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J)
            total_NCC.append(current_NCC / (2 ** i))

            I = nn.functional.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            J = nn.functional.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)

        return sum(total_NCC)
