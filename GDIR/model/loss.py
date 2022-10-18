import torch
import torch.nn.functional as F
import torch.nn as nn
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

    def __init__(self, dim, windows_size=9):
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

    def forward(self, I, J):
        '''
        Parameters
        ----------
        I and J : (n, 1, h, w) or (n, 1, d, h, w)
            Torch tensor of same shape. The number of image in the first dimension can be different, in which broadcasting will be used.
        I: y_pred warped image
        J: y_true implict template
        Returns
        -------
        NCC : scalar
            Average local normalized cross-correlation coefficient.
        '''
        try:
            I_sum = self.conv(I, self.sum_filter, padding=self.pad, stride=(1, 1, 1))
        except:
            self.sum_filter = torch.ones([1, 1] + [self.windows_size, ] * self.dim, dtype=I.dtype, device=I.device)
            I_sum = self.conv(I, self.sum_filter, padding=self.pad, stride=(1, 1, 1))

        J_sum = self.conv(J, self.sum_filter, padding=self.pad)  # (n, 1, h, w) or (n, 1, d, h, w)
        I2_sum = self.conv(I * I, self.sum_filter, padding=self.pad, stride=(1, 1, 1))
        J2_sum = self.conv(J * J, self.sum_filter, padding=self.pad, stride=(1, 1, 1))
        IJ_sum = self.conv(I * J, self.sum_filter, padding=self.pad, stride=(1, 1, 1))

        E_I = I_sum / self.window_volume
        E_J = J_sum / self.window_volume

        # cross = torch.clamp(IJ_sum - I_sum * J_sum / self.window_volume, min=self.num_stab_const)
        cross = IJ_sum - I_sum * E_J - J_sum * E_I + E_I * E_J * self.window_volume

        # I_var = torch.clamp(I2_sum - I_sum ** 2 / self.window_volume, min=self.num_stab_const)
        # J_var = torch.clamp(J2_sum - J_sum ** 2 / self.window_volume, min=self.num_stab_const)

        I_var = I2_sum - 2 * E_I * I_sum + E_I * E_I * self.window_volume
        J_var = J2_sum - 2 * E_J * J_sum + E_J * E_J * self.window_volume

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


def gradient(data):
    dy = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]
    dx = data[:, :, :, 1:, :] - data[:, :, :, :-1, :]
    dz = data[:, :, :, :, 1:] - data[:, :, :, :, :-1]
    return dx, dy, dz


def smooth_loss(disp, image):
    '''
    Calculate the smooth loss. Return mean of absolute or squared of the forward difference of flow field.

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

    #     img_dx, img_dy, img_dz = gradient(image)
    #     flow_dx, flow_dy, flow_dz = gradient(disp)
    #
    #     loss_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True)) * torch.abs(flow_dx) / 2.
    #     loss_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True)) * torch.abs(flow_dy) / 2.
    #     loss_z = torch.exp(-torch.mean(torch.abs(img_dz), 1, keepdim=True)) * torch.abs(flow_dz) / 2.
    #
    # loss_new = torch.mean(loss_x) / 3. + torch.mean(loss_y) / 3. + torch.mean(loss_z) / 3.

    loss = torch.mean(torch.sum(torch.abs(d_disp), dim=2, keepdim=True) * torch.exp(-torch.abs(d_image)))

    return loss


def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def DSC(target, pred):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def dice(y_true, y_pred):
    ndims = len(list(y_pred.size())) - 2
    vol_axes = list(range(2, ndims + 2))
    top = 2 * (y_true * y_pred).sum(dim=vol_axes)
    bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
    dice = torch.mean(top / bottom)
    return -dice


if __name__ == "__main__":
    ncc_loss = NCC(3, 9)
    # a1 = np.random.rand(10, 1, 90, 150, 150)
    # a2 = np.random.rand(1, 1, 90, 150, 150)
    # a1 = torch.Tensor(a1)
    # a2 = torch.Tensor(a2)
    # loss = ncc_loss(a1, a2)
    # print(loss)

    # import cv2
    #
    # im1_array = cv2.imread("1890_warped_Case4_T0_44_slice.png", cv2.IMREAD_GRAYSCALE).astype("float32")
    # im2_array = cv2.imread("1890_warped_Case4_T0_44_slice.png", cv2.IMREAD_GRAYSCALE).astype("float32")
    # im1_array /= 255.
    # im2_array /= 255.
    #
    # im1_array = torch.Tensor(im1_array)
    # im2_array = torch.Tensor(im2_array)
    #
    # im1_array = torch.unsqueeze(torch.unsqueeze(im1_array, 0), 0)
    # im2_array = torch.unsqueeze(torch.unsqueeze(im2_array, 0), 0)
    # loss = ncc_loss(im1_array, im2_array)
    #
    # print(loss.item())

    import utils.utilize as ut
    import os
    from GDIR.process.processing import data_standardization_0_255

    project_folder = ut.get_project_path("4DCT").split("4DCT")[0]
    img_path = os.path.join(project_folder, f'datasets/dirlab/Case4Pack/Images')
    for file_name in os.listdir(img_path):
        file_path = os.path.join(img_path, file_name)
        file = np.memmap(file_path, dtype=np.float16, mode='r')
        file = file.astype('float32')
        file = data_standardization_0_255(file).reshape(99, 256, 256)
        im1_array = torch.Tensor(file)
        im1_array = torch.unsqueeze(torch.unsqueeze(im1_array, 0), 0)
        min = torch.min(im1_array)
        max = torch.max(im1_array)
        im2_array = im1_array.clone()
        loss = ncc_loss(im1_array, im2_array)

        print(loss.item())
