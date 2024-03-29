import os

import numpy as np
import torch
import torch.nn as nn
from torch import nn as nn
from torch.nn import functional as nnf
from torch.utils import data as Data
import torch.nn.functional as F
import copy

from midir.model.loss import l2reg_loss
from midir.model.transformation import CubicBSplineFFDTransform, warp
from utils.datagenerators import Dataset, build_dataloader_dirlab
from utils.losses import smoothloss, neg_Jdet_loss, bending_energy_loss, Grad
from utils.metric import MSE, jacobian_determinant, landmark_loss, NCC as mtNCC, SSIM
from utils.utilize import save_image


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid

class AdaptiveSpatialTransformer(nn.Module):
    # 2D or 3d spatial transformer network to calculate the warped moving image
    # Adaptive any shape of input image
    def __init__(self, dim=3):
        super(AdaptiveSpatialTransformer, self).__init__()
        self.dim = dim

    def forward(self, input_image, flow, grid):
        '''
        input_image: (n, 1, h, w) or (n, 1, d, h, w)
        flow: (n, h, w, 2) or (n, d, h, w, 3)

        return:
            warped moving image, (n, 1, h, w) or (n, 1, d, h, w)
        '''
        new_grid = grid + flow

        if len(input_image) != len(new_grid):
            # make the image shape compatable by broadcasting
            input_image += torch.zeros_like(new_grid)
            new_grid += torch.zeros_like(input_image)

        # warped_input_img = torch.nn.functional.grid_sample(input_image, new_grid, mode='bilinear',
        #                                                    align_corners=True,
        #                                                    padding_mode='border')
        warped_input_img = torch.nn.functional.grid_sample(input_image, new_grid, mode='bilinear',
                                                           align_corners=True,
                                                           padding_mode='border')
        return warped_input_img


class SpatialTransform_unit(nn.Module):
    def __init__(self):
        super(SpatialTransform_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear', padding_mode="border",
                                               align_corners=True)
        return flow


def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def transform_unit_flow_to_flow(flow):
    _, z, y, x = flow.shape

    flow[2, :, :, :] = flow[2, :, :, :] * (z - 1) / 2
    flow[1, :, :, :] = flow[1, :, :, :] * (y - 1) / 2
    flow[0, :, :, :] = flow[0, :, :, :] * (x - 1) / 2
    # z, y, x, _ = flow.shape
    # flow[:, :, :, 2] = flow[:, :, :, 2] * (z-1)/2
    # flow[:, :, :, 1] = flow[:, :, :, 1] * (y-1)/2
    # flow[:, :, :, 0] = flow[:, :, :, 0] * (x-1)/2

    return flow


def get_loss(grid_class, loss_similarity, loss_Jdet, loss_smooth, F_X2Y, X2Y, Y):
    """

    get train loss
    Parameters
    ----------
    loss_similarity function
    loss_Jdet       function
    loss_smooth     function
    F_X2Y           flow of x to y
    X2Y             warped image (x to y)
    Y               fixed image

    Returns
    -------
    loss_multiNCC, loss_Jacobian, loss_regulation

    """

    # 3 level deep supervision NCC
    loss_multiNCC = loss_similarity(X2Y, Y)

    F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X2Y.permute(0, 2, 3, 4, 1).clone())

    grid_4 = grid_class.get_grid(img_shape=F_X_Y_norm.shape[1:4])

    loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

    # reg2 - use velocity
    _, _, z, y, x = F_X2Y.shape
    F_X2Y[:, 2, :, :, :] = F_X2Y[:, 2, :, :, :] * (z - 1)
    F_X2Y[:, 1, :, :, :] = F_X2Y[:, 1, :, :, :] * (y - 1)
    F_X2Y[:, 0, :, :, :] = F_X2Y[:, 0, :, :, :] * (x - 1)
    loss_regulation = loss_smooth(F_X2Y)

    return loss_multiNCC, loss_Jacobian, loss_regulation


def transform_unit_flow_to_flow_cuda(flow):
    b, z, y, x, c = flow.shape
    flow[:, :, :, :, 0] = flow[:, :, :, :, 0] * (x - 1) / 2
    flow[:, :, :, :, 1] = flow[:, :, :, :, 1] * (y - 1) / 2
    flow[:, :, :, :, 2] = flow[:, :, :, :, 2] * (z - 1) / 2
    return flow


# def load_4D(name):
#     X = nib.load(name)
#     X = X.get_fdata()
#     X = np.reshape(X, (1,) + X.shape)
#     return X
#
#
# def load_5D(name):
#     X = fixed_nii = nib.load(name)
#     X = X.get_fdata()
#     X = np.reshape(X, (1,) + (1,) + X.shape)
#     return X


def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)

    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


def validation_lapirn_ori(args, model, loss_similarity, grid_class, scale_factor):
    max_smooth = 10.
    antifold = args.antifold

    fixed_folder = os.path.join(args.val_dir, 'fixed')
    moving_folder = os.path.join(args.val_dir, 'moving')
    f_img_file_list = sorted([os.path.join(fixed_folder, file_name) for file_name in os.listdir(fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_img_file_list = sorted([os.path.join(moving_folder, file_name) for file_name in os.listdir(moving_folder) if
                              file_name.lower().endswith('.gz')])

    val_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    val_loader = Data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    transform = AdaptiveSpatialTransformer()
    with torch.no_grad():
        model.eval()  # m_name = "{}_affine.nii.gz".format(moving[1][0][:13])
        losses = []
        for batch, (moving, fixed) in enumerate(val_loader):
            input_moving = moving[0].to('cuda').float()
            input_fixed = fixed[0].to('cuda').float()

            reg_code = torch.rand(1, dtype=input_moving.dtype, device=input_moving.device).unsqueeze(dim=0)

            pred = model(input_moving, input_fixed, reg_code)
            F_X_Y = pred[0]
            if scale_factor > 1:
                F_X_Y = F.interpolate(pred[0], input_moving.shape[2:], mode='trilinear',
                                      align_corners=True, recompute_scale_factor=False)

            loss_multiNCC = loss_similarity(pred[1], pred[2])

            X_Y_up = transform(input_moving, F_X_Y.permute(0, 2, 3, 4, 1),
                               grid_class.get_grid(input_moving.shape[2:], True))
            mse_loss = MSE(X_Y_up, input_fixed)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = neg_Jdet_loss(F_X_Y_norm, grid_class.get_grid(input_moving.shape[2:]))

            _, _, x, y, z = F_X_Y.shape
            norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0, 0] = (z - 1)
            norm_vector[0, 1, 0, 0, 0] = (y - 1)
            norm_vector[0, 2, 0, 0, 0] = (x - 1)
            loss_regulation = smoothloss(F_X_Y * norm_vector)

            smo_weight = reg_code * max_smooth
            loss = loss_multiNCC + antifold * loss_Jacobian + smo_weight * loss_regulation

            losses.append([loss_multiNCC.item(), mse_loss.item(), loss_Jacobian.item(), loss.item()])

        mean_loss = np.mean(losses, 0)
        return mean_loss[0], mean_loss[1], mean_loss[2], mean_loss[3]


def validation_midir(args, model, imgshape, loss_similarity):
    fixed_folder = os.path.join(args.val_dir, 'fixed')
    moving_folder = os.path.join(args.val_dir, 'moving')
    f_img_file_list = sorted([os.path.join(fixed_folder, file_name) for file_name in os.listdir(fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_img_file_list = sorted([os.path.join(moving_folder, file_name) for file_name in os.listdir(moving_folder) if
                              file_name.lower().endswith('.gz')])

    val_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    val_loader = Data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # transform = CubicBSplineFFDTransform(ndim=3, img_size=imgshape, cps=(4, 4, 4), svf=True
    #                                      , svf_steps=7
    #                                      , svf_scale=1)
    transform = SpatialTransformer()
    reg_weight = 0.1
    with torch.no_grad():
        model.eval()  # m_name = "{}_affine.nii.gz".format(moving[1][0][:13])
        losses = []
        for batch, (moving, fixed) in enumerate(val_loader):
            input_moving = moving[0].to('cuda').float()
            input_fixed = fixed[0].to('cuda').float()

            disp = model(input_fixed, input_moving)
            # flow, disp = transform(svf)
            # wapred_x = warp(input_moving, disp)
            wapred_x = transform(input_moving, disp)
            mse_loss = MSE(wapred_x, input_fixed)
            loss_ncc = loss_similarity(wapred_x, input_fixed)
            loss_reg = l2reg_loss(disp)

            loss = loss_ncc + loss_reg * reg_weight
            losses.append([loss_ncc.item(), mse_loss.item(), loss_reg.item(), loss.item()])

        mean_loss = np.mean(losses, 0)
        return mean_loss[0], mean_loss[1], mean_loss[2], mean_loss[3]

@torch.no_grad()
def validation_vm(args, model, loss_similarity):
    val_loader = build_dataloader_dirlab(args, mode='val')

    transform = SpatialTransform_unit().cuda()
    transform.eval()

    model.eval()  # m_name = "{}_affine.nii.gz".format(moving[1][0][:13])
    losses = []
    for batch, (moving, fixed) in enumerate(val_loader):
        input_moving = moving[0].to('cuda').float()
        input_fixed = fixed[0].to('cuda').float()

        pred = model(input_moving, input_fixed, True)
        warped_image, flow = pred['warped_img'], pred['flow']
        mse_loss = MSE(warped_image, input_fixed)
        ncc_loss_ori = loss_similarity(warped_image, input_fixed)

        loss_reg = Grad('l2', loss_mult=2).loss(None, flow)
        loss_sum = ncc_loss_ori + args.alpha * loss_reg

        losses.append([ncc_loss_ori.item(), mse_loss.item(), loss_reg.item(), loss_sum.item()])

    mean_loss = np.mean(losses, 0)
    return mean_loss[0], mean_loss[1], mean_loss[2], mean_loss[3]


def validation_lapirn_bak(args, model, imgshape, loss_similarity, ori_shape):
    fixed_folder = os.path.join(args.val_dir, 'fixed')
    moving_folder = os.path.join(args.val_dir, 'moving')
    f_img_file_list = sorted([os.path.join(fixed_folder, file_name) for file_name in os.listdir(fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_img_file_list = sorted([os.path.join(moving_folder, file_name) for file_name in os.listdir(moving_folder) if
                              file_name.lower().endswith('.gz')])

    val_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    val_loader = Data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    transform = SpatialTransform_unit().cuda()
    transform.eval()

    grid = generate_grid_unit(ori_shape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    scale_factor = ori_shape[0] / imgshape[0]
    upsample = torch.nn.Upsample(scale_factor=scale_factor, mode="trilinear")
    with torch.no_grad():
        model.eval()  # m_name = "{}_affine.nii.gz".format(moving[1][0][:13])
        losses = []
        for batch, (moving, fixed) in enumerate(val_loader):
            input_moving = moving[0].to('cuda').float()
            input_fixed = fixed[0].to('cuda').float()
            pred = model(input_moving, input_fixed)

            F_X_Y = pred[0]
            if scale_factor > 1:
                F_X_Y = upsample(pred[0])

            X_Y_up = transform(input_moving, F_X_Y.permute(0, 2, 3, 4, 1), grid)
            mse_loss = MSE(X_Y_up, input_fixed)
            ncc_loss_ori = loss_similarity(X_Y_up, input_fixed)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = neg_Jdet_loss(F_X_Y_norm, grid)

            # reg2 - use velocity
            _, _, z, y, x = F_X_Y.shape
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (z - 1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (x - 1)
            loss_regulation = smoothloss(F_X_Y)

            loss_sum = ncc_loss_ori + args.antifold * loss_Jacobian + args.smooth * loss_regulation

            losses.append([ncc_loss_ori.item(), mse_loss.item(), loss_Jacobian.item(), loss_sum.item()])
            # save_flow(F_X_Y_cpu, args.output_dir + '/warpped_flow.nii.gz')
            # save_img(X_Y, args.output_dir + '/warpped_moving.nii.gz')
            # m_name = "{}_warped.nii.gz".format(moving[1][0].split('.nii')[0])
            # save_img(X_Y, args.output_dir + '/' + file_name + '_warpped_moving.nii.gz')
            # save_image(X_Y, input_fixed, args.output_dir, m_name)
            # if batch == 0:
            #     m_name = '{0}_{1}.nii.gz'.format(imgshape[0], step)
            #     save_image(pred[1], input_fixed, args.output_dir, m_name)
            #     m_name = '{0}_{1}_up.nii.gz'.format(imgshape[0], step)
            #     save_image(X_Y_up, input_fixed, args.output_dir, m_name)

        mean_loss = np.mean(losses, 0)
        return mean_loss[0], mean_loss[1], mean_loss[2], mean_loss[3]


class Grid():
    """
        generate grid set
    """

    def __init__(self):
        self.grid_dict = {}
        self.grid_unit_dict = {}
        # self.norm_coeff_dict = {}

    def get_grid(self, img_shape, is_norm=False):
        if is_norm and img_shape in self.grid_unit_dict:
            grid = self.grid_unit_dict[img_shape]
        elif is_norm == False and img_shape in self.grid_dict:
            grid = self.grid_dict[img_shape]
            # norm_coeff = self.norm_coeff_dict[img_shape]
        else:
            # grids = torch.meshgrid([torch.arange(0, s) for s in img_shape])
            # # grid = torch.stack(grids)
            # grid = torch.stack(grids[::-1],
            #                    dim=0)  # 2 x h x w or 3 x d x h x w, the data in second dimension is in the order of [w, h, d]
            # grid = torch.unsqueeze(grid, 0)
            # grid = grid.to(dtype=dtype, device=device)

            # norm_coeff = 2. / (torch.tensor(img_shape[::-1]) - 1.).cuda().float()  # the coefficients to map image coordinates to [-1, 1]
            if is_norm:
                grid = generate_grid_unit(img_shape)
            else:
                grid = generate_grid(img_shape)

            grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

            if is_norm:
                self.grid_unit_dict[img_shape] = grid
            else:
                self.grid_dict[img_shape] = grid
            # self.norm_coeff_dict[img_shape] = norm_coeff

        return grid

@torch.no_grad()
def test_dirlab(args, model, test_loader, norm=False, is_train=True, logging=None, is_save=False, suffix='', calc_tre=True):
    model.eval()
    losses = []
    # for batch, (moving, fixed, landmarks, img_name) in enumerate(test_loader):
    for batch, (data) in enumerate(test_loader):
        landmarks = None
        moving, fixed, img_name = data[0], data[1], data[2]
        moving_img = moving.to(args.device).float()
        fixed_img = fixed.to(args.device).float()
        pred = model(moving_img, fixed_img)
        flow = pred['flow']  # nibabel: b,c,w,h,d;simpleitk b,c,d,h,w
        warped_img = pred['warped_img']

        F_X_Y_norm = copy.deepcopy(flow)
        if norm:
            F_X_Y_norm = transform_unit_flow_to_flow_cuda(flow.permute(0, 2, 3, 4, 1).clone())
            F_X_Y_norm = F_X_Y_norm.permute(0, 4, 1, 2, 3)

        if calc_tre:
            landmarks = data[2]
            img_name = data[3]
            spacing = args.dirlab_cfg[batch + 1]['pixel_spacing']
            landmarks00 = landmarks['landmark_00'].squeeze().cuda()
            landmarks50 = landmarks['landmark_50'].squeeze().cuda()
            crop_range = args.dirlab_cfg[batch + 1]['crop_range']
            # TRE
            _mean, _std = landmark_loss(F_X_Y_norm[0],
                                        m_landmarks=landmarks00 - torch.tensor([crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1,3).cuda(),
                                        f_landmarks=landmarks50 - torch.tensor([crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1,3).cuda(),
                                        spacing=args.dirlab_cfg[batch + 1]['pixel_spacing'])
            _mean, _std = _mean.item(), _std.item()
        else:
            _mean, _std = 0, 0
            spacing = None

        ncc = mtNCC(fixed_img.cpu().detach().numpy(), warped_img.cpu().detach().numpy())

        # loss_Jacobian = neg_Jdet_loss(y_pred[1].permute(0, 2, 3, 4, 1), grid)
        jac = jacobian_determinant(warped_img[0].cpu().detach().numpy())

        # SSIM
        ssim = SSIM(fixed_img.cpu().detach().numpy()[0, 0], warped_img.cpu().detach().numpy()[0, 0])

        losses.append([_mean, _std, ncc.item(), ssim.item(), jac])
        if not is_train:
            print('case=%d after warped, TRE=%.2f+-%.2f Jac=%.6f ncc=%.6f ssim=%.6f' % (
                batch + 1, _mean, _std, jac, ncc.item(), ssim.item()))

        if is_save:
            # Save DVF
            # b,3,d,h,w-> d,h,w,3    (dhw or whd) depend on the shape of image
            m2f_name = '{}_warpped_flow_{}.nii.gz'.format(img_name[0][:13], suffix)
            save_image(flow[0].permute((1, 2, 3, 0)), args.output_dir, m2f_name, spacing=spacing)

            m_name = '{}_warpped_img_{}.nii.gz'.format(img_name[0][:13], suffix)
            save_image(warped_img.squeeze(), args.output_dir, m_name, spacing=spacing)

    mean_tre, mean_std, mean_ncc, mean_ssim, mean_jac = np.mean(losses, 0)

    print('mean TRE=%.2f+-%.2f NCC=%.6f SSIM=%.6f J=%.6f' % (
        mean_tre, mean_std, mean_ncc, mean_ssim, mean_jac))
    if logging is not None:
        logging.info('mean TRE=%.2f+-%.2f NCC=%.6f SSIM=%.6f J=%.6f' % (
        mean_tre, mean_std, mean_ncc, mean_ssim, mean_jac))

    return mean_tre


class SpatialTransformer(nn.Module):
    # 2D or 3d spatial transformer network to calculate the warped moving image

    def __init__(self, dim=3):
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

        warped_input_img = nnf.grid_sample(input_image, new_grid * norm_coeff - 1., mode='bilinear', align_corners=True,
                                         padding_mode='border')
        return warped_input_img