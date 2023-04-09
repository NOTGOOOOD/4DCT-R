import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as Data

from midir.model.loss import l2reg_loss
from midir.model.transformation import CubicBSplineFFDTransform, warp
from utils.datagenerators import Dataset
from utils.losses import smoothloss, neg_Jdet_loss, bending_energy_loss
from utils.metric import MSE, jacobian_determinant


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


def validation_ccregnet(args, model, loss_similarity, grid_class, scale_factor):
    fixed_folder = os.path.join(args.val_dir, 'fixed')
    moving_folder = os.path.join(args.val_dir, 'moving')
    f_img_file_list = sorted([os.path.join(fixed_folder, file_name) for file_name in os.listdir(fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_img_file_list = sorted([os.path.join(moving_folder, file_name) for file_name in os.listdir(moving_folder) if
                              file_name.lower().endswith('.gz')])

    val_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    val_loader = Data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    transform = AdaptiveSpatialTransformer()

    upsample = torch.nn.Upsample(scale_factor=scale_factor, mode="trilinear")
    with torch.no_grad():
        model.eval()  # m_name = "{}_affine.nii.gz".format(moving[1][0][:13])
        losses = []
        for batch, (moving, fixed) in enumerate(val_loader):
            input_moving = moving[0].to('cuda').float()
            input_fixed = fixed[0].to('cuda').float()
            pred = model(input_moving, input_fixed)
            F_X_Y = pred[0]

            grid = grid_class.get_grid(input_moving.shape[2:], True)

            if scale_factor > 1:
                F_X_Y = upsample(pred[0])

            X_Y_up = transform(input_moving, F_X_Y.permute(0, 2, 3, 4, 1), grid)
            mse_loss = MSE(X_Y_up, input_fixed)
            ncc_loss_ori = loss_similarity(X_Y_up, input_fixed)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = neg_Jdet_loss(F_X_Y_norm, grid)
            # loss_Jacobian = jacobian_determinant(F_X_Y[0].cpu().detach().numpy())

            # reg2 - use velocity
            _, _, z, y, x = F_X_Y.shape
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (z - 1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (x - 1)
            loss_regulation = smoothloss(F_X_Y)
            # loss_regulation = bending_energy_loss(F_X_Y)
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


def validation_lapirn_ori(args, model, imgshape, loss_similarity, ori_shape):
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

            reg_code = torch.rand(1, dtype=input_moving.dtype, device=input_moving.device).unsqueeze(dim=0)

            pred = model(input_moving, input_fixed, reg_code)
            F_X_Y = pred[0]
            if scale_factor > 1:
                F_X_Y = upsample(pred[0])

            loss_multiNCC = loss_similarity(pred[1], pred[2])

            X_Y_up = transform(input_moving, F_X_Y.permute(0, 2, 3, 4, 1), grid)
            mse_loss = MSE(X_Y_up, input_fixed)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = neg_Jdet_loss(F_X_Y_norm, grid)

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

    transform = CubicBSplineFFDTransform(ndim=3, img_size=imgshape, cps=(4, 4, 4), svf=True
                                         , svf_steps=7
                                         , svf_scale=1)
    reg_weight = 0.1
    with torch.no_grad():
        model.eval()  # m_name = "{}_affine.nii.gz".format(moving[1][0][:13])
        losses = []
        for batch, (moving, fixed) in enumerate(val_loader):
            input_moving = moving[0].to('cuda').float()
            input_fixed = fixed[0].to('cuda').float()

            svf = model(input_fixed, input_moving)
            flow, disp = transform(svf)
            wapred_x = warp(input_moving, disp)

            mse_loss = MSE(wapred_x, input_fixed)
            loss_ncc = loss_similarity(wapred_x, input_fixed)
            loss_reg = l2reg_loss(disp)

            grid = generate_grid(imgshape)
            grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

            loss_Jacobian = neg_Jdet_loss(disp.permute(0, 2, 3, 4, 1), grid)

            loss = loss_ncc + loss_reg * reg_weight

            losses.append([loss_ncc.item(), mse_loss.item(), loss_Jacobian.item(), loss.item()])

        mean_loss = np.mean(losses, 0)
        return mean_loss[0], mean_loss[1], mean_loss[2], mean_loss[3]


def validation_vm(args, model, imgshape, loss_similarity):
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

    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    with torch.no_grad():
        model.eval()  # m_name = "{}_affine.nii.gz".format(moving[1][0][:13])
        losses = []
        for batch, (moving, fixed) in enumerate(val_loader):
            input_moving = moving[0].to('cuda').float()
            input_fixed = fixed[0].to('cuda').float()

            warped_image, flow = model(input_moving, input_fixed, True)

            mse_loss = MSE(warped_image, input_fixed)
            ncc_loss_ori = loss_similarity(warped_image, input_fixed)

            # F_X_Y_norm = transform_unit_flow_to_flow_cuda(flow.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = neg_Jdet_loss(flow.permute(0, 2, 3, 4, 1), grid)

            # # reg2 - use velocity
            # _, _, z, y, x = flow.shape
            # flow[:, 2, :, :, :] = flow[:, 2, :, :, :] * (z - 1)
            # flow[:, 1, :, :, :] = flow[:, 1, :, :, :] * (y - 1)
            # flow[:, 0, :, :, :] = flow[:, 0, :, :, :] * (x - 1)
            # loss_regulation = smoothloss(flow)

            loss_sum = ncc_loss_ori + args.antifold * loss_Jacobian

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
