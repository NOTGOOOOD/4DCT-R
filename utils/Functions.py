import os

import numpy as np
import torch.utils.data as Data
# import nibabel as nib
import torch
import itertools
import torch.nn as nn
from torch.utils import data as Data

from utils.datagenerators import Dataset
from utils.losses import neg_Jdet_loss, smoothloss
from utils.metric import MSE


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


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


def validation_lapirn(args, model, imgshape, loss_similarity, ori_shape):
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

    grid = generate_grid_unit(imgshape)
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

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(flow.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = neg_Jdet_loss(F_X_Y_norm, grid)

            # reg2 - use velocity
            _, _, z, y, x = flow.shape
            flow[:, 2, :, :, :] = flow[:, 2, :, :, :] * (z - 1)
            flow[:, 1, :, :, :] = flow[:, 1, :, :, :] * (y - 1)
            flow[:, 0, :, :, :] = flow[:, 0, :, :, :] * (x - 1)
            loss_regulation = smoothloss(flow)

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