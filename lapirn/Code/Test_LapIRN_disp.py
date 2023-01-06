import os
import numpy as np
import torch
import torch.utils.data as Data
import SimpleITK as sitk

from Functions import generate_grid_unit, transform_unit_flow_to_flow
from miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3, SpatialTransform_unit, \
    multi_resolution_NCC, neg_Jdet_loss
from utils.utilize import load_landmarks, save_image
from utils.config import get_args
from utils.metric import calc_tre, MSE, landmark_loss
from utils.losses import NCC
from utils.datagenerators import TestDataset, Dataset


def validation(args, model, imgshape, loss_similarity, step):
    # range_flow = 0.4

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

    grid = generate_grid_unit([args.size] * 3)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    scale_factor = args.size / imgshape[0]
    upsample = torch.nn.Upsample(scale_factor=scale_factor, mode="trilinear")
    with torch.no_grad():
        model.eval()  # m_name = "{}_affine.nii.gz".format(moving[1][0][:13])
        losses = []
        for batch, (moving, fixed) in enumerate(val_loader):
            input_moving = moving[0].to('cuda').float()
            input_fixed = fixed[0].to('cuda').float()
            pred = model(input_moving, input_fixed)

            disp_up = pred[0]
            if scale_factor > 1:
                disp_up = upsample(pred[0])

            X_Y_up = transform(input_moving, disp_up.permute(0, 2, 3, 4, 1), grid)

            # F_X_Y_cpu = F_X_Y.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
            # F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

            mse_loss = MSE(X_Y_up, input_fixed)
            # ncc_loss = loss_similarity(X_Y, Y_4x)
            ncc_loss_ori = loss_similarity(X_Y_up, input_fixed)
            losses.append([ncc_loss_ori.item(), mse_loss.item()])

            # save_flow(F_X_Y_cpu, args.output_dir + '/warpped_flow.nii.gz')
            # save_img(X_Y, args.output_dir + '/warpped_moving.nii.gz')
            # m_name = "{}_warped.nii.gz".format(moving[1][0].split('.nii')[0])
            # save_img(X_Y, args.output_dir + '/' + file_name + '_warpped_moving.nii.gz')
            # save_image(X_Y, input_fixed, args.output_dir, m_name)
            if batch == 0:
                m_name = '{0}_{1}.nii.gz'.format(imgshape[0], step)
                save_image(pred[1], input_fixed, args.output_dir, m_name)
                m_name = '{0}_{1}_up.nii.gz'.format(imgshape[0], step)
                save_image(X_Y_up, input_fixed, args.output_dir, m_name)

        mean_loss = np.mean(losses, 0)
        return mean_loss[0], mean_loss[1]


def test_single(args):
    range_flow = 0.4

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    landmark_list = load_landmarks(args.landmark_dir)
    fixed_folder = os.path.join(args.test_dir, 'fixed')
    moving_folder = os.path.join(args.test_dir, 'moving')
    f_img_file_list = sorted([os.path.join(fixed_folder, file_name) for file_name in os.listdir(fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_img_file_list = sorted([os.path.join(moving_folder, file_name) for file_name in os.listdir(moving_folder) if
                              file_name.lower().endswith('.gz')])

    test_dataset = TestDataset(moving_files=m_img_file_list, fixed_files=f_img_file_list, landmark_files=landmark_list)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        losses = []
        for batch, (moving, fixed, landmarks, img_name) in enumerate(test_loader):
            moving_img = moving.to(args.device).float()
            fixed_img = fixed.to(args.device).float()
            landmarks00 = landmarks['landmark_00'].squeeze().cuda()
            landmarks50 = landmarks['landmark_50'].squeeze().cuda()

            imgshape = fixed_img.shape[2:]

            imgshape_4 = (imgshape[0] / 4, imgshape[1] / 4, imgshape[2] / 4)
            imgshape_2 = (imgshape[0] / 2, imgshape[1] / 2, imgshape[2] / 2)

            model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, args.initial_channels, is_train=True,
                                                                     imgshape=imgshape_4,
                                                                     range_flow=range_flow).cuda()
            model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, args.initial_channels, is_train=True,
                                                                     imgshape=imgshape_2,
                                                                     range_flow=range_flow,
                                                                     model_lvl1=model_lvl1).cuda()

            model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(2, 3, args.initial_channels, is_train=False,
                                                                imgshape=imgshape,
                                                                range_flow=range_flow, model_lvl2=model_lvl2).cuda()

            transform = SpatialTransform_unit().cuda()

            model.load_state_dict(torch.load(args.checkpoint_path))
            model.eval()
            transform.eval()
            grid = generate_grid_unit(imgshape)
            grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

            F_X_Y = model(moving_img, fixed_img)  # nibabel: b,c,w,h,d;simpleitk b,c,d,h,w

            X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid)

            F_X_Y_cpu = F_X_Y[0, :, :, :, :]
            F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

            Jac = neg_Jdet_loss(F_X_Y_cpu.unsqueeze(0).permute(0, 2, 3, 4, 1), grid)

            crop_range = args.dirlab_cfg[batch + 1]['crop_range']

            # TRE
            # _mean, _std = calc_tre(F_X_Y[0], landmarks00 - torch.tensor(
            #     [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 1, 3).cuda(),
            #                        torch.tensor(landmarks['disp_00_50']).squeeze(),
            #                        args.dirlab_cfg[batch + 1]['pixel_spacing'])
            # _mean, _std = calc_tre(flow_hr, landmarks00 - torch.tensor(
            #     [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 1, 3).cuda(),
            #                        landmarks['disp_affine'].squeeze(), args.dirlab_cfg[index]['pixel_spacing'])

            # MSE
            _mse = MSE(fixed_img, X_Y)
            _mean, _std = landmark_loss(F_X_Y[0], landmarks00 - torch.tensor(
                [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
                                        landmarks50 - torch.tensor(
                                            [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1,
                                                                                                                  3).cuda(),
                                        args.dirlab_cfg[batch + 1]['pixel_spacing'])

            losses.append([_mean.item(), _std.item(), _mse.item()])
            print('case=%d after warped, TRE=%.5f+-%.5f MSE=%.5f Jac=%.6f' % (batch + 1, _mean.item(), _std.item(), _mse.item(), Jac.item()))

            # Save DVF
            # b,3,d,h,w-> d,h,w,3    (dhw or whd) depend on the shape of image
            m2f_name = img_name[0][:13] + '_warpped_flow.nii.gz'
            save_image(torch.permute(F_X_Y_cpu, (1, 2, 3, 0)), fixed_img[0], args.output_dir,
                       m2f_name)
            m_name = "{}_warped_lapirn.nii.gz".format(img_name[0][:13])
            # save_img(X_Y, args.output_dir + '/' + file_name + '_warpped_moving.nii.gz')
            save_image(X_Y, fixed_img, args.output_dir, m_name)

    # # respectively
    # losses = []
    # for i in range(len(f_img_file_list)):
    #     file_name = m_img_file_list[i].split('moving\\')[1] if platform.system().lower() == 'windows' else \
    #         m_img_file_list[i].split('moving/')[1]
    #     file_name = file_name[:13]
    #
    #     fixed_img = load_4D(f_img_file_list[i])
    #     moving_img = load_4D(m_img_file_list[i])
    #     fixed_img = torch.from_numpy(fixed_img).float().to(args.device).unsqueeze(dim=0)
    #     moving_img = torch.from_numpy(moving_img).float().to(args.device).unsqueeze(dim=0)
    #
    #     fixed_img = data_standardization_0_n(1, fixed_img)
    #     moving_img = data_standardization_0_n(1, moving_img)
    #
    #     landmarks = landmark_list[i]
    #
    #     imgshape = fixed_img.shape[2:]
    #
    #     imgshape_4 = (imgshape[0] / 4, imgshape[1] / 4, imgshape[2] / 4)
    #     imgshape_2 = (imgshape[0] / 2, imgshape[1] / 2, imgshape[2] / 2)
    #
    #     model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, args.initial_channels, is_train=True,
    #                                                              imgshape=imgshape_4,
    #                                                              range_flow=range_flow).cuda()
    #     model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, args.initial_channels, is_train=True,
    #                                                              imgshape=imgshape_2,
    #                                                              range_flow=range_flow, model_lvl1=model_lvl1).cuda()
    #
    #     model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(2, 3, args.initial_channels, is_train=False,
    #                                                         imgshape=imgshape,
    #                                                         range_flow=range_flow, model_lvl2=model_lvl2).cuda()
    #
    #     transform = SpatialTransform_unit().cuda()
    #
    #     model.load_state_dict(torch.load(args.checkpoint_path))
    #     model.eval()
    #     transform.eval()
    #
    #     grid = generate_grid_unit(imgshape)
    #     grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()
    #
    #     with torch.no_grad():
    #         F_X_Y = model(moving_img, fixed_img)    # nibabel: b,c,w,h,d
    #
    #         X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]
    #
    #         F_X_Y_cpu = F_X_Y.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
    #         F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)
    #
    #         crop_range = args.dirlab_cfg[i + 1]['crop_range']
    #
    #         landmarks00 = torch.tensor(landmarks['landmark_00']).cuda()
    #
    #         # TRE
    #         _mean, _std = calc_tre(torch.tensor(F_X_Y_cpu).permute(3, 2, 1, 0), landmarks00 - torch.tensor(
    #             [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 1, 3).cuda(),
    #                                torch.tensor(landmarks['disp_00_50']).squeeze(),
    #                                args.dirlab_cfg[i + 1]['pixel_spacing'])
    #         # _mean, _std = calc_tre(flow_hr, landmarks00 - torch.tensor(
    #         #     [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 1, 3).cuda(),
    #         #                        landmarks['disp_affine'].squeeze(), args.dirlab_cfg[index]['pixel_spacing'])
    #
    #         # MSE
    #         _mse = MSE(fixed_img.squeeze(), torch.tensor(X_Y))
    #         # _mean, _std = landmark_loss(flow_hr, landmarks00 - torch.tensor(
    #         #     [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 1, 3).cuda(),
    #         #                             landmarks50 - torch.tensor(
    #         #                                 [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 1,
    #         #                                                                                                       3).cuda(),
    #         #                             args.dirlab_cfg[index]['pixel_spacing'])
    #
    #         losses.append([_mean.item(), _std.item(), _mse.item()])
    #         print('case=%d after warped, TRE=%.5f+-%.5f MSE=%.5f' % (i + 1, _mean.item(), _std.item(), _mse.item()))
    #
    #         # save_flow(F_X_Y_cpu, args.output_dir + '/' + file_name + '_warpped_flow.nii.gz')
    #         save_img(X_Y, args.output_dir + '/' + file_name + '_warpped_moving.nii.gz')
    #
    #     print("Finished")


if __name__ == '__main__':
    args = get_args()
    test_single(args)
    # validation(args)
