import os
import numpy as np
import torch
import torch.utils.data as Data

from Functions import generate_grid_unit, transform_unit_flow_to_flow, transform_unit_flow_to_flow_cuda
from miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3, SpatialTransform_unit, \
    neg_Jdet_loss, smoothloss
from utils.utilize import load_landmarks, save_image
from utils.config import get_args
from utils.metric import calc_tre, MSE, landmark_loss
from utils.datagenerators import DirLabDataset, PatientDataset, Dataset


def validation(args, model, imgshape, loss_similarity, ori_shape):
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


def test_single(args, checkpoint, is_save=False):
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

            model.load_state_dict(torch.load(checkpoint))
            model.eval()
            transform.eval()
            grid = generate_grid_unit(imgshape)
            grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

            F_X_Y, lv1_out, lv2_out, lv3_out = model(moving_img, fixed_img)  # nibabel: b,c,w,h,d;simpleitk b,c,d,h,w

            # X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid)

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
            _mse = MSE(fixed_img, lv3_out)

            # # TRE
            # _mean, _std = landmark_loss(F_X_Y[0], landmarks00 - torch.tensor(
            #     [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
            #                             landmarks50 - torch.tensor(
            #                                 [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1,
            #                                                                                                       3).cuda(),
            #                             args.dirlab_cfg[batch + 1]['pixel_spacing'])

            # losses.append([_mean.item(), _std.item(), _mse.item(), Jac.item()])
            # print('case=%d after warped, TRE=%.2f+-%.2f MSE=%.5f Jac=%.6f' % (
            #     batch + 1, _mean.item(), _std.item(), _mse.item(), Jac.item()))
            losses.append([_mse.item(), Jac.item()])
            print('case=%d after warped,MSE=%.5f Jac=%.6f' % (
                batch + 1, _mse.item(), Jac.item()))

            if is_save:
                # Save DVF
                # b,3,d,h,w-> d,h,w,3    (dhw or whd) depend on the shape of image
                m2f_name = img_name[0][:13] + '_warpped_flow.nii.gz'
                save_image(torch.permute(F_X_Y_cpu, (1, 2, 3, 0)), fixed_img[0], args.output_dir,
                           m2f_name)

                # m_name = "{}_warped_lapirn.nii.gz".format(img_name[0][:13])
                # # save_img(X_Y, args.output_dir + '/' + file_name + '_warpped_moving.nii.gz')
                # save_image(X_Y, fixed_img, args.output_dir, m_name)

                m_name = "{}_warped_lv1.nii.gz".format(img_name[0][:13])
                save_image(lv1_out, fixed_img, args.output_dir, m_name)

                m_name = "{}_warped_lv2.nii.gz".format(img_name[0][:13])
                save_image(lv2_out, fixed_img, args.output_dir, m_name)

                m_name = "{}_warped_lv3.nii.gz".format(img_name[0][:13])
                save_image(lv3_out, fixed_img, args.output_dir, m_name)

    mean_total = np.mean(losses, 0)
    mean_mse = mean_total[0]
    mean_jac = mean_total[1]
    # print('mean TRE=%.2f+-%.2f MSE=%.3f Jac=%.6f' % (mean_tre, mean_std, mean_mse, mean_jac))
    print('mean MSE=%.3f Jac=%.6f' % (mean_mse, mean_jac))
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

    # test_dataset = DirLabDataset(moving_files=m_img_file_list, fixed_files=f_img_file_list, landmark_files=landmark_list)
    test_dataset = PatientDataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    prefix = '2023-03-04-12-26-02'
    model_dir = args.checkpoint_path

    if args.checkpoint_name is not None:
        test_single(args, os.path.join(model_dir, args.checkpoint_name), True)
    else:
        checkpoint_list = sorted([os.path.join(model_dir, file) for file in os.listdir(model_dir) if prefix in file])
        for checkpoint in checkpoint_list:
            print(checkpoint)
            test_single(args, checkpoint)

    # validation(args)
