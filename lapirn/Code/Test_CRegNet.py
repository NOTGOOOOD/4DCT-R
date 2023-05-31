import os
import numpy as np
import torch
import torch.utils.data as Data

from utils.Functions import transform_unit_flow_to_flow, Grid
# from CRegNet import CRegNet_lv1, \
#     CRegNet_lv2, CRegNet_lv3
from CCRegNet_planB import CCRegNet_planB_lv1 as CRegNet_lv1, CCRegNet_planB_lv2 as CRegNet_lv2, \
    CCRegNet_planB_lvl3 as CRegNet_lv3

# from CCENet_single import CCRegNet_planB_lv1 as CRegNet_lv1, CCRegNet_planB_lv2 as CRegNet_lv2, \
#     CCRegNet_planB_lvl3 as CRegNet_lv3

from utils.utilize import load_landmarks, save_image
from utils.config import get_args
from utils.metric import MSE, landmark_loss, SSIM, NCC, jacobian_determinant
from utils.losses import neg_Jdet_loss
from utils.datagenerators import DirLabDataset, PatientDataset


def test_dirlab(args, checkpoint, is_save=False):
    model_lvl1 = CRegNet_lv1(1, 3, args.initial_channels, is_train=True,
                             range_flow=range_flow, grid=grid_class).cuda()
    model_lvl2 = CRegNet_lv2(1, 3, args.initial_channels, is_train=True,
                             range_flow=range_flow,
                             model_lvl1=model_lvl1, grid=grid_class).cuda()

    model = CRegNet_lv3(1, 3, args.initial_channels, is_train=False,
                        range_flow=range_flow, model_lvl2=model_lvl2,
                        grid=grid_class).cuda()

    model.load_state_dict(torch.load(checkpoint)['model'])
    model.eval()

    with torch.no_grad():
        losses = []
        for batch, (moving, fixed, landmarks, img_name) in enumerate(test_loader_dirlab):
            moving_img = moving.to(args.device).float()
            fixed_img = fixed.to(args.device).float()
            landmarks00 = landmarks['landmark_00'].squeeze().cuda()
            landmarks50 = landmarks['landmark_50'].squeeze().cuda()

            F_X_Y, lv1_out, lv2_out, lv3_out = model(moving_img, fixed_img)  # nibabel: b,c,w,h,d;simpleitk b,c,d,h,w

            F_X_Y_cpu = F_X_Y[0, :, :, :, :]
            F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

            # X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid)

            # Jac = neg_Jdet_loss(F_X_Y_cpu.unsqueeze(0).permute(0, 2, 3, 4, 1), grid_class.get_grid(lv3_out.shape[2:]))

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

            # TRE
            _mean, _std = landmark_loss(F_X_Y[0], landmarks00 - torch.tensor(
                [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
                                        landmarks50 - torch.tensor(
                                            [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1,
                                                                                                                  3).cuda(),
                                        args.dirlab_cfg[batch + 1]['pixel_spacing'],
                                        fixed_img.cpu().detach().numpy()[0, 0], is_save)

            ncc = NCC(fixed_img.cpu().detach().numpy(), lv3_out.cpu().detach().numpy())

            # loss_Jacobian = neg_Jdet_loss(y_pred[1].permute(0, 2, 3, 4, 1), grid)
            jac = jacobian_determinant(lv3_out[0].cpu().detach().numpy())

            # SSIM
            _ssim = SSIM(fixed_img.cpu().detach().numpy()[0, 0], lv3_out.cpu().detach().numpy()[0, 0])

            losses.append([_mean.item(), _std.item(), _mse.item(), jac, ncc.item(), _ssim.item()])
            print('case=%d after warped, TRE=%.2f+-%.2f MSE=%.5f Jac=%.6f ncc=%.6f ssim=%.6f' % (
                batch + 1, _mean.item(), _std.item(), _mse.item(), jac, ncc.item(), _ssim.item()))

            # if is_save:
            #     # F_X_Y_cpu = F_X_Y[0, :, :, :, :]
            #     # F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)
            #     # Save DVF
            #     # b,3,d,h,w-> d,h,w,3    (dhw or whd) depend on the shape of image
            #     m2f_name = img_name[0][:13] + '_warpped_flow.nii.gz'
            #     save_image(torch.permute(F_X_Y_cpu, (1, 2, 3, 0)), fixed_img[0], args.output_dir,
            #                m2f_name)
            #
            #     # m_name = "{}_warped_lapirn.nii.gz".format(img_name[0][:13])
            #     # # save_img(X_Y, args.output_dir + '/' + file_name + '_warpped_moving.nii.gz')
            #     # save_image(X_Y, fixed_img, args.output_dir, m_name)
            #
            #     m_name = "{}_warped_lv1.nii.gz".format(img_name[0][:13])
            #     save_image(lv1_out, fixed_img, args.output_dir, m_name)
            #
            #     m_name = "{}_warped_lv2.nii.gz".format(img_name[0][:13])
            #     save_image(lv2_out, fixed_img, args.output_dir, m_name)
            #
            #     m_name = "{}_warped_lv3.nii.gz".format(img_name[0][:13])
            #     save_image(lv3_out, fixed_img, args.output_dir, m_name)

    mean_total = np.mean(losses, 0)
    mean_tre = mean_total[0]
    mean_std = mean_total[1]
    mean_mse = mean_total[2]
    mean_jac = mean_total[3]
    mean_ncc = mean_total[4]
    mean_ssim = mean_total[5]
    print('mean TRE=%.2f+-%.2f MSE=%.3f Jac=%.6f ncc=%.6f ssim=%.6f' % (
        mean_tre, mean_std, mean_mse, mean_jac, mean_ncc, mean_ssim))
    # print('mean MSE=%.3f Jac=%.6f' % (mean_mse, mean_jac))
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


def test_patient(args, checkpoint, is_save=False):
    with torch.no_grad():
        losses = []
        model_lvl1 = CRegNet_lv1(1, 3, args.initial_channels, is_train=True,
                                 grid=grid_class,
                                 range_flow=range_flow).cuda()

        model_lvl1.eval()

        model_lvl2 = CRegNet_lv2(1, 3, args.initial_channels, is_train=True,
                                 range_flow=range_flow,
                                 model_lvl1=model_lvl1, grid=grid_class).cuda()
        model_lvl2.eval()

        model = CRegNet_lv3(1, 3, args.initial_channels, is_train=False,
                            range_flow=range_flow, model_lvl2=model_lvl2,
                            grid=grid_class).cuda()
        # model = model_lvl2
        model.load_state_dict(torch.load(checkpoint)['model'])
        model.eval()

        for batch, (moving, fixed, img_name) in enumerate(test_loader_patient):
            if batch ==6: break
            moving_img = moving.to(args.device).float()
            fixed_img = fixed.to(args.device).float()

            F_X_Y, lv1_out, lv2_out, lv3_out = model(moving_img, fixed_img)  # nibabel: b,c,w,h,d;simpleitk b,c,d,h,w

            # X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid)
            F_X_Y = torch.nn.functional.interpolate(F_X_Y, size=moving_img.shape[2:],
                                            mode='trilinear',
                                            align_corners=True)
            lv3_out = torch.nn.functional.interpolate(lv3_out, size=moving_img.shape[2:],
                                            mode='trilinear',
                                            align_corners=True)
            F_X_Y_cpu = F_X_Y[0, :, :, :, :]
            F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

            # Jac = neg_Jdet_loss(F_X_Y_cpu.unsqueeze(0).permute(0, 2, 3, 4, 1), grid_class.get_grid(lv3_out.shape[2:]))
            # J = jacobian_determinant_vxm(F_X_Y_cpu)
            Jac = jacobian_determinant(F_X_Y_cpu.cpu().detach().numpy())

            # NCC
            _ncc = NCC(lv3_out.cpu().detach().numpy(), fixed_img.cpu().detach().numpy())

            # MSE
            _mse = MSE(fixed_img, lv3_out)
            # SSIM
            _ssim = SSIM(fixed_img.cpu().detach().numpy()[0, 0], lv3_out.cpu().detach().numpy()[0, 0])

            losses.append([_mse.item(), Jac, _ssim.item(), _ncc.item()])
            print('case=%d after warped,MSE=%.5f Jac=%.8f, SSIM=%.5f, NCC=%.5f' % (
                batch + 1, _mse.item(), Jac, _ssim.item(), _ncc.item()))

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
    mean_ssim = mean_total[2]
    mean_ncc = mean_total[3]
    # print('mean TRE=%.2f+-%.2f MSE=%.3f Jac=%.6f' % (mean_tre, mean_std, mean_mse, mean_jac))
    print('mean SSIM=%.5f Jac=%.8f MSE=%.5f NCC=%.5f' % (mean_ssim, mean_jac, mean_mse, mean_ncc))


if __name__ == '__main__':
    args = get_args()
    grid_class = Grid()
    range_flow = 0.4

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    landmark_list = load_landmarks(args.landmark_dir)

    pa_fixed_folder = r'D:\xxf\test_ori\fixed'
    pa_moving_folder = r'D:\xxf\test_ori\moving'

    # pa_fixed_folder = r'E:\datasets\registration\patient\fixed'
    # pa_moving_folder = r'E:\datasets\registration\patient\moving'

    f_patient_file_list = sorted(
        [os.path.join(pa_fixed_folder, file_name) for file_name in os.listdir(pa_fixed_folder) if
         file_name.lower().endswith('.gz')])
    m_patient_file_list = sorted(
        [os.path.join(pa_moving_folder, file_name) for file_name in os.listdir(pa_moving_folder) if
         file_name.lower().endswith('.gz')])

    test_dataset_patient = PatientDataset(moving_files=m_patient_file_list, fixed_files=f_patient_file_list)

    test_loader_patient = Data.DataLoader(test_dataset_patient, batch_size=args.batch_size, shuffle=False,
                                          num_workers=0)

    dir_fixed_folder = os.path.join(args.test_dir, 'fixed')
    dir_moving_folder = os.path.join(args.test_dir, 'moving')
    f_dir_file_list = sorted([os.path.join(dir_fixed_folder, file_name) for file_name in os.listdir(dir_fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_dir_file_list = sorted(
        [os.path.join(dir_moving_folder, file_name) for file_name in os.listdir(dir_moving_folder) if
         file_name.lower().endswith('.gz')])

    test_dataset_dirlab = DirLabDataset(moving_files=m_dir_file_list, fixed_files=f_dir_file_list,
                                        landmark_files=landmark_list)
    test_loader_dirlab = Data.DataLoader(test_dataset_dirlab, batch_size=args.batch_size, shuffle=False, num_workers=0)

    prefix = '2023-04-26-13-17-59' # CCENet

    model_dir = args.checkpoint_path

    if args.checkpoint_name is not None:
        test_dirlab(args, os.path.join(model_dir, args.checkpoint_name), True)
        # test_patient(args, os.path.join(model_dir, args.checkpoint_name), True)
    else:
        checkpoint_list = sorted([os.path.join(model_dir, file) for file in os.listdir(model_dir) if prefix in file])
        for checkpoint in checkpoint_list:
            print(checkpoint)
            # test_dirlab(args, checkpoint)
            test_patient(args, checkpoint)

    # validation(args)
