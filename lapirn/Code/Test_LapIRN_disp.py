import os
import numpy as np
import torch
import torch.utils.data as Data
# from CRegNet import CRegNet_lv0 as RegNet_v0, CRegNet_lv1 as RegNet_v1, \
#     CRegNet_lv2 as RegNet_v2, CRegNet_lv3 as RegNet_v3

from LapIRN import Miccai2020_LDR_laplacian_unit_disp_add_lvl0 as RegNet_v0,\
    Miccai2020_LDR_laplacian_unit_disp_add_lvl1 as RegNet_v1, \
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2 as RegNet_v2, \
    Miccai2020_LDR_laplacian_unit_disp_add_lvl3 as RegNet_v3
# from CCENet_single import CCRegNet_planB_lv1 as Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
#     CCRegNet_planB_lv2 as Miccai2020_LDR_laplacian_unit_disp_add_lvl2, \
#     CCRegNet_planB_lvl3 as Miccai2020_LDR_laplacian_unit_disp_add_lvl3
from utils.utilize import load_landmarks, save_image
from utils.config import get_args
from utils.metric import MSE, landmark_loss, SSIM, NCC, jacobian_determinant
from utils.losses import neg_Jdet_loss
from utils.datagenerators import DirLabDataset, PatientDataset
from utils.Functions import transform_unit_flow_to_flow, Grid


def test_dirlab(args, checkpoint, is_save=False):
    with torch.no_grad():
        losses = []
        for batch, (moving, fixed, landmarks, img_name) in enumerate(test_loader_dirlab):
            spacing = args.dirlab_cfg[batch+1]['pixel_spacing']
            moving_img = moving.to(args.device).float()
            fixed_img = fixed.to(args.device).float()
            landmarks00 = landmarks['landmark_00'].squeeze().cuda()
            landmarks50 = landmarks['landmark_50'].squeeze().cuda()

            model_lvl0 = RegNet_v0(2, 3, args.initial_channels, is_train=True, range_flow=range_flow, grid=grid_class).cuda()
            model_lvl1 = RegNet_v1(2, 3, args.initial_channels, is_train=True,
                                                                     range_flow=range_flow, grid=grid_class, model_lvl0=None).cuda()
            model_lvl2 = RegNet_v2(2, 3, args.initial_channels, is_train=True,
                                                                     range_flow=range_flow,
                                                                     model_lvl1=model_lvl1, grid=grid_class).cuda()

            model = RegNet_v3(2, 3, args.initial_channels, is_train=False,
                                                                range_flow=range_flow, model_lvl2=model_lvl2,
                                                                grid=grid_class).cuda()

            model_state = model.state_dict()
            loaded_state = torch.load(os.path.join(checkpoint))["model"]
            for loaded_key in loaded_state:
                if loaded_key in model_state:
                    if model_state[loaded_key].shape != loaded_state[loaded_key].shape:
                        print("{}: model_state_shape: {}, loaded_state_shape: {}".format(
                            loaded_key, model_state[loaded_key].shape, loaded_state[loaded_key].shape))
                        continue
                    else:
                        model_state[loaded_key].copy_(loaded_state[loaded_key])
                else:
                    print("{}: In checkpoint but not in model".format(loaded_key))

            # model.load_state_dict(torch.load(checkpoint)['model'])
            # model.load_state_dict(torch.load(checkpoint))
            model.eval()
            res = model(moving_img, fixed_img)  # nibabel: b,c,w,h,d;simpleitk b,c,d,h,w
            F_X_Y, lv1_out, lv2_out, lv3_out = res['flow'], res['warped_imglv1'], res['warped_imglv2'] ,res['warped_img']
            # X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid)
            F_X_Y_cpu = F_X_Y[0, :, :, :, :]
            F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)
            crop_range = args.dirlab_cfg[batch + 1]['crop_range']

            # MSE
            ncc = NCC(fixed_img.cpu().detach().numpy(), lv3_out.cpu().detach().numpy())

            # loss_Jacobian = neg_Jdet_loss(y_pred[1].permute(0, 2, 3, 4, 1), grid)
            jac = jacobian_determinant(lv3_out[0].cpu().detach().numpy())

            # SSIM
            ssim = SSIM(fixed_img.cpu().detach().numpy()[0, 0], lv3_out.cpu().detach().numpy()[0, 0])

            # TRE
            _mean, _std = landmark_loss(F_X_Y[0], landmarks00 - torch.tensor(
                [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
                                        landmarks50 - torch.tensor(
                                            [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1,
                                                                                                                  3).cuda(),
                                        args.dirlab_cfg[batch + 1]['pixel_spacing'],
                                        fixed_img.cpu().detach().numpy()[0, 0], is_save)

            losses.append([_mean.item(), _std.item(), ncc.item(), ssim.item(), jac])
            if is_save:
                print('case=%d after warped, TRE=%.2f+-%.2f Jac=%.6f ncc=%.6f ssim=%.6f' % (
                    batch + 1, _mean.item(), _std.item(), jac, ncc.item(), ssim.item()))
            if is_save:
                # Save DVF
                # b,3,d,h,w-> d,h,w,3    (dhw or whd) depend on the shape of image
                m2f_name = img_name[0][:13] + '_warpped_flow_lap.nii.gz'
                save_image(F_X_Y_cpu.permute(1,2,3,0), args.output_dir,
                           m2f_name,spacing=spacing)
                m_name = "{}_warped_lv1_lap.nii.gz".format(img_name[0][:13])
                save_image(lv1_out.squeeze(), args.output_dir, m_name,spacing=spacing)

                m_name = "{}_warped_lv2_lap.nii.gz".format(img_name[0][:13])
                save_image(lv2_out.squeeze(),args.output_dir, m_name,spacing=spacing)

                m_name = "{}_warped_lv3_lap.nii.gz".format(img_name[0][:13])
                save_image(lv3_out.squeeze(), args.output_dir, m_name,spacing=spacing)

        mean_tre, mean_std, mean_ncc, mean_ssim, mean_jac = np.mean(losses, 0)

        print('mean TRE=%.2f+-%.2f NCC=%.6f SSIM=%.6f J=%.6f' % (
            mean_tre, mean_std, mean_ncc, mean_ssim, mean_jac))


def test_patient(args, checkpoint, is_save=False):
    with torch.no_grad():
        losses = []
        for batch, (moving, fixed, img_name) in enumerate(test_loader_patient):
            moving_img = moving.to(args.device).float()
            fixed_img = fixed.to(args.device).float()

            model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, args.initial_channels, is_train=True,
                                                                     grid=grid_class,
                                                                     range_flow=range_flow).cuda()
            model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, args.initial_channels, is_train=True,
                                                                     range_flow=range_flow,
                                                                     model_lvl1=model_lvl1, grid=grid_class).cuda()

            model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(2, 3, args.initial_channels, is_train=False,
                                                                range_flow=range_flow, model_lvl2=model_lvl2,
                                                                grid=grid_class).cuda()

            # model.load_state_dict(torch.load(checkpoint)['model'])
            model.load_state_dict(torch.load(checkpoint))
            model.eval()

            F_X_Y, lv1_out, lv2_out, lv3_out = model(moving_img, fixed_img)  # nibabel: b,c,w,h,d;simpleitk b,c,d,h,w

            # X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid)

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
                save_image(lv1_out, args.output_dir, m_name)

                m_name = "{}_warped_lv2.nii.gz".format(img_name[0][:13])
                save_image(lv2_out, args.output_dir, m_name)

                m_name = "{}_warped_lv3.nii.gz".format(img_name[0][:13])
                save_image(lv3_out, args.output_dir, m_name)

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
    dir_fixed_folder = os.path.join(args.test_dir, 'fixed')
    dir_moving_folder = os.path.join(args.test_dir, 'moving')
    pa_fixed_folder = r'D:\xxf\test_patient\fixed'
    pa_moving_folder = r'D:\xxf\test_patient\moving'

    f_dir_file_list = sorted([os.path.join(dir_fixed_folder, file_name) for file_name in os.listdir(dir_fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_dir_file_list = sorted(
        [os.path.join(dir_moving_folder, file_name) for file_name in os.listdir(dir_moving_folder) if
         file_name.lower().endswith('.gz')])

    f_patient_file_list = sorted(
        [os.path.join(pa_fixed_folder, file_name) for file_name in os.listdir(pa_fixed_folder) if
         file_name.lower().endswith('.gz')])
    m_patient_file_list = sorted(
        [os.path.join(pa_moving_folder, file_name) for file_name in os.listdir(pa_moving_folder) if
         file_name.lower().endswith('.gz')])

    test_dataset_dirlab = DirLabDataset(moving_files=m_dir_file_list, fixed_files=f_dir_file_list,
                                        landmark_files=landmark_list)
    test_dataset_patient = PatientDataset(moving_files=m_patient_file_list, fixed_files=f_patient_file_list)
    test_loader_dirlab = Data.DataLoader(test_dataset_dirlab, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader_patient = Data.DataLoader(test_dataset_patient, batch_size=args.batch_size, shuffle=False,
                                          num_workers=0)

    # prefix = '2023-05-24-18-26-12'
    prefix = '2023-08-16-23-12-51'
    model_dir = args.checkpoint_path

    if args.checkpoint_name is not None:
        test_dirlab(args, os.path.join(model_dir, args.checkpoint_name), True)
        # test_patient(args, os.path.join(model_dir, args.checkpoint_name), True)
    else:
        checkpoint_list = sorted([os.path.join(model_dir, file) for file in os.listdir(model_dir) if prefix in file])
        for checkpoint in checkpoint_list:
            print(checkpoint)
            test_dirlab(args, checkpoint)
            # test_patient(args, checkpoint)

    # validation(args)
