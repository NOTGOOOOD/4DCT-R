import os
import numpy as np
import torch
import torch.utils.data as Data

from utils.Functions import transform_unit_flow_to_flow, Grid, generate_grid
from miccai2021_model import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
    Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2, Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3

from utils.utilize import load_landmarks, save_image
from utils.config import get_args
from utils.metric import MSE, landmark_loss, SSIM, NCC, jacobian_determinant
from utils.datagenerators import DirLabDataset, PatientDataset


def test_patient(args, checkpoint, is_save=False):
    reg_input = 0.4
    with torch.no_grad():
        losses = []
        for batch, (moving, fixed, img_name) in enumerate(test_loader_patient):
            moving_img = moving.to(args.device).float()
            fixed_img = fixed.to(args.device).float()

            reg_code = torch.tensor([reg_input], dtype=fixed_img.dtype, device=fixed_img.device).unsqueeze(dim=0)

            # imgshape = fixed_img.shape[2:]
            #
            # imgshape_4 = (imgshape[0] / 4, imgshape[1] / 4, imgshape[2] / 4)
            # imgshape_2 = (imgshape[0] / 2, imgshape[1] / 2, imgshape[2] / 2)

            model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, args.initial_channels, is_train=True,
                                                                                 range_flow=range_flow, grid=grid_class).cuda()
            model_lvl2 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 3, args.initial_channels, is_train=True,
                                                                                 range_flow=range_flow, grid=grid_class,
                                                                                 model_lvl1=model_lvl1).cuda()

            model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3(2, 3, args.initial_channels, is_train=True,
                                                                            range_flow=range_flow, grid=grid_class,
                                                                            model_lvl2=model_lvl2).cuda()

            model.load_state_dict(torch.load(checkpoint)['model'])
            model.eval()

            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(moving_img, fixed_img,reg_code)  # nibabel: b,c,w,h,d;simpleitk b,c,d,h,w

            # X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid)

            F_X_Y_cpu = F_X_Y[0, :, :, :, :]
            F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

            # Jac = neg_Jdet_loss(F_X_Y_cpu.unsqueeze(0).permute(0, 2, 3, 4, 1), grid)
            Jac = jacobian_determinant(F_X_Y_cpu.cpu().detach().numpy())
            # Jac = Get_Ja(F_X_Y.cpu().detach().numpy())

            # NCC
            _ncc = NCC(X_Y.cpu().detach().numpy(), fixed_img.cpu().detach().numpy())

            # MSE
            _mse = MSE(fixed_img, X_Y)
            # SSIM
            _ssim = SSIM(fixed_img.cpu().detach().numpy()[0, 0], X_Y.cpu().detach().numpy()[0, 0])

            losses.append([_mse.item(), Jac, _ssim.item(), _ncc.item()])
            print('case=%d after warped,MSE=%.5f Jac=%.6f, SSIM=%.5f, NCC=%.5f' % (
                batch + 1, _mse.item(), Jac, _ssim.item(), _ncc.item()))

            if is_save:
                # Save DVF
                # b,3,d,h,w-> d,h,w,3    (dhw or whd) depend on the shape of image
                m2f_name = img_name[0][:13] + '_clapirn_warpped_flow.nii.gz'
                save_image(torch.permute(F_X_Y_cpu, (1, 2, 3, 0)), args.output_dir,
                           m2f_name)

                # m_name = "{}_warped_lapirn.nii.gz".format(img_name[0][:13])
                # # save_img(X_Y, args.output_dir + '/' + file_name + '_warpped_moving.nii.gz')
                # save_image(X_Y, fixed_img, args.output_dir, m_name)

                m_name = "{}_clapirn_warped_lv1.nii.gz".format(img_name[0][:13])
                save_image(F_xy_lvl1, args.output_dir, m_name)

                m_name = "{}_clapirn_warped_lv2.nii.gz".format(img_name[0][:13])
                save_image(F_xy_lvl2, args.output_dir, m_name)

                m_name = "{}_clapirn_warped_lv3.nii.gz".format(img_name[0][:13])
                save_image(X_Y, args.output_dir, m_name)

    mean_total = np.mean(losses, 0)
    mean_mse = mean_total[0]
    mean_jac = mean_total[1]
    mean_ssim = mean_total[2]
    mean_ncc = mean_total[3]
    # print('mean TRE=%.2f+-%.2f MSE=%.3f Jac=%.6f' % (mean_tre, mean_std, mean_mse, mean_jac))
    print('mean SSIM=%.5f Jac=%.6f MSE=%.5f NCC=%.5f' % (mean_ssim, mean_jac, mean_mse, mean_ncc))


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

    prefix = '2023-05-01-09-44-11'
    model_dir = args.checkpoint_path

    if args.checkpoint_name is not None:
        # test_dirlab(args, os.path.join(model_dir, args.checkpoint_name), True)
        test_patient(args, os.path.join(model_dir, args.checkpoint_name), True)
    else:
        checkpoint_list = sorted([os.path.join(model_dir, file) for file in os.listdir(model_dir) if prefix in file])
        for checkpoint in checkpoint_list:
            print(checkpoint)
            if os.path.getsize(checkpoint) > 0:
                # test_dirlab(args, checkpoint)
                test_patient(args, checkpoint)

    # validation(args)
