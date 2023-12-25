import os
import numpy as np
import torch
import torch.utils.data as Data

from utils.Functions import transform_unit_flow_to_flow, Grid
# from CRegNet import CRegNet_lv1, \
#     CRegNet_lv2, CRegNet_lv3, CRegNet_lv0

# from CCENet_single import CCRegNet_planB_lv1 as CRegNet_lv1, CCRegNet_planB_lv2 as CRegNet_lv2, \
#     CCRegNet_planB_lvl3 as CRegNet_lv3

from LapIRN import Miccai2020_LDR_laplacian_unit_disp_add_lvl0 as CRegNet_lv0,Miccai2020_LDR_laplacian_unit_disp_add_lvl1 as CRegNet_lv1,\
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2 as CRegNet_lv2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3 as CRegNet_lv3

from utils.utilize import load_landmarks, save_image, count_parameters
from utils.config import get_args
from utils.metric import MSE, landmark_loss, SSIM, NCC, jacobian_determinant
from utils.losses import neg_Jdet_loss
from utils.datagenerators import DirLabDataset, PatientDataset, build_dataloader_dirlab
from utils.Functions import test_dirlab


def test_dirlab_bak(args, model, is_save=False):

    with torch.no_grad():
        losses = []
        for batch, (moving, fixed, landmarks, img_name) in enumerate(test_loader_dirlab):
            moving_img = moving.to(args.device).float()
            fixed_img = fixed.to(args.device).float()
            landmarks00 = landmarks['landmark_00'].squeeze().cuda()
            landmarks50 = landmarks['landmark_50'].squeeze().cuda()

            res = model(moving_img, fixed_img)  # nibabel: b,c,w,h,d;simpleitk b,c,d,h,w
            F_X_Y, warped_img = res['flow'], res['warped_img']
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

    mean_total = np.mean(losses, 0)
    mean_tre = mean_total[0]
    mean_std = mean_total[1]
    mean_mse = mean_total[2]
    mean_jac = mean_total[3]
    mean_ncc = mean_total[4]
    mean_ssim = mean_total[5]
    print('mean TRE=%.2f+-%.2f MSE=%.3f Jac=%.6f ncc=%.6f ssim=%.6f' % (
        mean_tre, mean_std, mean_mse, mean_jac, mean_ncc, mean_ssim))


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
                save_image(torch.permute(F_X_Y_cpu, (1, 2, 3, 0)), args.output_dir,
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

    test_loader_dirlab = build_dataloader_dirlab(args, mode='test')

    prefix = '2023-04-26-13-17-59' # CCENet

    model_dir = args.checkpoint_path

    model_lvl0 = CRegNet_lv0(2, 3, args.initial_channels, is_train=True,
                             range_flow=range_flow, grid=grid_class).cuda()
    model_lvl1 = CRegNet_lv1(2, 3, args.initial_channels, is_train=True,
                             range_flow=range_flow, grid=grid_class, model_lvl0=None).cuda()
    model_lvl2 = CRegNet_lv2(2, 3, args.initial_channels, is_train=True,
                             range_flow=range_flow, model_lvl1=model_lvl1, grid=grid_class).cuda()
    model = CRegNet_lv3(2, 3, args.initial_channels, is_train=False,
                        range_flow=range_flow, model_lvl2=model_lvl2,
                        grid=grid_class).cuda()

    for param in model_lvl0.parameters():
        param.requires_grad = False

    for param in model_lvl1.parameters():
        param.requires_grad = False

    for param in model_lvl2.parameters():
        param.requires_grad = False

    print(count_parameters(model))

    # from thop import profile
    # tensor = (torch.randn(1,1,96,256,256).cuda().float(), torch.randn(1,1,96,256,256).cuda().float(),)
    # flops, params = profile(model, tensor)
    # (407800656.0, 288708.0)
    # (6964150608.0, 660996.0)  (6485806080.0, 370020.0)
    # (59410968912.0, 1033284.0)    (58932624384.0, 742308.0)   (51882467328.0, 370020.0)
    # (478981534032.0, 1405572.0)   (478503189504.0, 1114596.0) (471453032448.0, 742308.0)  (415055757312.0, 370020.0)

    # attenion
    # lv1=(415055757312.0, 370020.0)
    # lv2=(471641895936.0, 742482.0)
    # lv3=(478715660928.0, 1114944.0)
    # lv4=(479196956448.0, 1406094.0)
    if args.checkpoint_name is not None:
        model.load_state_dict(torch.load(os.path.join(model_dir, args.checkpoint_name))['model'])
        test_dirlab(args, model, test_loader_dirlab, norm=True, is_train=False, is_save=False, suffix='cce_nocor',calc_tre=False)
        # test_patient(args, os.path.join(model_dir, args.checkpoint_name), True)
    else:
        checkpoint_list = sorted([os.path.join(model_dir, file) for file in os.listdir(model_dir) if prefix in file])
        for checkpoint in checkpoint_list:
            print(checkpoint)
            model.load_state_dict(torch.load(checkpoint)['model'])
            test_dirlab(args, model, test_loader_dirlab, norm=True, is_train=True)
            # test_dirlab(args, checkpoint)
            # test_patient(args, checkpoint)

    # validation(args)
