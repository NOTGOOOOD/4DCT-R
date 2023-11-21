import os
import numpy as np
import torch
import torch.utils.data as Data

from utils.utilize import save_image, load_landmarks
from utils.config import get_args
from utils.metric import MSE, SSIM, NCC, jacobian_determinant, landmark_loss
from utils.datagenerators import PatientDataset, DirLabDataset, build_dataloader_dirlab
from utils.Functions import test_dirlab
from voxelmorph.vmmodel import vmnetwork


def test_patient(args, checkpoint, is_save=False):
    model.load_state_dict(torch.load(checkpoint)['model'])
    with torch.no_grad():
        model.eval()
        losses = []
        for batch, (moving, fixed, img_name) in enumerate(test_loader_patient):
            moving_img = moving.to(args.device).float()
            fixed_img = fixed.to(args.device).float()

            img_shape = fixed_img.shape[2:]

            y_pred = model(moving_img, fixed_img, True)  # b, c, d, h, w warped_image, flow_m2f

            ncc = NCC(fixed_img.cpu().detach().numpy(), y_pred[0].cpu().detach().numpy())

            # loss_Jacobian = neg_Jdet_loss(y_pred[1].permute(0, 2, 3, 4, 1), grid)
            jac = jacobian_determinant(y_pred[1][0].cpu().detach().numpy())

            # MSE
            _mse = MSE(fixed_img, y_pred[0])
            # SSIM
            _ssim = SSIM(fixed_img.cpu().detach().numpy()[0, 0], y_pred[0].cpu().detach().numpy()[0, 0])

            losses.append([_mse.item(), jac, _ssim.item(), ncc.item()])
            print('case=%d after warped,MSE=%.5f Jac=%.8f, SSIM=%.5f, NCC=%.5f' % (
                batch + 1, _mse.item(), jac, _ssim.item(), ncc.item()))

            if is_save:
                # Save DVF
                # b,3,d,h,w-> d,h,w,3    (dhw or whd) depend on the shape of image
                m2f_name = img_name[0][:13] + '_warpped_flow_vm.nii.gz'
                save_image(torch.permute(y_pred[1][0], (1, 2, 3, 0)), fixed_img[0], args.output_dir,
                           m2f_name)

                # m_name = "{}_warped_lapirn.nii.gz".format(img_name[0][:13])
                # # save_img(X_Y, args.output_dir + '/' + file_name + '_warpped_moving.nii.gz')
                # save_image(X_Y, fixed_img, args.output_dir, m_name)

                m_name = "{}_warped_vm.nii.gz".format(img_name[0][:13])
                save_image(y_pred[0], fixed_img, args.output_dir, m_name)

    mean_total = np.mean(losses, 0)
    mean_mse = mean_total[0]
    mean_jac = mean_total[1]
    mean_ssim = mean_total[2]
    mean_ncc = mean_total[3]
    # print('mean TRE=%.2f+-%.2f MSE=%.3f Jac=%.6f' % (mean_tre, mean_std, mean_mse, mean_jac))
    print('mean SSIM=%.5f Jac=%.8f MSE=%.5f NCC=%.5f' % (mean_ssim, mean_jac, mean_mse, mean_ncc))


if __name__ == '__main__':
    args = get_args()
    device = args.device

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    test_loader_dirlab = build_dataloader_dirlab(args, mode="test")
    prefix = '2023-09-11-23-40-13'
    model_dir = args.checkpoint_path

    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    model = vmnetwork.VxmDense(
        dim=3,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=args.bidir,
        int_steps=7,
        int_downsize=2
    )
    model.to(device)

    # from thop import profile
    # tensor = (torch.randn(1,1,96,144,144).cuda().float(), torch.randn(1,1,96,144,144).cuda().float(),)
    # flops, params = profile(model, tensor)

    model_dir = args.checkpoint_path
    prefix = '2023-04-21-17-47-16'
    if args.checkpoint_name is not None:
        model.load_state_dict(torch.load(os.path.join(model_dir, args.checkpoint_name))['model'])
        test_dirlab(args, model, test_loader_dirlab, is_train=False)
        # test_patient(args, os.path.join(model_dir, args.checkpoint_name), True)
    else:
        checkpoint_list = sorted([os.path.join(model_dir, file) for file in os.listdir(model_dir) if prefix in file])
        for checkpoint in checkpoint_list:
            print(checkpoint)
            model.load_state_dict(torch.load(checkpoint)['model'])
            test_dirlab(args, model, test_loader_dirlab, is_train=False)

    # validation(args)
