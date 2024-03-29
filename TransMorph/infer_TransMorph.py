import os
import numpy as np
import torch

from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import torch.utils.data as Data

from utils.config import get_args
from utils.datagenerators import Dataset, DirLabDataset, build_dataloader_dirlab
from utils.metric import MSE, SSIM, NCC, jacobian_determinant, landmark_loss
from utils.utilize import save_image, load_landmarks
from utils.Functions import SpatialTransformer

def test_dirlab(args, checkpoint, is_save=False, calc_tre=True):
    model.load_state_dict(torch.load(checkpoint)['model'])

    with torch.no_grad():
        losses = []
        model.eval()
        for batch, (data) in enumerate(test_loader):
            landmarks = None
            moving, fixed, img_name = data[0], data[1], data[2]
            if calc_tre:
                landmarks = data[2]
                img_name = data[3]
                spacing = args.dirlab_cfg[batch + 1]['pixel_spacing']
                landmarks00 = landmarks['landmark_00'].squeeze().cuda()
                landmarks50 = landmarks['landmark_50'].squeeze().cuda()
                crop_range = args.dirlab_cfg[batch + 1]['crop_range']
                # TRE
                _mean, _std = landmark_loss(flow[0], landmarks00 - torch.tensor(
                    [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
                                            landmarks50 - torch.tensor(
                                                [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1,
                                                                                                                      3).cuda(),
                                            args.dirlab_cfg[batch + 1]['pixel_spacing'])
                _mean, _std = _mean.item(), _std.item()
            else:
                _mean, _std = 0, 0
                spacing = None

            x = moving.to(args.device).float()
            y = fixed.to(args.device).float()

            x_in = torch.cat((x, y), dim=1)
            flow = model(x_in, True)  # warped,DVF

            x_def = STN(x, flow)

            ncc = NCC(y.cpu().detach().numpy(), x_def.cpu().detach().numpy())
            jac = jacobian_determinant(flow.squeeze().cpu().detach().numpy())
            mse = MSE(y, x_def)
            ssim = SSIM(y.cpu().detach().numpy()[0, 0], x_def.cpu().detach().numpy()[0, 0])


            losses.append([_mean, _std, mse.item(), jac, ncc.item(), ssim.item()])
            print('case=%d after warped, TRE=%.2f+-%.2f MSE=%.5f Jac=%.6f ncc=%.6f ssim=%.6f' % (
                batch + 1, _mean, _std, mse.item(), jac, ncc.item(), ssim.item()))

            if is_save:
                # Save DVF
                # b,3,d,h,w-> d,h,w,3    (dhw or whd) depend on the shape of image
                m2f_name = img_name[0][:13] + '_flow_TM.nii.gz'
                save_image(flow[0].permute(1, 2, 3, 0), args.output_dir,m2f_name, spacing=spacing)

                m_name = "{}_warped_TM.nii.gz".format(img_name[0][:13])
                save_image(x_def.squeeze(), args.output_dir, m_name, spacing=spacing)

    mean_total = np.mean(losses, 0)
    mean_tre = mean_total[0]
    mean_std = mean_total[1]
    mean_mse = mean_total[2]
    mean_jac = mean_total[3]
    mean_ncc = mean_total[4]
    mean_ssim = mean_total[5]
    print('mean TRE=%.2f+-%.2f MSE=%.3f Jac=%.6f ncc=%.6f ssim=%.6f' % (
        mean_tre, mean_std, mean_mse, mean_jac, mean_ncc, mean_ssim))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':

    args = get_args()
    device = args.device
    STN = SpatialTransformer()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    test_loader = build_dataloader_dirlab(args,mode='test')

    prefix = '2023-08-21-17-27-29'
    model_dir = args.checkpoint_path

    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    model = model.to(device)

    if args.checkpoint_name is not None:
        test_dirlab(args, os.path.join(model_dir, args.checkpoint_name), is_save=False,calc_tre=False)
        # test_patient(args, os.path.join(model_dir, args.checkpoint_name))
    else:
        checkpoint_list = sorted([os.path.join(model_dir, file) for file in os.listdir(model_dir) if prefix in file])
        for checkpoint in checkpoint_list:
            print(checkpoint)
            # test_dirlab(args, checkpoint)
            test_dirlab(args, checkpoint)

