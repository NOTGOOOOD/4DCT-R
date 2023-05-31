import os
import numpy as np
import torch

from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import torch.utils.data as Data

from utils.config import get_args
from utils.datagenerators import Dataset, DirLabDataset
from utils.metric import MSE, SSIM, NCC, jacobian_determinant, landmark_loss
from utils.utilize import save_image, load_landmarks
from utils.Functions import SpatialTransformer

def test_dirlab(args, checkpoint):
    model.load_state_dict(torch.load(checkpoint)['model'])

    with torch.no_grad():
        losses = []
        model.eval()
        for batch, (moving, fixed, landmarks, img_name) in enumerate(test_loader):
            x = moving.to(args.device).float()
            y = fixed.to(args.device).float()
            landmarks00 = landmarks['landmark_00'].squeeze().cuda()
            landmarks50 = landmarks['landmark_50'].squeeze().cuda()

            x_in = torch.cat((x, y), dim=1)
            flow = model(x_in, True)  # warped,DVF

            x_def = STN(x, flow)

            ncc = NCC(y.cpu().detach().numpy(), x_def.cpu().detach().numpy())
            jac = jacobian_determinant(flow.squeeze().cpu().detach().numpy())
            mse = MSE(y, x_def)
            ssim = SSIM(y.cpu().detach().numpy()[0, 0], x_def.cpu().detach().numpy()[0, 0])

            crop_range = args.dirlab_cfg[batch + 1]['crop_range']
            # TRE
            _mean, _std = landmark_loss(flow[0], landmarks00 - torch.tensor(
                [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
                                        landmarks50 - torch.tensor(
                                            [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1,
                                                                                                                  3).cuda(),
                                        args.dirlab_cfg[batch + 1]['pixel_spacing'],
                                        y.cpu().detach().numpy()[0, 0])
            losses.append([_mean.item(), _std.item(), mse.item(), jac, ncc.item(), ssim.item()])
            print('case=%d after warped, TRE=%.2f+-%.2f MSE=%.5f Jac=%.6f ncc=%.6f ssim=%.6f' % (
                batch + 1, _mean.item(), _std.item(), mse.item(), jac, ncc.item(), ssim.item()))

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

    # pa_fixed_folder = r'D:\xxf\test_patient\fixed'
    # pa_moving_folder = r'D:\xxf\test_patient\moving'
    # f_patient_file_list = sorted(
    #     [os.path.join(pa_fixed_folder, file_name) for file_name in os.listdir(pa_fixed_folder) if
    #      file_name.lower().endswith('.gz')])
    # m_patient_file_list = sorted(
    #     [os.path.join(pa_moving_folder, file_name) for file_name in os.listdir(pa_moving_folder) if
    #      file_name.lower().endswith('.gz')])
    #
    # test_dataset_patient = PatientDataset(moving_files=m_patient_file_list, fixed_files=f_patient_file_list)
    # test_loader_patient = Data.DataLoader(test_dataset_patient, batch_size=args.batch_size, shuffle=False,
    #                                       num_workers=0)

    landmark_list = load_landmarks(args.landmark_dir)
    dir_fixed_folder = os.path.join(args.test_dir, 'fixed')
    dir_moving_folder = os.path.join(args.test_dir, 'moving')

    f_dir_file_list = sorted([os.path.join(dir_fixed_folder, file_name) for file_name in os.listdir(dir_fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_dir_file_list = sorted(
        [os.path.join(dir_moving_folder, file_name) for file_name in os.listdir(dir_moving_folder) if
         file_name.lower().endswith('.gz')])
    test_dataset = DirLabDataset(moving_files=m_dir_file_list, fixed_files=f_dir_file_list,
                                        landmark_files=landmark_list)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    prefix = '2023-05-23-22-00-43'
    model_dir = args.checkpoint_path

    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    model = model.to(device)

    if args.checkpoint_name is not None:
        # test_dirlab(args, os.path.join(model_dir, args.checkpoint_name), True)
        test_dirlab(args, os.path.join(model_dir, args.checkpoint_name))
    else:
        checkpoint_list = sorted([os.path.join(model_dir, file) for file in os.listdir(model_dir) if prefix in file])
        for checkpoint in checkpoint_list:
            print(checkpoint)
            # test_dirlab(args, checkpoint)
            test_dirlab(args, checkpoint)

