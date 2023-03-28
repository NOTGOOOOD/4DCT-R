import os
import numpy as np
import torch
import torch.utils.data as Data

from utils.Functions import generate_grid_unit, transform_unit_flow_to_flow
from LapIRN import Miccai2020_LDR_laplacian_unit_add_lvl1, Miccai2020_LDR_laplacian_unit_add_lvl2, \
    Miccai2020_LDR_laplacian_unit_add_lvl3, SpatialTransform_unit, neg_Jdet_loss
from utils.utilize import load_landmarks, save_image
from utils.config import get_args
from utils.metric import MSE, landmark_loss
from utils.datagenerators import TestDataset


def test(args, checkpoint, is_save=False):
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

            model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, args.initial_channels, is_train=True,
                                                                imgshape=imgshape_4,
                                                                range_flow=range_flow).cuda()
            model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3, args.initial_channels, is_train=True,
                                                                imgshape=imgshape_2,
                                                                range_flow=range_flow, model_lvl1=model_lvl1).cuda()

            model = Miccai2020_LDR_laplacian_unit_add_lvl3(2, 3, args.initial_channels, is_train=False,
                                                           imgshape=imgshape,
                                                           range_flow=range_flow, model_lvl2=model_lvl2).cuda()

            transform = SpatialTransform_unit().cuda()

            model.load_state_dict(torch.load(checkpoint))
            model.eval()
            transform.eval()

            grid = generate_grid_unit(imgshape)
            grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

            F_X_Y = model(moving_img, fixed_img)

            X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid)

            F_X_Y_cpu = F_X_Y[0, :, :, :, :]
            F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

            Jac = neg_Jdet_loss(F_X_Y_cpu.unsqueeze(0).permute(0, 2, 3, 4, 1), grid)
            crop_range = args.dirlab_cfg[batch + 1]['crop_range']

            # MSE
            _mse = MSE(fixed_img, X_Y)

            # TRE
            _mean, _std = landmark_loss(F_X_Y[0], landmarks00 - torch.tensor(
                [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
                                        landmarks50 - torch.tensor(
                                            [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1,
                                                                                                                  3).cuda(),
                                        args.dirlab_cfg[batch + 1]['pixel_spacing'])

            losses.append([_mean.item(), _std.item(), _mse.item()])
            print('case=%d after warped, TRE=%.5f+-%.5f MSE=%.5f Jac=%.6f' % (
                batch + 1, _mean.item(), _std.item(), _mse.item(), Jac.item()))

            if is_save:
                # Save DVF
                # b,3,d,h,w-> d,h,w,3    (dhw or whd) depend on the shape of image
                m2f_name = img_name[0][:13] + '_warpped_flow.nii.gz'
                save_image(torch.permute(F_X_Y_cpu, (1, 2, 3, 0)), fixed_img[0], args.output_dir,
                           m2f_name)
                m_name = "{}_warped_lapirn.nii.gz".format(img_name[0][:13])
                # save_img(X_Y, args.output_dir + '/' + file_name + '_warpped_moving.nii.gz')
                save_image(X_Y, fixed_img, args.output_dir, m_name)


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

    test_dataset = TestDataset(moving_files=m_img_file_list, fixed_files=f_img_file_list, landmark_files=landmark_list)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    prefix = '2023-02-01-19-38-09'
    model_dir = args.checkpoint_path
    checkpoint_list = sorted([os.path.join(model_dir, file) for file in os.listdir(model_dir) if prefix in file])
    for checkpoint in checkpoint_list:
        print(checkpoint)
        test(args, checkpoint)
