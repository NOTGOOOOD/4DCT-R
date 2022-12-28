import os
import platform
import numpy as np
import torch

from Functions import generate_grid_unit, save_img, save_flow, transform_unit_flow_to_flow, load_4D
from miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3, SpatialTransform_unit
from utils.utilize import load_landmarks
from utils.config import get_args
from utils.metric import calc_tre, MSE
from voxelmorph.process.processing import data_standardization_0_n

args = get_args()

savepath = args.output_dir

if not os.path.isdir(savepath):
    os.mkdir(savepath)

start_channel = args.initial_channels


def test():
    imgshape = [144, 144, 144]
    imgshape_4 = (imgshape[0] / 4, imgshape[1] / 4, imgshape[2] / 4)
    imgshape_2 = (imgshape[0] / 2, imgshape[1] / 2, imgshape[2] / 2)

    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                             range_flow=range_flow).cuda()
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                                             range_flow=range_flow, model_lvl1=model_lvl1).cuda()

    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=False, imgshape=imgshape,
                                                        range_flow=range_flow, model_lvl2=model_lvl2).cuda()

    transform = SpatialTransform_unit().cuda()

    model.load_state_dict(torch.load(args.modelpath))
    model.eval()
    transform.eval()

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    fixed_folder = os.path.join(args.test_dir, 'fixed')
    moving_folder = os.path.join(args.test_dir, 'moving')
    f_img_file_list = sorted([os.path.join(fixed_folder, file_name) for file_name in os.listdir(fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_img_file_list = sorted([os.path.join(moving_folder, file_name) for file_name in os.listdir(moving_folder) if
                              file_name.lower().endswith('.gz')])

    fixed_img = load_4D(f_img_file_list[0])
    moving_img = load_4D(m_img_file_list[0])

    fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)
    moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)

    with torch.no_grad():
        F_X_Y = model(moving_img, fixed_img)

        X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]

        F_X_Y_cpu = F_X_Y.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
        F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

        save_flow(F_X_Y_cpu, savepath + '/warpped_flow.nii.gz')
        save_img(X_Y, savepath + '/warpped_moving.nii.gz')

    print("Finished")


def test_single():
    landmark_list = load_landmarks(args.landmark_dir)
    fixed_folder = os.path.join(args.test_dir, 'fixed')
    moving_folder = os.path.join(args.test_dir, 'moving')
    f_img_file_list = sorted([os.path.join(fixed_folder, file_name) for file_name in os.listdir(fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_img_file_list = sorted([os.path.join(moving_folder, file_name) for file_name in os.listdir(moving_folder) if
                              file_name.lower().endswith('.gz')])

    # respectively
    losses = []
    for i in range(len(f_img_file_list)):
        file_name = m_img_file_list[i].split('moving\\')[1] if platform.system().lower() == 'windows' else \
            m_img_file_list[i].split('moving/')[1]
        file_name = file_name[:13]

        fixed_img = load_4D(f_img_file_list[i])
        moving_img = load_4D(m_img_file_list[i])
        fixed_img = torch.from_numpy(fixed_img).float().to(args.device).unsqueeze(dim=0)
        moving_img = torch.from_numpy(moving_img).float().to(args.device).unsqueeze(dim=0)

        fixed_img = data_standardization_0_n(1, fixed_img)
        moving_img = data_standardization_0_n(1, moving_img)

        landmarks = landmark_list[i]

        imgshape = fixed_img.shape[2:]

        imgshape_4 = (imgshape[0] / 4, imgshape[1] / 4, imgshape[2] / 4)
        imgshape_2 = (imgshape[0] / 2, imgshape[1] / 2, imgshape[2] / 2)

        model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True,
                                                                 imgshape=imgshape_4,
                                                                 range_flow=range_flow).cuda()
        model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True,
                                                                 imgshape=imgshape_2,
                                                                 range_flow=range_flow, model_lvl1=model_lvl1).cuda()

        model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=False, imgshape=imgshape,
                                                            range_flow=range_flow, model_lvl2=model_lvl2).cuda()

        transform = SpatialTransform_unit().cuda()

        model.load_state_dict(torch.load(args.checkpoint_path))
        model.eval()
        transform.eval()

        grid = generate_grid_unit(imgshape)
        grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

        with torch.no_grad():
            F_X_Y = model(moving_img, fixed_img)

            X_Y = transform(moving_img, F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]

            F_X_Y_cpu = F_X_Y.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
            F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

            crop_range = args.dirlab_cfg[i+1]['crop_range']

            landmarks00 = torch.tensor(landmarks['landmark_00']).cuda()

            # TRE
            _mean, _std = calc_tre(torch.tensor(F_X_Y_cpu).permute(3,2,1,0), landmarks00 - torch.tensor(
                [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 1, 3).cuda(),
                                   torch.tensor(landmarks['disp_00_50']).squeeze(), args.dirlab_cfg[i+1]['pixel_spacing'])
            # _mean, _std = calc_tre(flow_hr, landmarks00 - torch.tensor(
            #     [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 1, 3).cuda(),
            #                        landmarks['disp_affine'].squeeze(), args.dirlab_cfg[index]['pixel_spacing'])

            # MSE
            _mse = MSE(fixed_img.squeeze(), torch.tensor(X_Y))
            # _mean, _std = landmark_loss(flow_hr, landmarks00 - torch.tensor(
            #     [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 1, 3).cuda(),
            #                             landmarks50 - torch.tensor(
            #                                 [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 1,
            #                                                                                                       3).cuda(),
            #                             args.dirlab_cfg[index]['pixel_spacing'])

            losses.append([_mean.item(), _std.item(), _mse.item()])
            print('case=%d after warped, TRE=%.5f+-%.5f MSE=%.5f' % (i+1, _mean.item(), _std.item(), _mse.item()))

            # save_flow(F_X_Y_cpu, savepath + '/' + file_name + '_warpped_flow.nii.gz')
            save_img(X_Y, savepath + '/' + file_name + '_warpped_moving.nii.gz')

        print("Finished")


if __name__ == '__main__':
    range_flow = 0.4
    test_single()
