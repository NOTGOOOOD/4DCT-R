import torch
import torch.utils.data as Data
import os

from utils.metric import calc_tre, MSE
from utils.utilize import load_landmarks, save_image
from datagenerators import TestDataset
from voxelmorph.model.regnet import RegNet_pairwise
from config import get_args


def do_test(args):
    landmark_list = load_landmarks(args.landmark_dir)

    test_fixed_folder = os.path.join(args.test_dir, 'fixed')
    test_moving_folder = os.path.join(args.test_dir, 'moving')
    test_fixed_list = sorted(
        [os.path.join(test_fixed_folder, file_name) for file_name in os.listdir(test_fixed_folder) if
         file_name.lower().endswith('.gz')])
    test_moving_list = sorted(
        [os.path.join(test_moving_folder, file_name) for file_name in os.listdir(test_moving_folder) if
         file_name.lower().endswith('.gz')])

    test_dataset = TestDataset(moving_files=test_moving_list, fixed_files=test_fixed_list, landmark_files=landmark_list)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = RegNet_pairwise(3, scale=1, depth=5, initial_channels=args.initial_channels, normalization=True)
    model.load_state_dict(torch.load(args.checkpoint_path)['model'])
    model = model.to(args.device)

    with torch.no_grad():
        model.eval()
        losses = []
        for batch, (moving, fixed, landmarks) in enumerate(test_loader):
            input_moving = moving[0].to('cuda').float()
            input_fixed = fixed[0].to('cuda').float()

            landmarks00 = landmarks['landmark_00'].squeeze().cuda()
            # landmarks50 = landmarks['landmark_50'].squeeze().cuda()

            warped_image, flow,  = model(input_fixed, input_moving)
            flow_hr = flow[0]
            index = batch + 1

            crop_range = args.dirlab_cfg[index]['crop_range']

            # TRE
            _mean, _std = calc_tre(flow_hr, landmarks00 - torch.tensor(
                [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 1, 3).cuda(),
                                   landmarks['disp_00_50'].squeeze(), args.dirlab_cfg[index]['pixel_spacing'])

            # MSE
            _mse = MSE(input_fixed, warped_image)
            # _mean, _std = landmark_loss(flow_hr, landmarks00 - torch.tensor(
            #     [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 1, 3).cuda(),
            #                             landmarks50 - torch.tensor(
            #                                 [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 1,
            #                                                                                                       3).cuda(),
            #                             args.dirlab_cfg[index]['pixel_spacing'])

            losses.append([_mean.item(), _std.item(), _mse.item()])
            print('case=%d after warped, TRE=%.5f+-%.5f MSE=%.5f' % (index, _mean.item(), _std.item(), _mse.item()))

            # save warped image0
            m_name = "{}_warped.nii.gz".format(moving[1][0][:13])
            save_image(warped_image, input_fixed, args.output_dir, m_name)
            print("warped images have saved.")

            # # Save DVF
            # # b,3,d,h,w-> w,h,d,3  # maybe have sth. wrong, the shape getting from elastix is (d,h,w,3)
            # m2f_name = "{}_dvf.nii.gz".format(moving[1][:15])
            # # save_image(torch.permute(flow[0], (3, 2, 1, 0)), input_fixed, args.output_dir,
            # #            m2f_name)
            #
            # # b,3,d,h,w-> d,h,w,3
            # save_image(torch.permute(flow[0], (1, 2, 3, 0)), input_fixed, args.output_dir,
            #            m2f_name)
            # print("dvf have saved.")

        mean_tre = torch.mean(torch.tensor(losses), 0)[0]
        mean_std = torch.mean(torch.tensor(losses), 0)[1]
        print("mean TRE=%.5f+-%.5f" % (mean_tre.item(), mean_std.item()))


if __name__ == '__main__':
    args = get_args()
    do_test(args)
