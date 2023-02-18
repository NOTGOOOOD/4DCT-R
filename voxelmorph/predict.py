import torch
import torch.utils.data as Data
import os

from utils.metric import MSE
from utils.utilize import save_image
from utils.datagenerators import Dataset
from voxelmorph.model.regnet import RegNet_pairwise
from utils.config import get_args


def do_test(args):
    test_fixed_folder = os.path.join(args.test_dir, 'fixed')
    test_moving_folder = os.path.join(args.test_dir, 'moving')
    test_fixed_list = sorted(
        [os.path.join(test_fixed_folder, file_name) for file_name in os.listdir(test_fixed_folder) if
         file_name.lower().endswith('.gz')])
    test_moving_list = sorted(
        [os.path.join(test_moving_folder, file_name) for file_name in os.listdir(test_moving_folder) if
         file_name.lower().endswith('.gz')])

    test_dataset = Dataset(moving_files=test_moving_list, fixed_files=test_fixed_list)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = RegNet_pairwise(3, scale=1, depth=5, initial_channels=args.initial_channels, normalization=True)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.checkpoint_name))['model'])
    model = model.to(args.device)

    with torch.no_grad():
        model.eval()
        for batch, (moving, fixed) in enumerate(test_loader):
            input_moving = moving[0].to('cuda').float()
            input_fixed = fixed[0].to('cuda').float()
            warped_image, flow, = model(input_fixed, input_moving)
            _mse = MSE(input_fixed, warped_image)

            # save warped image0
            m_name = "{}_warped.nii.gz".format(moving[1][0].split('.nii')[0])
            save_image(warped_image, input_fixed, args.output_dir, m_name)
            print('%s MSE=%.5f' % (m_name, _mse.item()))
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


if __name__ == '__main__':
    args = get_args()
    do_test(args)
