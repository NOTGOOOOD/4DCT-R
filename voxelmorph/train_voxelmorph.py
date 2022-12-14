import os
import warnings

import torch
import numpy as np
import torch.utils.data as Data
from tqdm import tqdm
import logging
import time

from config import get_args
from datagenerators import Dataset, TestDataset
from voxelmorph.vmmodel import vmnetwork
from voxelmorph.vmmodel.losses import NCC, Grad, MSE
from utils.utilize import set_seed, load_landmarks

args = get_args()


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


def train():
    # set gpu
    landmark_list = load_landmarks(args.landmark_dir)
    device = args.device
    # load file
    fixed_folder = os.path.join(args.train_dir, 'fixed')
    moving_folder = os.path.join(args.train_dir, 'moving')
    f_img_file_list = sorted([os.path.join(fixed_folder, file_name) for file_name in os.listdir(fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_img_file_list = sorted([os.path.join(moving_folder, file_name) for file_name in os.listdir(moving_folder) if
                              file_name.lower().endswith('.gz')])

    test_fixed_folder = os.path.join(args.test_dir, 'fixed')
    test_moving_folder = os.path.join(args.test_dir, 'moving')
    test_fixed_list = sorted(
        [os.path.join(test_fixed_folder, file_name) for file_name in os.listdir(test_fixed_folder) if
         file_name.lower().endswith('.gz')])
    test_moving_list = sorted(
        [os.path.join(test_moving_folder, file_name) for file_name in os.listdir(test_moving_folder) if
         file_name.lower().endswith('.gz')])

    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]
    model = vmnetwork.VxmDense(
        inshape=[144, 144, 144],
        nb_unet_features=[enc_nf, dec_nf],
        bidir=args.bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )
    model = model.to(device)

    # Set optimizer and losses
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # prepare image loss
    if args.sim_loss == 'ncc':
        image_loss_func = NCC().loss
    elif args.sim_loss == 'mse':
        image_loss_func = MSE().loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

    # need two image loss functions if bidirectional
    if args.bidir:
        losses = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses = [image_loss_func]
        weights = [1]

    # prepare deformation loss
    losses += [Grad('l2', loss_mult=args.int_downsize).loss]
    weights += [
        args.weight]  # parser.add_argument('--lambda', type=float, dest='weight', default=0.01, help='weight of deformation loss (default: 0.01)')

    # # set scheduler
    # scheduler = WarmupCosineSchedule(opt, warmup_steps=args.warmup_steps, t_total=args.n_iter)
    # stop_criterion = StopCriterion(stop_std=args.stop_std, query_len=args.stop_query_len)

    # load data
    train_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    test_dataset = TestDataset(moving_files=test_moving_list, fixed_files=test_fixed_list, landmark_files=landmark_list)
    print("Number of training images: ", len(train_dataset))
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best_tre = 99.
    # Training
    # fold-validation
    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    for i in range(1, args.n_iter + 1):
        model.train()
        loss_total = []
        print('iter:{} start'.format(i))

        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for i_step, (moving_file, fixed_file) in enumerate(epoch_iterator):
            # [B, C, D, H, W]
            input_moving = moving_file[0].to(device).float()
            input_fixed = fixed_file[0].to(device).float()

            # Run the data through the model to produce warp and flow field
            warped_image, flow_m2f = model(input_moving, input_fixed)  # b, c, d, h, w

            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                curr_loss = loss_function(input_fixed[n], warped_image[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss

            loss_total.append(loss.item())

            moving_name = moving_file[1][0].split('moving/')[1]
            logging.info("img_name:{}".format(moving_name))
            logging.info("iter: %d batch: %d  loss: %.5f  sim: %.5f  grad: %.5f" % (
                i, i_step, loss.item(), loss_list[0].item(), loss_list[1].item()))

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (i_step, len(train_loader), loss.item())
            )
            # Backwards and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if i % args.n_save_iter == 0:
            #     # save warped image0
            #     m_name = "{}_{}.nii.gz".format(i, moving_name)
            #     save_image(warped_image, input_fixed, args.output_dir, m_name)
            #     print("warped images have saved.")
            #
            #     # Save DVF
            #     # b,3,d,h,w-> w,h,d,3
            #     m2f_name = str(i) + "_dvf.nii.gz"
            #     save_image(torch.permute(flow_m2f[0], (3, 2, 1, 0)), input_fixed, args.output_dir,
            #                m2f_name)
            #     print("dvf have saved.")

        # losses_test = get_test_photo_loss(args, logging, model, test_loader)
        # mean_tre = torch.mean(torch.tensor(losses_test), 0)[0]
        # mean_std = torch.mean(torch.tensor(losses_test), 0)[1]
        # mean_mse = torch.mean(torch.tensor(losses_test), 0)[2]
        #
        # if mean_tre < best_tre and best_tre - mean_tre > 0.01:
        #     best_tre = mean_tre
        #     save_model(args, model, optimizer, None, train_time)
        #     logging.info("best tre{}".format(losses_test))
        #
        # print("iter: %d, mean loss:%2.5f, test tre:%2.5f+-%2.5f, test mse:%2.5f" % (
        #     i, np.mean(loss_total), mean_tre.item(), mean_std.item(), mean_mse.item()))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    set_seed(1024)
    make_dirs()
    log_index = len([file for file in os.listdir(args.log_dir) if file.endswith('.txt')])

    logging.basicConfig(level=logging.INFO,
                        filename=f'Log/log{log_index}.txt',
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    train()
