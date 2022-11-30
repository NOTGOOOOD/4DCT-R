import os
import warnings

import torch
import numpy as np
from torch.optim import Adam
import torch.utils.data as Data

import losses
from config import get_args
from datagenerators import Dataset, TestDataset
from voxelmorph.model import regnet
from utils.scheduler import WarmupCosineSchedule
from utils.utilize import set_seed, save_image, save_model
from utils.metric import get_test_photo_loss

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


def load_landmarks():
    landmark_folder = args.landmark_dir
    landmarks = []
    for i in sorted(
            [os.path.join(landmark_folder, file) for file in os.listdir(landmark_folder) if file.endswith('.pt')]):
        landmarks.append(torch.load(i))

    return landmarks


def train():
    # set gpu
    make_dirs()
    landmark_list = load_landmarks()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

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

    # # VM and STN
    # nf_enc = [16, 32, 32, 32]
    # if args.model == "vm":
    #     nf_dec = [32, 32, 32, 32, 8, 8]
    # else:
    #     nf_dec = [32, 32, 32, 32, 32, 16, 16]  # vm2
    #
    # UNet = U_Network(3, nf_enc, nf_dec).to(device)
    # STN = SpatialTransformer([144, 256, 256]).to(device)
    # # STN2 = SpatialTransformer_new(3).to(device)
    # UNet.train()
    # STN.train()
    # # STN2.train()
    model = regnet.RegNet_pairwise(3, scale=1, depth=5, initial_channels=16, normalization=True)
    model = model.to(device)
    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter('runs/voxelmorph')
    # test_images = torch.randn(1, 1, 96, 256, 256)
    # writer.add_graph(UNet, [test_images.to(device), test_images.to(device)])
    # writer.close()
    # # 模型参数个数
    # print("UNet: ", count_parameters(UNet))

    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=args.lr)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # set scheduler
    scheduler = WarmupCosineSchedule(opt, warmup_steps=args.warmup_steps, t_total=args.n_iter)

    # load data
    train_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    test_dataset = TestDataset(moving_files=test_moving_list, fixed_files=test_fixed_list, landmark_files=landmark_list)
    print("Number of training images: ", len(train_dataset))
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best_tre = 99.
    # Training
    for i in range(1, args.n_iter + 1):
        model.train()
        loss_total = []
        print('iter:{} start'.format(i))
        for i_step, (moving_file, fixed_file) in enumerate(train_loader):
            # [B, C, D, H, W]
            input_moving = moving_file[0].to(device).float()
            input_fixed = fixed_file[0].to(device).float()

            # Run the data through the model to produce warp and flow field
            flow_m2f, warped_image = model(input_fixed, input_moving)  # b, c, d, h, w

            # Calculate loss
            sim_loss = sim_loss_fn(warped_image, input_fixed, [9] * 3)
            # sim_loss2 = sim_loss_fn(m2f_new, input_fixed, [9] * 3)
            grad_loss = grad_loss_fn(flow_m2f)  # b*c*h*w*d
            loss = sim_loss + args.alpha * grad_loss
            loss_total.append(loss.item())

            # moving_name = moving_file[1].split(r'moving\\')[1]
            # print("img_name:{}".format(moving_name))
            # print("iter: %d batch: %d  loss: %f  sim: %f  grad: %f" % (
            #     i, i_step, loss.item(), sim_loss.item(), grad_loss.item()), flush=True)

            # Backwards and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

        if i % args.n_save_iter == 0:
            # if test ncc is best ,save the model
            mean_tre = get_test_photo_loss(args, model, test_loader)
            if mean_tre < best_tre:
                best_tre = mean_tre
                save_model(args, model, opt, scheduler, i)

                # # Save images
                # moving_name = moving_file[1].split(r'moving\\')[1]
                # m_name = "{}_{}.nii.gz".format(i, moving_name)
                # save_image(warped_image, input_fixed, args.output_dir, m_name)
                # print("warped images have saved.")
                #
                # # Save DVF
                # # b,3,d,h,w-> w,h,d,3
                # m2f_name = str(i) + "_dvf.nii.gz"
                # save_image(torch.permute(flow_m2f[0], (3, 2, 1, 0)), input_fixed, args.output_dir,
                #            m2f_name)
                # print("dvf have saved.")

        print("iter:{}, mean loss:{}".format(i, loss_total.mean()))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    set_seed(1024)
    train()
