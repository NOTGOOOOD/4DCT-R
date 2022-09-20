import os
import warnings

import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
import utils.utilize as ut

import losses
from config import args
from datagenerators import Dataset
from model import U_Network, SpatialTransformer, SpatialTransformer_new
from utils.utilize import tre, save_image


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)



def train():
    # 创建需要的文件夹并指定gpu
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # 日志文件
    log_name = str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

    # 读取相应的文件
    # case = 2
    # data_folder = os.path.join(project_path.split("4DCT")[0], f'datasets/dirlab/Case{case}_mhd/')
    project_path = ut.get_project_path("4DCT")
    fixed_folder = os.path.join(project_path.split("4DCT")[0], f'datasets/registration/fixed/')
    moving_folder = os.path.join(project_path.split("4DCT")[0], f'datasets/registration/moving/')
    f_img_file_list = sorted([file_name for file_name in os.listdir(fixed_folder) if file_name.lower().endswith('mhd')])
    m_img_file_list = sorted([file_name for file_name in os.listdir(moving_folder) if file_name.lower().endswith('mhd')])


    # 创建配准网络（UNet）和STN
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]   # vm2
    temp = [112, 256, 256]
    UNet = U_Network(3, nf_enc, nf_dec).to(device)
    STN = SpatialTransformer(temp).to(device)
    STN2 = SpatialTransformer_new(3).to(device)
    UNet.train()
    STN.train()
    STN2.train()

    # 模型参数个数
    print("UNet: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))
    print("STN2: ", count_parameters(STN2))

    # Set optimizer and losses
    opt = Adam(UNet.parameters(), lr=args.lr)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # Get all the names of the training data
    # train_files = glob.glob(os.path.join(args.train_dir, '*.mhd'))

    DS = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Training loop.
    for i in range(1, args.n_iter + 1):

        for i_step, (input_moving, input_fixed) in enumerate(DL):
            if i_step > args.n_iter:
                break

            # [B, C, D, W, H]
            input_moving = input_moving.to(device).float()
            input_fixed = input_fixed.to(device).float()

            # Run the data through the model to produce warp and flow field
            flow_m2f = UNet(input_moving, input_fixed)
            m2f = STN(input_moving, flow_m2f)
            m2f_new = STN2(input_moving, flow_m2f)

            # Calculate loss
            sim_loss = sim_loss_fn(m2f, input_fixed, [9]*3)
            sim_loss2 = sim_loss_fn(m2f_new, input_fixed, [9]*3)
            grad_loss = grad_loss_fn(flow_m2f)
            loss = sim_loss + args.alpha * grad_loss
            tre_score = tre()
            print("i: %d  loss: %f  sim: %f  sim2: %f  grad: %f" % (i, loss.item(), sim_loss.item(), sim_loss2.item(), grad_loss.item()), flush=True)
            print("%d, %f, %f, %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), file=f)

            # Backwards and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % args.n_save_iter == 0:
                # Save model checkpoint
                save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
                torch.save(UNet.state_dict(), save_file_name)
                # Save images
                m_name = str(i) + "_m.nii.gz"
                m2f_name = str(i) + "_m2f.nii.gz"
                save_image(input_moving, input_fixed, m_name)
                save_image(m2f, input_fixed, m2f_name)
                print("warped images have saved.")

    f.close()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
