import os
import warnings
import platform
import torch
import numpy as np
from torch.optim import SGD
import torch.utils.data as Data
from tqdm import tqdm
import time
import copy

from utils.Functions import SpatialTransform_unit, generate_grid, test_dirlab
from utils.losses import NCC, gradient_loss, neg_Jdet_loss, Grad
from utils.config import get_args
from utils.datagenerators import Dataset, PatientDataset, DirLabDataset, build_dataloader_dirlab
from ResUNet import ResUnetModel
from utils.scheduler import StopCriterion
from utils.utilize import set_seed, save_model, save_image, count_parameters, load_landmarks, make_dirs
from utils.metric import MSE, jacobian_determinant, SSIM, NCC as calc_NCC, landmark_loss

@torch.no_grad()
def validation(args, model, loss_similarity):
    model.eval()  # m_name = "{}_affine.nii.gz".format(moving[1][0][:13])
    losses = []
    for batch, (moving, fixed) in enumerate(val_loader):
        input_moving = moving[0].to('cuda').float()
        input_fixed = fixed[0].to('cuda').float()

        res = model(input_moving,input_fixed)  # b, c, d, h, w  disp, scale_disp, warped
        warped_image, flow = res['warped_img'], res['flow']

        mse_loss = MSE(warped_image, input_fixed)
        ncc_loss_ori = loss_similarity(warped_image, input_fixed)
        grad_loss = args.alpha * gradient_loss(flow)
        loss_sum = ncc_loss_ori + grad_loss

        losses.append([ncc_loss_ori.item(), mse_loss.item(), grad_loss.item(), loss_sum.item()])

    mean_loss = np.mean(losses, 0)
    return mean_loss[0], mean_loss[1], mean_loss[2], mean_loss[3]


def train_unet(model):
    """
    uent 系列都用这个
    Returns
    -------

    """
    print(count_parameters(model))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    image_loss_func = NCC(win=args.win_size)
    reg_loss = Grad('l2', loss_mult=2).loss

    stop_criterion = StopCriterion(min_epoch=300)
    best_loss = 99.
    # Training
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

            res = model(input_moving, input_fixed)  # b, c, d, h, w  disp, scale_disp, warped
            warped_image, flow_m2f = res['warped_img'], res['flow']

            loss_list = []
            sim_loss = image_loss_func(warped_image, input_fixed)
            grad_loss = args.alpha * reg_loss(None, flow_m2f)  # b*c*h*w*d
            loss = grad_loss + sim_loss
            loss_list.append(grad_loss.item())
            loss_list.append(sim_loss.item())

            loss_total.append(loss.item())

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f, grad=%.5f)" % (
                    i_step, len(train_loader), loss.item(), grad_loss.item())
            )
            # Backwards and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss = validation(args, model, image_loss_func)
        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(i) + '{:.4f}best.pth'.format(
                val_total_loss)
        else:
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(i) + '{:.4f}.pth'.format(
                val_total_loss)

        save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                   stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)
        print("save model:{}".format(modelname))

        mean_loss = np.mean(np.array(loss_total), 0)
        print(
            "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                mean_loss, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        mean_tre = test_dirlab(args, model, test_loader_dirlab)

        stop_criterion.add(val_ncc_loss, val_jac_loss, val_total_loss, train_loss=mean_loss)
        if stop_criterion.stop():
            break


def test_unet(model):
    prefix = '2023-04-21-17-47-16'
    model_dir = args.checkpoint_path

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


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    args = get_args()
    model_dir = args.checkpoint_path
    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = "{}_resunet_lr{}_".format(train_time, args.lr)
    set_seed(42)
    make_dirs(args)
    device = args.device

    # load data
    train_loader = build_dataloader_dirlab(args, "train")
    val_loader = build_dataloader_dirlab(args, mode="val")
    test_loader_dirlab = build_dataloader_dirlab(args, mode="test")

    model = ResUnetModel()
    model = model.to(device)
    train_unet(model)
    # test_unet(model)
