import glob
import os
import sys
from argparse import ArgumentParser
import time

from utils.scheduler import StopCriterion
from utils.utilize import set_seed
set_seed(20)

import numpy as np
import torch
import torch.utils.data as Data

from utils.Functions import generate_grid, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit, Grid, validation_lapirn_ori, smoothloss,SpatialTransform_unit
from miccai2021_model import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
    Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2, Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3

from utils.datagenerators import Dataset
from utils.config import get_args
from utils.losses import NCC, multi_resolution_NCC, neg_Jdet_loss
from utils.utilize import save_model

# parser = ArgumentParser()
# parser.add_argument("--lr", type=float,
#                     dest="lr", default=1e-4, help="learning rate")
# parser.add_argument("--iteration_lvl1", type=int,
#                     dest="iteration_lvl1", default=30001,
#                     help="number of lvl1 iterations")
# parser.add_argument("--iteration_lvl2", type=int,
#                     dest="iteration_lvl2", default=30001,
#                     help="number of lvl2 iterations")
# parser.add_argument("--iteration_lvl3", type=int,
#                     dest="iteration_lvl3", default=60001,
#                     help="number of lvl3 iterations")
# parser.add_argument("--antifold", type=float,
#                     dest="antifold", default=100.,
#                     help="Anti-fold loss: suggested range 1 to 10000")
# parser.add_argument("--checkpoint", type=int,
#                     dest="checkpoint", default=5000,
#                     help="frequency of saving models")
# parser.add_argument("--start_channel", type=int,
#                     dest="start_channel", default=7,  # default:8, 7 for stage
#                     help="number of start channels")
# parser.add_argument("--datapath", type=str,
#                     dest="datapath",
#                     default='../Dataset/Brain_dataset/OASIS/crop_min_max/norm',
#                     help="data path for training images")
# parser.add_argument("--freeze_step", type=int,
#                     dest="freeze_step", default=3000,
#                     help="Number of step to freeze the previous level")
# opt = parser.parse_args()



def train_lvl1():
    print("Training lvl1...")
    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                                    range_flow=range_flow).cuda()

    loss_similarity = NCC(win=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    grid_4 = generate_grid(imgshape_4)
    grid_4 = torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).cuda().float()


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model/Stage'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    train_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    stop_criterion = StopCriterion()
    step = 0

    best_loss = 99.

    while step <= iteration_lvl1:
        lossall = []
        model.train()
        for batch, (moving, fixed) in enumerate(train_loader):

            X = moving[0].to(device).float()
            Y = fixed[0].to(device).float()
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y, reg_code)

            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

            _, _, x, y, z = F_X_Y.shape
            norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 2, 0, 0, 0] = (z-1)
            norm_vector[0, 1, 0, 0, 0] = (y-1)
            norm_vector[0, 0, 0, 0, 0] = (x-1)
            loss_regulation = loss_smooth(F_X_Y * norm_vector)

            smo_weight = reg_code * max_smooth
            loss = loss_multiNCC + antifold * loss_Jacobian + smo_weight * loss_regulation
            lossall.append([loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            sys.stdout.write(
                "\r" + 'lv1:step:batch "{0}:{1}" -> training loss "{2:.4f}" - sim_NCC "{3:4f}" - Jdet "{4:.10f}" -smo "{5:.4f}"'.format(
                    step, batch, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()

        # validation
        val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss = validation_lapirn_ori(args, model, imgshape_4,loss_similarity,
                                                                                       imgshape)

        mean_loss = np.mean(np.array(lossall), 0)[0]
        print(
            "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                mean_loss, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        stop_criterion.add(val_ncc_loss, val_jac_loss, val_total_loss, train_loss=mean_loss)

        # save model
        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            modelname = model_dir + '/' + model_name + "stagelvl1" + '_{:03d}_'.format(step) + '{:.4f}.pth'.format(
                best_loss)
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        if stop_criterion.stop():
            break

        step += 1
        if step > iteration_lvl1:
            break



def train_lvl2():
    print("Training lvl2...")
    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True,
                                                                         imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()

    model_list = []
    for f in os.listdir('../Model/Stage'):
        if model_name + "stagelvl1" in f:
            model_list.append(os.path.join('../Model/Stage', f))

    model_path = sorted(model_list)[-1]

    model_lvl1.load_state_dict(torch.load(model_path)['model'])
    print("Loading weight for model_lvl1...", model_path)
    # Freeze model_lvl1 weight
    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                                                    range_flow=range_flow, model_lvl1=model_lvl1).cuda()

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True


    grid_2 = generate_grid(imgshape_2)
    grid_2 = torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stop_criterion = StopCriterion()
    model_dir = '../Model/Stage'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    train_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    step = 0
    best_loss = 99.

    while step <= iteration_lvl2:
        lossall = []
        model.train()
        for batch, (moving, fixed) in enumerate(train_loader):

            X = moving[0].to(device).float()
            Y = fixed[0].to(device).float()
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y, reg_code)

            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            _, _, x, y, z = F_X_Y.shape
            norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 2, 0, 0, 0] = (z-1)
            norm_vector[0, 1, 0, 0, 0] = (y-1)
            norm_vector[0, 0, 0, 0, 0] = (x-1)
            loss_regulation = loss_smooth(F_X_Y * norm_vector)

            smo_weight = reg_code * max_smooth
            loss = loss_multiNCC + antifold * loss_Jacobian + smo_weight * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall.append([loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'lv2:step:batch "{0}:{1}" -> training loss "{2:.4f}" - sim_NCC "{3:4f}" - Jdet "{4:.10f}" -smo "{5:.4f}"'.format(
                    step, batch, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()

        # validation
        val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss = validation_lapirn_ori(args, model, imgshape_2,
                                                                                         loss_similarity,
                                                                                         imgshape)

        mean_loss = np.mean(np.array(lossall), 0)[0]
        print(
            "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                mean_loss, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        stop_criterion.add(val_ncc_loss, val_jac_loss, val_total_loss, train_loss=mean_loss)

        # save model
        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            modelname = model_dir + '/' + model_name + "stagelvl2" + '_{:03d}_'.format(step) + '{:.4f}.pth'.format(
                best_loss)

            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        if stop_criterion.stop():
            break

        if step == freeze_step:
            model.unfreeze_modellvl1()

        step += 1
        if step > iteration_lvl2:
            break



def train_lvl3():
    print("Training lvl3...")
    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True,
                                                                         imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()
    model_lvl2 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True,
                                                                         imgshape=imgshape_2,
                                                                         range_flow=range_flow, model_lvl1=model_lvl1).cuda()

    model_list = []
    for f in os.listdir('../Model/Stage'):
        if model_name + "stagelvl2" in f:
            model_list.append(os.path.join('../Model/Stage', f))

    model_path = sorted(model_list)[-1]
    model_lvl2.load_state_dict(torch.load(model_path)['model'])
    print("Loading weight for model_lvl2...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=True, imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2).cuda()

    loss_similarity = multi_resolution_NCC(win=7, scale=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True


    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    grid_unit = generate_grid_unit(imgshape)
    grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    train_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    stop_criterion = StopCriterion()
    step = 0
    best_loss = 99.

    while step <= iteration_lvl3:
        lossall = []
        model.train()
        for batch, (moving, fixed) in enumerate(train_loader):

            X = moving[0].to(device).float()
            Y = fixed[0].to(device).float()
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y, reg_code)

            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

            _, _, x, y, z = F_X_Y.shape
            norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 2, 0, 0, 0] = (z-1)
            norm_vector[0, 1, 0, 0, 0] = (y-1)
            norm_vector[0, 0, 0, 0, 0] = (x-1)
            loss_regulation = loss_smooth(F_X_Y * norm_vector)

            smo_weight = reg_code * max_smooth
            loss = loss_multiNCC + antifold * loss_Jacobian + smo_weight * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall.append([loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])

            sys.stdout.write(
                "\r" + 'lv3:step:batch "{0}:{1}" -> training loss "{2:.4f}" - sim_NCC "{3:4f}" - Jdet "{4:.10f}" -smo "{5:.4f}"'.format(
                    step, batch, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()

        # validation
        val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss = validation_lapirn_ori(args, model, imgshape,
                                                                                         loss_similarity,
                                                                                         imgshape)

        mean_loss = np.mean(np.array(lossall), 0)[0]
        print(
            "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                mean_loss, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        stop_criterion.add(val_ncc_loss, val_jac_loss, val_total_loss, train_loss=mean_loss)

        # save model
        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            # modelname = model_dir + '/' + model_name + "{:.4f}_stagelvl3_".format(best_loss) + str(step) + '.pth'
            modelname = model_dir + '/' + model_name + "stagelvl3" + '_{:03d}_'.format(
                step) + '{:.4f}best.pth'.format(
                val_total_loss)

            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)
        else:
            modelname = model_dir + '/' + model_name + "stagelvl3" + '_{:03d}_'.format(step) + '{:.4f}.pth'.format(
                val_total_loss)

            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        if stop_criterion.stop():
            break

        if step == freeze_step:
            model.unfreeze_modellvl2()

        step += 1
        if step > iteration_lvl3:
            break



def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

if __name__ == "__main__":
    args = get_args()

    lr = args.lr
    device = args.device
    start_channel = args.initial_channels
    antifold = args.antifold
    # n_checkpoint = args.n_save_iter
    smooth = args.smooth
    # datapath = opt.datapath
    freeze_step = args.freeze_step

    iteration_lvl1 = args.iteration_lvl1
    iteration_lvl2 = args.iteration_lvl2
    iteration_lvl3 = args.iteration_lvl3
    max_smooth = 10.

    fixed_folder = os.path.join(args.train_dir, 'fixed')
    moving_folder = os.path.join(args.train_dir, 'moving')
    f_img_file_list = sorted([os.path.join(fixed_folder, file_name) for file_name in os.listdir(fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_img_file_list = sorted([os.path.join(moving_folder, file_name) for file_name in os.listdir(moving_folder) if
                              file_name.lower().endswith('.gz')])

    make_dirs()
    log_index = len([file for file in os.listdir(args.log_dir) if file.endswith('.txt')])

    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = "{}_cLapIRN_".format(train_time)

    size = [144, 192, 160]  # z y x
    imgshape = (size[0], size[1], size[2])
    imgshape_4 = (size[0] / 4, size[1] / 4, size[2] / 4)
    imgshape_2 = (size[0] / 2, size[1] / 2, size[2] / 2)

    grid_class = Grid()
    range_flow = 0.4
    train_lvl1()
    train_lvl2()
    train_lvl3()