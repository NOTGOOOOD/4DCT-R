import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import numpy as np
import torch
import torch.utils.data as Data
import logging
import time

from utils.config import get_args
from utils.scheduler import StopCriterion
from utils.utilize import set_seed, save_model

set_seed(20)

from network import CubicBSplineNet
from loss import LNCCLoss, l2reg_loss
from utils.datagenerators import Dataset
from utils.Functions import validation_midir, generate_grid
from utils.losses import NCC, neg_Jdet_loss
from transformation import CubicBSplineFFDTransform, warp


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


def train():
    print("Training midir...")
    device = args.device
    img_shape = [144, 192, 160]
    cps = (4, 4, 4)
    model = CubicBSplineNet(ndim=3,
                            img_size=img_shape,
                            cps=cps).to(device)
    model_path = ''
    # model_path = r'D:\xxf\4DCT-R\midir\model\2023-03-19-23-54-14_midir__125_-0.3505.pth'
    if len(model_path) > 1:
        print("load model: ", model_path)
        model.load_state_dict(torch.load(model_path)['model'])

    # loss_similarity = LNCCLoss(window_size=7)
    loss_similarity = NCC(win=7)
    transformer = CubicBSplineFFDTransform(ndim=3, img_size=img_shape, cps=(4, 4, 4), svf=True
                                           , svf_steps=7
                                           , svf_scale=1)

    # weight for l2 reg
    reg_weight = 0.08

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                             step_size=100,
    #                                             gamma=0.1,
    #                                             last_epoch=-1)

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
    #                                      shuffle=True, num_workers=2)
    train_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    stop_criterion = StopCriterion()
    step = 0
    best_loss = 99.

    while step <= iteration:
        lossall = []
        model.train()
        for batch, (moving, fixed) in enumerate(train_loader):
            X = moving[0].to(device).float()
            Y = fixed[0].to(device).float()

            # compose_field_e0_lvl1, warpped_inputx_lvl1_out,warpped_inputx_lvl2_out,warpped_inputx_lvl3_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
            svf = model(Y, X)  # b,c,d,h,w
            flow, disp = transformer(svf)
            wapred_x = warp(X, disp)

            loss_ncc = loss_similarity(wapred_x, Y)
            loss_reg = l2reg_loss(disp)

            grid = generate_grid(img_shape)
            grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

            loss_Jacobian = neg_Jdet_loss(disp.permute(0, 2, 3, 4, 1), grid)

            loss = loss_ncc + loss_reg * reg_weight

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            # lossall[:, step] = np.array(
            #     [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            lossall.append([loss.item(), loss_ncc.item(), loss_Jacobian.item(), loss_reg.item()])

            sys.stdout.write(
                "\r" + 'midir:step:batch "{0}:{1}" -> training loss "{2:.4f}" - sim_NCC "{3:4f}" - Jdet "{4:.10f}" -smo "{5:.4f}"'.format(
                    step, batch, loss.item(), loss_ncc.item(), loss_Jacobian.item(), loss_reg.item()))
            sys.stdout.flush()

            # if batch == 0:
            #     m_name = 'l3_' + str(step) + 'moving_' + moving[1][0]
            #     save_image(X_Y, Y, args.output_dir, m_name)
            #     m_name = 'l3_' + str(step) + 'fixed_' + moving[1][0]
            #     save_image(Y_4x, Y, args.output_dir, m_name)

        # validation
        val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss = validation_midir(args, model, img_shape,
                                                                                    loss_similarity)
        # scheduler.step()

        mean_loss = np.mean(np.array(lossall), 0)[0]
        print(
            "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                mean_loss, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        stop_criterion.add(val_ncc_loss, val_jac_loss, val_total_loss, train_loss=mean_loss)

        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            # modelname = model_dir + '/' + model_name + "{:.4f}_stagelvl3_".format(best_loss) + str(step) + '.pth'
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(step) + '{:.4f}best.pth'.format(
                val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)
        else:
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(step) + '{:.4f}.pth'.format(
                val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        if stop_criterion.stop():
            break

        step += 1
        if step > iteration:
            break


if __name__ == "__main__":
    args = get_args()

    lr = args.lr

    iteration = 200000

    fixed_folder = os.path.join(args.train_dir, 'fixed')
    moving_folder = os.path.join(args.train_dir, 'moving')
    f_img_file_list = sorted([os.path.join(fixed_folder, file_name) for file_name in os.listdir(fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_img_file_list = sorted([os.path.join(moving_folder, file_name) for file_name in os.listdir(moving_folder) if
                              file_name.lower().endswith('.gz')])

    make_dirs()

    log_index = len([file for file in os.listdir(args.log_dir) if file.endswith('.txt')])

    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = "{}_midir_".format(train_time)

    logging.basicConfig(level=logging.INFO,
                        filename=f'Log/log{log_index}.txt',
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    # size = [144,192,160] # z y x
    # imgshape = (size[0], size[1], size[2])
    # imgshape_4 = (size[0] / 4,  size[1] / 4, size[2] / 4)
    # imgshape_2 = (size[0] / 2,  size[1] / 2, size[2] / 2)
    train()
