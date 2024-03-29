import os
import sys
import numpy as np
import torch
import torch.utils.data as Data
import logging
import time

from utils.utilize import set_seed,save_model

set_seed(20)

from utils.Functions import generate_grid, get_loss, validation_ccregnet, Grid
from CCENet import CCENet_lv1, CCENet_lv2, CCENet_lv3

from utils.datagenerators import Dataset
from utils.config import get_args
from utils.losses import NCC, smoothloss, multi_resolution_NCC, neg_Jdet_loss
from utils.scheduler import StopCriterion


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def make_dirs():
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


def train_lvl1():
    print("Training lvl1...")
    device = args.device

    model = CCRegNet_lv1(2, 3, start_channel, is_train=True,
                         range_flow=range_flow, grid=grid_class).to(device)

    loss_similarity = NCC(win=3)
    loss_Jdet = neg_Jdet_loss
    loss_smooth = smoothloss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model/Stage'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
    #                                      shuffle=True, num_workers=2)

    train_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    stop_criterion = StopCriterion()
    step = 0
    # load_model = False
    # if load_model is True:
    #     model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
    #     print("Loading weight: ", model_path)
    #     step = 3000
    #     model.load_state_dict(torch.load(model_path)['model'])
    #     temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
    #     lossall[:, 0:3000] = temp_lossall[:, 0:3000]
    best_loss = 99.

    while step <= iteration_lvl1:
        lossall = []
        model.train()
        for batch, (moving, fixed) in enumerate(train_loader):
            X = moving[0].to(device).float()
            Y = fixed[0].to(device).float()

            # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y)

            loss_multiNCC, loss_Jacobian, loss_regulation = get_loss(grid_class, loss_similarity, loss_Jdet,
                                                                     loss_smooth, F_X_Y,
                                                                     X_Y, Y_4x)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall.append([loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])

            sys.stdout.write(
                "\r" + 'lv1:step:batch "{0}:{1}" -> training loss "{2:.4f}" - sim_NCC "{3:4f}" - Jdet "{4:.10f}" -smo "{5:.4f}"'.format(
                    step, batch, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()

            logging.info("img_name:{}".format(moving[1][0]))
            logging.info("modelv1, iter: %d batch: %d  loss: %.4f  sim: %.4f  grad: %.4f" % (
                step, batch, loss.item(), loss_multiNCC.item(), loss_regulation.item()))

        # validation
        val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss = validation_ccregnet(args, model, loss_similarity,
                                                                                       grid_class, 4)


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
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        if stop_criterion.stop():
            break

        step += 1
        if step > iteration_lvl1:
            break


def train_lvl2():
    print("Training lvl2...")
    device = args.device

    model_lvl1 = CCRegNet_lv1(2, 3, start_channel, is_train=True, range_flow=range_flow,
                              grid=grid_class).to(device)

    # model_path = r'D:\xxf\4DCT-R\lapirn\Model\Stage\2023-02-21-09-56-51_NCC_reg_disp_stagelvl1_070_-0.4229.pth'
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

    model = CCRegNet_lv2(2, 3, start_channel, is_train=True, range_flow=range_flow,
                         model_lvl1=model_lvl1, grid=grid_class).to(device)

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
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

            # compose_field_e0_lvl1, warpped_inputx_lvl1_out, lv2_out, down_y, output_disp_e0_v, lvl1_v, e0
            F_X_Y, _, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y)

            loss_multiNCC, loss_Jacobian, loss_regulation = get_loss(grid_class, loss_similarity, loss_Jdet,
                                                                     loss_smooth, F_X_Y,
                                                                     X_Y, Y_4x)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            # lossall[:, step] = np.array(
            #     [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            lossall.append([loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'lv2:step:batch "{0}:{1}" -> training loss "{2:.4f}" - sim_NCC "{3:4f}" - Jdet "{4:.10f}" -smo "{5:.4f}"'.format(
                    step, batch, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()

            logging.info("img_name:{}".format(moving[1][0]))
            logging.info("modelv2, iter: %d batch: %d  loss: %.4f  sim: %.4f  grad: %.4f" % (
                step, batch, loss.item(), loss_multiNCC.item(), loss_regulation.item()))

            # if batch == 0:
            #     m_name = str(step) + 'warped_' + moving[1][0]
            #     save_image(X_Y, Y, args.output_dir, m_name)
            #     m_name = str(step) + 'fixed_' + moving[1][0]
            #     save_image(Y_4x, Y, args.output_dir, m_name)

        # validation
        val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss = validation_ccregnet(args, model, loss_similarity,
                                                                                       grid_class, 2)

        mean_loss = np.mean(np.array(lossall), 0)[0]
        print(
            "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                mean_loss, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        stop_criterion.add(val_ncc_loss, val_jac_loss, val_total_loss, train_loss=mean_loss)

        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            modelname = model_dir + '/' + model_name + "stagelvl2" + '_{:03d}_'.format(step) + '{:.4f}.pth'.format(
                best_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list, stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        if stop_criterion.stop():
            break

        if step == freeze_step:
            model.unfreeze_modellvl1()

        step += 1
        if step > iteration_lvl2:
            break


def train_lvl3():
    print("Training lvl3...")
    device = args.device

    model_lvl1 = CCRegNet_lv1(2, 3, start_channel, is_train=True,
                              range_flow=range_flow, grid=grid_class).to(device)
    model_lvl2 = CCRegNet_lv2(2, 3, start_channel, is_train=True,
                              range_flow=range_flow, model_lvl1=model_lvl1,
                              grid=grid_class).to(device)

    # model_path = '/home/cqut/project/xxf/4DCT-R/lapirn/Model/Stage/2023-02-17-21-50-40_NCC_reg_disp_stagelvl2_073_-0.8829.pth'
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

    model = CCRegNet_lv3(2, 3, start_channel, is_train=True,
                         range_flow=range_flow, model_lvl2=model_lvl2,
                         grid=grid_class).to(device)

    loss_similarity = multi_resolution_NCC(win=7, scale=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
    # load_model = False
    # if load_model is True:
    #     model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
    #     print("Loading weight: ", model_path)
    #     step = 3000
    #     model.load_state_dict(torch.load(model_path)['model'])
    #     temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
    #     lossall[:, 0:3000] = temp_lossall[:, 0:3000]
    best_loss = 99.

    while step <= iteration_lvl3:
        lossall = []
        model.train()
        for batch, (moving, fixed) in enumerate(train_loader):
            X = moving[0].to(device).float()
            Y = fixed[0].to(device).float()

            # compose_field_e0_lvl1, warpped_inputx_lvl1_out,warpped_inputx_lvl2_out,warpped_inputx_lvl3_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
            pred = model(X, Y)

            loss_multiNCC, loss_Jacobian, loss_regulation = get_loss(grid_class, loss_similarity, loss_Jdet,
                                                                     loss_smooth, F_X_Y,
                                                                     X_Y, Y_4x)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            # lossall[:, step] = np.array(
            #     [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            lossall.append([loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])

            sys.stdout.write(
                "\r" + 'lv3:step:batch "{0}:{1}" -> training loss "{2:.4f}" - sim_NCC "{3:4f}" - Jdet "{4:.10f}" -smo "{5:.4f}"'.format(
                    step, batch, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()

            logging.info("img_name:{}".format(moving[1][0]))
            logging.info("modelv3, iter: %d batch: %d  loss: %.4f  sim: %.4f  grad: %.4f" % (
                step, batch, loss.item(), loss_multiNCC.item(), loss_regulation.item()))

            # if batch == 0:
            #     m_name = 'l3_' + str(step) + 'moving_' + moving[1][0]
            #     save_image(X_Y, Y, args.output_dir, m_name)
            #     m_name = 'l3_' + str(step) + 'fixed_' + moving[1][0]
            #     save_image(Y_4x, Y, args.output_dir, m_name)

        # validation
        val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss = validation_ccregnet(args, model, loss_similarity,
                                                                                       grid_class, 1)

        mean_loss = np.mean(np.array(lossall), 0)[0]
        print(
            "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                mean_loss, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        stop_criterion.add(val_ncc_loss, val_jac_loss, val_total_loss, train_loss=mean_loss)

        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            # modelname = model_dir + '/' + model_name + "{:.4f}_stagelvl3_".format(best_loss) + str(step) + '.pth'
            modelname = model_dir + '/' + model_name + "stagelvl3" + '_{:03d}_'.format(step) + '{:.4f}best.pth'.format(
                val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list, stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)
        else:
            modelname = model_dir + '/' + model_name + "stagelvl3" + '_{:03d}_'.format(step) + '{:.4f}.pth'.format(
                val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list, stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        if stop_criterion.stop():
            break

        if step == freeze_step:
            model.unfreeze_modellvl2()

        step += 1
        if step > iteration_lvl3:
            break


if __name__ == "__main__":
    args = get_args()

    lr = args.lr
    start_channel = args.initial_channels
    antifold = args.antifold
    # n_checkpoint = args.n_save_iter
    smooth = args.smooth
    # datapath = opt.datapath
    freeze_step = args.freeze_step

    iteration_lvl1 = args.iteration_lvl1
    iteration_lvl2 = args.iteration_lvl2
    iteration_lvl3 = args.iteration_lvl3

    fixed_folder = os.path.join(args.train_dir, 'fixed')
    moving_folder = os.path.join(args.train_dir, 'moving')
    f_img_file_list = sorted([os.path.join(fixed_folder, file_name) for file_name in os.listdir(fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_img_file_list = sorted([os.path.join(moving_folder, file_name) for file_name in os.listdir(moving_folder) if
                              file_name.lower().endswith('.gz')])

    make_dirs()

    log_index = len([file for file in os.listdir(args.log_dir) if file.endswith('.txt')])

    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = "{}_lapirn_corr_att_".format(train_time)

    logging.basicConfig(level=logging.INFO,
                        filename=f'Log/log{log_index}.txt',
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    # size = [144,192,160] # z y x
    # imgshape = (size[0], size[1], size[2])
    # imgshape_4 = (size[0] / 4,  size[1] / 4, size[2] / 4)
    # imgshape_2 = (size[0] / 2,  size[1] / 2, size[2] / 2)

    grid_class = Grid()
    range_flow = 0.4
    train_lvl1()
    train_lvl2()
    train_lvl3()
