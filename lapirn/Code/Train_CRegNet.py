import os
import sys
import numpy as np
import torch
import torch.utils.data as Data
import logging
import time
import torch.nn.functional as F

from utils.utilize import set_seed, save_model, load_landmarks, count_parameters

set_seed(1024)

# from CRegNet import CRegNet_lv0, CRegNet_lv1, \
#     CRegNet_lv2, CRegNet_lv3
from LapIRN import Miccai2020_LDR_laplacian_unit_disp_add_lvl0 as CRegNet_lv0, Miccai2020_LDR_laplacian_unit_disp_add_lvl1 as CRegNet_lv1,\
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2 as CRegNet_lv2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3 as CRegNet_lv3

from utils.datagenerators import Dataset, DirLabDataset, build_dataloader_dirlab
from utils.config import get_args
from utils.losses import NCC, smoothloss, multi_resolution_NCC, neg_Jdet_loss
from utils.scheduler import StopCriterion
from utils.Functions import Grid, get_loss, AdaptiveSpatialTransformer, transform_unit_flow_to_flow_cuda, test_dirlab
from utils.metric import MSE


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def validation_ccregnet(args, model, loss_similarity, grid_class, scale_factor):
    transform = AdaptiveSpatialTransformer()

    # upsample = torch.nn.Upsample(scale_factor=scale_factor, mode="trilinear")
    with torch.no_grad():
        model.eval()  # m_name = "{}_affine.nii.gz".format(moving[1][0][:13])
        losses = []
        for batch, (moving, fixed) in enumerate(val_loader):
            input_moving = moving[0].to('cuda').float()
            input_fixed = fixed[0].to('cuda').float()
            pred = model(input_moving, input_fixed)
            F_X_Y = pred['flow']

            if scale_factor > 1:
                F_X_Y = F.interpolate(F_X_Y, input_moving.shape[2:], mode='trilinear',
                                      align_corners=True, recompute_scale_factor=False)

            X_Y_up = transform(input_moving, F_X_Y.permute(0, 2, 3, 4, 1),
                               grid_class.get_grid(input_moving.shape[2:], True))
            mse_loss = MSE(X_Y_up, input_fixed)
            ncc_loss_ori = loss_similarity(X_Y_up, input_fixed)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = neg_Jdet_loss(F_X_Y_norm, grid_class.get_grid(input_moving.shape[2:]))
            # loss_Jacobian = jacobian_determinant(F_X_Y[0].cpu().detach().numpy())

            # reg2 - use velocity
            _, _, z, y, x = F_X_Y.shape
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (z - 1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (x - 1)
            loss_regulation = smoothloss(F_X_Y)
            # loss_regulation = bending_energy_loss(F_X_Y)
            loss_sum = ncc_loss_ori + args.antifold * loss_Jacobian + args.smooth * loss_regulation

            losses.append([ncc_loss_ori.item(), mse_loss.item(), loss_Jacobian.item(), loss_sum.item()])

        mean_loss = np.mean(losses, 0)
        return mean_loss[0], mean_loss[1], mean_loss[2], mean_loss[3]


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

    # mode_lvl0 = CRegNet_lv0(2, 3, start_channel, is_train=True, range_flow=range_flow, grid=grid_class).to(device)
    # model_list = []
    # for f in os.listdir('../Model/Stage'):
    #     if model_name + "stagelvl0" in f:
    #         model_list.append(os.path.join('../Model/Stage', f))
    # model_path = sorted(model_list)[-1]
    #
    # mode_lvl0.load_state_dict(torch.load(model_path)['model'])
    # print("Loading weight for model_lvl0...", model_path)
    #
    # # Freeze model_lvl1 weight
    # for param in mode_lvl0.parameters():
    #     param.requires_grad = False

    model = CRegNet_lv1(2, 3, start_channel, is_train=True,
                        range_flow=range_flow, grid=grid_class, model_lvl0=None).to(device)
    print(count_parameters(model))
    loss_similarity = multi_resolution_NCC(win=5, scale=1)
    loss_Jdet = neg_Jdet_loss
    loss_smooth = smoothloss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir = '../Model/Stage'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

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
            pred = model(X, Y)
            F_X_Y, X_Y, Y_4x = pred['flow'], pred['warped_img'], pred['down_y']

            loss_multiNCC, loss_Jacobian, loss_regulation = get_loss(grid_class, loss_similarity, loss_Jdet,
                                                                     loss_smooth, F_X_Y,
                                                                     X_Y, Y_4x)

            # loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation
            loss = loss_multiNCC
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
        val_total_loss = val_ncc_loss
        mean_loss = np.mean(np.array(lossall), 0)[0]
        print(
            "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                mean_loss, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        stop_criterion.add(val_ncc_loss, val_jac_loss, val_total_loss, train_loss=mean_loss)

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
    model_lvl0 = CRegNet_lv0(2, 3, start_channel, is_train=True, range_flow=range_flow,
                             grid=grid_class,).to(device)
    model_lvl1 = CRegNet_lv1(2, 3, start_channel, is_train=True, range_flow=range_flow,
                             grid=grid_class, model_lvl0=None).to(device)

    # model_path = r'D:\xxf\4DCT-R\lapirn\Model\Stage\2023-02-19-17-18-31_NCC_reg_disp_stagelvl1_057_-0.4263.pth'
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

    model = CRegNet_lv2(2, 3, start_channel, is_train=True, range_flow=range_flow,
                        model_lvl1=model_lvl1, grid=grid_class).to(device)
    print(count_parameters(model) - count_parameters(model_lvl1))
    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    stop_criterion = StopCriterion()
    model_dir = '../Model/Stage'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # train_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    # train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    step = 0
    best_loss = 99.
    while step <= iteration_lvl2:
        lossall = []
        model.train()
        for batch, (moving, fixed) in enumerate(train_loader):
            X = moving[0].to(device).float()
            Y = fixed[0].to(device).float()

            # compose_field_e0_lvl1, warpped_inputx_lvl1_out, lv2_out, down_y, output_disp_e0_v, lvl1_v, e0
            pred = model(X, Y)
            F_X_Y, X_Y, Y_4x = pred['flow'], pred['warped_img'], pred['down_y']

            loss_multiNCC, loss_Jacobian, loss_regulation = get_loss(grid_class, loss_similarity, loss_Jdet,
                                                                     loss_smooth, F_X_Y,
                                                                     X_Y, Y_4x)

            # loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation
            loss = loss_multiNCC
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

        val_total_loss = val_ncc_loss
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
            logging.info("save model:{}".format(modelname))
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
    device = args.device
    model_lvl0 = CRegNet_lv0(2, 3, start_channel, is_train=True, range_flow=range_flow,
                             grid=grid_class, ).to(device)
    model_lvl1 = CRegNet_lv1(2, 3, start_channel, is_train=True,
                             range_flow=range_flow, grid=grid_class,model_lvl0=None).to(device)
    model_lvl2 = CRegNet_lv2(2, 3, start_channel, is_train=True,
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

    model = CRegNet_lv3(2, 3, start_channel, is_train=True,
                        range_flow=range_flow, model_lvl2=model_lvl2, grid=grid_class).to(device)


    print(count_parameters(model) - count_parameters(model_lvl2))
    loss_similarity = multi_resolution_NCC(win=9, scale=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # train_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    # train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

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
            F_X_Y, X_Y = pred['flow'], pred['warped_img']

            loss_multiNCC, loss_Jacobian, loss_regulation = get_loss(grid_class, loss_similarity, loss_Jdet,
                                                                     loss_smooth, F_X_Y,
                                                                     X_Y, Y)

            # loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation
            loss = loss_multiNCC
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
        val_total_loss = val_ncc_loss
        mean_loss = np.mean(np.array(lossall), 0)[0]
        print(
            "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                mean_loss, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        stop_criterion.add(val_ncc_loss, val_jac_loss, val_total_loss, train_loss=mean_loss)

        # save model
        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            # modelname = model_dir + '/' + model_name + "{:.4f}_stagelvl3_".format(best_loss) + str(step) + '.pth'
            modelname = model_dir + '/' + model_name + "stagelvl3" + '_{:03d}_'.format(step) + '{:.4f}best.pth'.format(
                val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)
        else:
            modelname = model_dir + '/' + model_name + "stagelvl3" + '_{:03d}_'.format(step) + '{:.4f}.pth'.format(
                val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        test_dirlab(args, model, test_loader_dirlab, norm=True)

        if stop_criterion.stop():
            break

        if step == freeze_step:
            model.unfreeze_modellvl2()

        step += 1
        if step > iteration_lvl3:
            break


def train_lvl0():
    print("Training lvl0...")
    device = args.device

    model = CRegNet_lv0(2, 3, start_channel, is_train=True, range_flow=range_flow, grid=grid_class).to(device)
    loss_similarity = multi_resolution_NCC(win=5, scale=1)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir = '../Model/Stage'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    stop_criterion = StopCriterion()
    step = 0
    best_loss = 99.

    while step <= iteration_lvl3:
        lossall = []
        model.train()
        for batch, (moving, fixed) in enumerate(train_loader):
            X = moving[0].to(device).float()
            Y = fixed[0].to(device).float()

            # compose_field_e0_lvl1, warpped_inputx_lvl1_out,warpped_inputx_lvl2_out,warpped_inputx_lvl3_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
            pred = model(X, Y)
            F_X_Y, X_Y, Y_4x = pred['flow'], pred['warped_img'], pred['down_y']

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
                "\r" + 'lv0:step:batch "{0}:{1}" -> training loss "{2:.4f}" - sim_NCC "{3:4f}" - Jdet "{4:.10f}" -smo "{5:.4f}"'.format(
                    step, batch, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()

            logging.info("img_name:{}".format(moving[1][0]))
            logging.info("modelv0, iter: %d batch: %d  loss: %.4f  sim: %.4f  grad: %.4f" % (
                step, batch, loss.item(), loss_multiNCC.item(), loss_regulation.item()))

            # if batch == 0:
            #     m_name = 'l3_' + str(step) + 'moving_' + moving[1][0]
            #     save_image(X_Y, Y, args.output_dir, m_name)
            #     m_name = 'l3_' + str(step) + 'fixed_' + moving[1][0]
            #     save_image(Y_4x, Y, args.output_dir, m_name)

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
            # modelname = model_dir + '/' + model_name + "{:.4f}_stagelvl3_".format(best_loss) + str(step) + '.pth'
            modelname = model_dir + '/' + model_name + "stagelvl0" + '_{:03d}_'.format(step) + '{:.4f}best.pth'.format(
                val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)
        else:
            modelname = model_dir + '/' + model_name + "stagelvl0" + '_{:03d}_'.format(step) + '{:.4f}.pth'.format(
                val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        if stop_criterion.stop():
            break

        step += 1
        if step > iteration_lvl3:
            break


if __name__ == "__main__":
    args = get_args()

    lr = args.lr
    start_channel = args.initial_channels
    antifold = args.antifold
    smooth = args.smooth
    freeze_step = args.freeze_step

    iteration_lvl1 = args.iteration_lvl1
    iteration_lvl2 = args.iteration_lvl2
    iteration_lvl3 = args.iteration_lvl3

    train_loader = build_dataloader_dirlab(args, mode='train')
    val_loader = build_dataloader_dirlab(args, mode='val')
    test_loader_dirlab = build_dataloader_dirlab(args, mode='test')

    make_dirs()
    log_index = len([file for file in os.listdir(args.log_dir) if file.endswith('.txt')])

    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = "{}_lap_noreg_".format(train_time)

    logging.basicConfig(level=logging.INFO,
                        filename=f'Log/log{log_index}.txt',
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    grid_class = Grid()
    range_flow = 0.4

    # train_lvl0()
    train_lvl1()
    train_lvl2()
    train_lvl3()
