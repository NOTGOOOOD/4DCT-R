import os
import sys
import numpy as np
import torch
import logging
import time
import torch.nn.functional as F

from utils.Functions import get_loss, Grid, transform_unit_flow_to_flow_cuda, AdaptiveSpatialTransformer, test_dirlab
from CCENet_single import Model_lv1, \
    Model_lv2, Model_lv3
# from lapirn.Code.LapIRN import Miccai2020_LDR_laplacian_unit_disp_add_lvl1 as Model_lv1,\
#     Miccai2020_LDR_laplacian_unit_disp_add_lvl2 as Model_lv2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3 as Model_lv3
from utils.datagenerators import build_dataloader
from utils.config import get_args
from utils.losses import NCC, multi_resolution_NCC, neg_Jdet_loss, gradient_loss as smoothloss
from utils.scheduler import StopCriterion
from utils.utilize import set_seed, save_model, count_parameters
from utils.metric import MSE

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
            F_X_Y = pred['disp']

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

def train_lvl1():
    print("Training lvl1...")
    device = args.device

    model = Model_lv1(2, 3, start_channel, is_train=True,
                      range_flow=range_flow, grid=grid_class).to(device)
    print(count_parameters(model))

    loss_similarity = NCC(win=7)
    loss_Jdet = neg_Jdet_loss
    loss_smooth = smoothloss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = './ckpt/stage'

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

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
            # F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y)
            pred = model(X, Y)
            disp, wapred = pred['disp'], pred['warped_img']
            if Y.shape[2:] != disp.shape[2:]:
                Y = F.interpolate(Y, size=disp.shape[2:], mode='trilinear', align_corners=True)

            loss_multiNCC, loss_Jacobian, loss_regulation = get_loss(grid_class, loss_similarity, loss_Jdet,
                                                                     loss_smooth, disp,
                                                                     wapred, Y)

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
        # mean_tre = test_dirlab(args, model, test_loader_dirlab, norm=True, logging=logging)
        mean_loss = np.mean(np.array(lossall), 0)[0]
        stop_criterion.add(val_ncc_loss, val_jac_loss, val_total_loss, train_loss=mean_loss)

        # save model
        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            modelname = model_dir + '/' + model_name + "stagelvl1" + '_{:03d}_'.format(step) + '{:.4f}.pth'.format(
                best_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        # print(
        #     "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
        #         mean_loss, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        if stop_criterion.stop():
            break

        step += 1
        if step > iteration_lvl1:
            break

        break

def train_lvl2():
    print("Training lvl2...")
    device = args.device

    model_lvl1 = Model_lv1(2, 3, start_channel, is_train=True,
                           range_flow=range_flow, grid=grid_class).to(device)

    # model_path = r'D:\project\xxf\4DCT\lapirn\Model\Stage\2023-04-08-21-0-27_CCENet_planB_stagelvl1_243_-0.3347.pth'
    model_list = []
    for f in os.listdir('./ckpt/stage'):
        if model_name + "stagelvl1" in f:
            model_list.append(os.path.join('./ckpt/stage', f))

    model_path = sorted(model_list)[-1]

    model_lvl1.load_state_dict(torch.load(model_path)['model'])
    print("Loading weight for model_lvl1...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = Model_lv2(2, 3, start_channel, is_train=True,
                      range_flow=range_flow, model_lvl1=model_lvl1, grid=grid_class).to(device)

    print(count_parameters(model) - count_parameters(model_lvl1))

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    # loss_smooth = bending_energy_loss
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    stop_criterion = StopCriterion()
    model_dir = './ckpt/stage'

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

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
            disp, wapred, down_y = pred['disp'], pred['warped_img'], pred['down_y']

            loss_multiNCC, loss_Jacobian, loss_regulation = get_loss(grid_class, loss_similarity, loss_Jdet,
                                                                     loss_smooth, disp,
                                                                     wapred, down_y)

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
        stop_criterion.add(val_ncc_loss, val_jac_loss, val_total_loss, train_loss=mean_loss)

        # save model
        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            modelname = model_dir + '/' + model_name + "stagelvl2" + '_{:03d}_'.format(step) + '{:.4f}.pth'.format(
                best_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        print(
            "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                mean_loss, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        if stop_criterion.stop():
            break

        if step == freeze_step:
            model.unfreeze_modellvl1()

        step += 1
        if step > iteration_lvl2:
            break
        break

def train_lvl3():
    print("Training lvl3...")
    device = args.device

    model_lvl1 = Model_lv1(2, 3, start_channel, is_train=True,
                           range_flow=range_flow, grid=grid_class).to(device)
    model_lvl2 = Model_lv2(2, 3, start_channel, is_train=True,
                           range_flow=range_flow, model_lvl1=model_lvl1, grid=grid_class).to(device)

    # model_path = '/home/cqut/project/xxf/4DCT-R/lapirn/Model/Stage/2023-02-27-20-18-12_lapirn_corr_att_planB_stagelvl2_000_-0.7056.pth'
    # model_path = r'D:\project\xxf\4DCT\lapirn\Model\Stage\2023-03-27-20-40-00_lapirn_corr_att_planB_stagelvl2_000_-0.6411.pth'
    model_list = []
    for f in os.listdir('./ckpt/stage'):
        if model_name + "stagelvl2" in f:
            model_list.append(os.path.join('./ckpt/stage', f))

    model_path = sorted(model_list)[-1]
    model_lvl2.load_state_dict(torch.load(model_path)['model'])
    print("Loading weight for model_lvl2...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = Model_lv3(2, 3, start_channel, is_train=True,
                      range_flow=range_flow, model_lvl2=model_lvl2, grid=grid_class).to(device)
    print(count_parameters(model) - count_parameters(model_lvl2) -count_parameters(model_lvl1))
    loss_similarity = multi_resolution_NCC(win=7, scale=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = './ckpt/'

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

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
            F_X_Y, X_Y = pred['disp'], pred['warped_img']

            loss_multiNCC, loss_Jacobian, loss_regulation = get_loss(grid_class, loss_similarity, loss_Jdet,
                                                                     loss_smooth, F_X_Y,
                                                                     X_Y, Y)

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
        # val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss = validation_ccregnet(args, model, loss_similarity,
        #                                                                                grid_class, 1)
        mean_tre = test_dirlab(args, model, test_loader_dirlab, norm=True, logging=logging)
        mean_loss = np.mean(np.array(lossall), 0)[0]
        stop_criterion.add(mean_tre, mean_tre, mean_tre, train_loss=mean_loss)

        # save model
        if mean_tre <= best_loss:
            best_loss = mean_tre
            # modelname = model_dir + '/' + model_name + "{:.4f}_stagelvl3_".format(best_loss) + str(step) + '.pth'
            modelname = model_dir + '/' + model_name + "stagelvl3" + '_{:03d}_'.format(step) + '{:.4f}best.pth'.format(
                best_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)
        else:
            modelname = model_dir + '/' + model_name + "stagelvl3" + '_{:03d}_'.format(step) + '{:.4f}.pth'.format(
                best_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        # print(
            # "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
            #     mean_loss, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        if stop_criterion.stop():
            break

        if step == freeze_step:
            model.unfreeze_modellvl2()

        step += 1
        if step > iteration_lvl3:
            break

if __name__ == "__main__":
    set_seed(1024)
    args = get_args()
    make_dirs()

    log_index = len([file for file in os.listdir(args.log_dir) if file.endswith('.txt')])

    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = "{}_CCENet_planB_".format(train_time)

    logging.basicConfig(level=logging.INFO,
                        filename=f'Log/log{log_index}.txt',
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    lr = args.lr
    start_channel = args.initial_channels
    antifold = args.antifold
    smooth = args.smooth
    freeze_step = args.freeze_step

    iteration_lvl1 = args.iteration_lvl1
    iteration_lvl2 = args.iteration_lvl2
    iteration_lvl3 = args.iteration_lvl3

    grid_class = Grid()
    range_flow = 0.4
    train_loader = build_dataloader(args, 'train')
    val_loader = build_dataloader(args, 'val')
    test_loader_dirlab = build_dataloader(args, 'test')

    train_lvl1()
    train_lvl2()
    train_lvl3()
