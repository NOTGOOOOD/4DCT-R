import os
import sys
import numpy as np
import torch
import logging
import time
import torch.nn.functional as F

from utils.Functions import get_loss, Grid, transform_unit_flow_to_flow_cuda, AdaptiveSpatialTransformer, test_dirlab
from utils.datagenerators import build_dataloader
from utils.config import get_args
from utils.losses import NCC, multi_resolution_NCC, neg_Jdet_loss, gradient_loss as smoothloss
from utils.scheduler import StopCriterion
from utils.utilize import set_seed, save_model, count_parameters, make_dirs
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
    print("Training CCE_single...")
    device = args.device

    model = Model_lv1(2, 3, start_channel, is_train=True,
                      range_flow=range_flow, grid=grid_class).to(device)
    print(count_parameters(model))

    loss_similarity = NCC(win=3)
    loss_Jdet = neg_Jdet_loss
    loss_smooth = smoothloss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = './ckpt/stage'

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    stop_criterion = StopCriterion()
    step = 0
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

if __name__ == "__main__":
    set_seed(1024)
    args = get_args()
    make_dirs(args)

    log_index = len([file for file in os.listdir(args.log_dir) if file.endswith('.txt')])

    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = "{}_CCENet_single_".format(train_time)

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
    grid_class = Grid()
    range_flow = 0.4
    train_loader = build_dataloader(args, 'train')
    val_loader = build_dataloader(args, 'val')
    test_loader_dirlab = build_dataloader(args, 'test')

    train_lvl1()
