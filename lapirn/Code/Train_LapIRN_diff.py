import os
import sys
import numpy as np
import torch
import torch.utils.data as Data
import logging
import time

from utils.Functions import generate_grid, transform_unit_flow_to_flow_cuda
from LapIRN import Miccai2020_LDR_laplacian_unit_add_lvl1, Miccai2020_LDR_laplacian_unit_add_lvl2, \
    Miccai2020_LDR_laplacian_unit_add_lvl3, SpatialTransform_unit, smoothloss, \
    neg_Jdet_loss, multi_resolution_NCC
from utils.datagenerators import Dataset
from utils.config import get_args
from utils.losses import NCC
from utils.utilize import save_image,save_model
from utils.scheduler import StopCriterion
from Test_CRegNet import validation


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
#                     dest="antifold", default=0.,
#                     help="Anti-fold loss: suggested range 0 to 1000")
# parser.add_argument("--smooth", type=float,
#                     dest="smooth", default=3.5,
#                     help="Gradient smooth loss: suggested range 0.1 to 10")
# parser.add_argument("--checkpoint", type=int,
#                     dest="checkpoint", default=5000,
#                     help="frequency of saving models")
# parser.add_argument("--start_channel", type=int,
#                     dest="start_channel", default=7,
#                     help="number of start channels")
# parser.add_argument("--datapath", type=str,
#                     dest="datapath",
#                     default='/PATH/TO/YOUR/DATA',
#                     help="data path for training images")
# parser.add_argument("--freeze_step", type=int,
#                     dest="freeze_step", default=2000,
#                     help="Number step for freezing the previous level")
# opt = parser.parse_args()

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

    model = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                   range_flow=range_flow).to(device)

    loss_similarity = NCC(win=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # # OASIS
    # names = sorted(glob.glob(datapath + '/*.nii'))

    grid_4 = generate_grid(imgshape_4)
    grid_4 = torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model/Stage'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # lossall = np.zeros((4, iteration_lvl1 + 1))
    # training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
    #                                      shuffle=True, num_workers=4)

    train_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    stop_criterion = StopCriterion()
    step = 0
    best_loss = 99.
    # load_model = False
    # if load_model is True:
    #     model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
    #     print("Loading weight: ", model_path)
    #     step = 3000
    #     model.load_state_dict(torch.load(model_path)['model'])
    #     temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
    #     lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl1:
        lossall = []
        model.train()
        for batch, (moving, fixed) in enumerate(train_loader):
            X = moving[0].to(device).float()
            Y = fixed[0].to(device).float()

            # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

            # reg2 - use velocity
            _, _, z, y, x = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (x - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (z - 1)
            loss_regulation = loss_smooth(F_xy)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation
            # loss = loss_multiNCC + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall.append([loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])

            sys.stdout.write(
                "\r" + 'step:batch "{0}:{1}" -> training loss "{2:.4f}" - sim_NCC "{3:4f}" - Jdet "{4:.10f}" -smo "{5:.4f}"'.format(
                    step, batch, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()

            logging.info("img_name:{}".format(moving[1][0]))
            logging.info("modelv1, iter: %d batch: %d  loss: %.4f  sim: %.4f  grad: %.4f" % (
                step, batch, loss.item(), loss_multiNCC.item(), loss_regulation.item()))

        # validation
        val_ncc_loss, val_mse_loss = validation(args, model, imgshape_4, loss_similarity, step)

        # save model
        if val_ncc_loss <= best_loss:
            best_loss = val_ncc_loss
            modelname = model_dir + '/' + model_name + "stagelvl1" + '_{:03d}_'.format(step) + '{:.4f}.pth'.format(
                best_loss)
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list, stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        mean_loss = np.mean(np.array(lossall), 0)[0]
        print("\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f" % (
            mean_loss, val_ncc_loss, val_mse_loss))

        stop_criterion.add(val_ncc_loss, val_mse_loss)
        if stop_criterion.stop():
            break

        step += 1
        if step > iteration_lvl1:
            break


def train_lvl2():
    print("Training lvl2...")
    device = args.device

    model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                        range_flow=range_flow).to(device)

    # model_path = "../Model/Stage/LDR_LPBA_NCC_1_1_stagelvl1_1500.pth"
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

    model = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                                   range_flow=range_flow, model_lvl1=model_lvl1).to(device)

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # # OASIS
    # names = sorted(glob.glob(datapath + '/*.nii'))

    grid_2 = generate_grid(imgshape_2)
    grid_2 = torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir = '../Model/Stage'
    stop_criterion = StopCriterion()

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    train_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    step = 0
    best_loss = 99.
    # if load_model is True:
    #     model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
    #     print("Loading weight: ", model_path)
    #     step = 3000
    #     model.load_state_dict(torch.load(model_path)['model'])
    #     temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
    #     lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl2:
        lossall = []
        model.train()
        for batch, (moving, fixed) in enumerate(train_loader):
            X = moving[0].to(device).float()
            Y = fixed[0].to(device).float()

            # output_disp_e0, warpped_inputx_lvl1_out, y_down, compose_field_e0_lvl1v, lvl1_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            # reg2 - use velocity
            _, _, z, y, x = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (x - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (z - 1)
            loss_regulation = loss_smooth(F_xy)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall.append([loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step:batch "{0}:{1}" -> training loss "{2:.4f}" - sim_NCC "{3:4f}" - Jdet "{4:.10f}" -smo "{5:.4f}"'.format(
                    step, batch, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()

            logging.info("img_name:{}".format(moving[1][0]))
            logging.info("modelv2, iter: %d batch: %d  loss: %.4f  sim: %.4f  grad: %.4f" % (
                step, batch, loss.item(), loss_multiNCC.item(), loss_regulation.item()))

            if batch == 0:
                m_name = str(step) + 'warped_' + moving[1][0]
                save_image(X_Y, Y, args.output_dir, m_name)
                m_name = str(step) + 'fixed_' + moving[1][0]
                save_image(Y_4x, Y, args.output_dir, m_name)

        # validation
        val_ncc_loss, val_mse_loss = validation(args, model, imgshape_2, loss_similarity, step)

        # save model
        if val_ncc_loss <= best_loss:
            best_loss = val_ncc_loss
            modelname = model_dir + '/' + model_name + "stagelvl2" + '_{:03d}_'.format(
                step) + '{:.4f}.pth'.format(best_loss)
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list, stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        mean_loss = np.mean(np.array(lossall), 0)[0]
        print("\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f" % (
            mean_loss, val_ncc_loss, val_mse_loss))

        stop_criterion.add(val_ncc_loss, val_mse_loss)
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

    model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                        range_flow=range_flow).to(device)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                                        range_flow=range_flow, model_lvl1=model_lvl1).to(device)

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

    model = Miccai2020_LDR_laplacian_unit_add_lvl3(2, 3, start_channel, is_train=True, imgshape=imgshape,
                                                   range_flow=range_flow, model_lvl2=model_lvl2).to(device)

    loss_similarity = multi_resolution_NCC(win=7, scale=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)
    # transform_nearest = SpatialTransformNearest_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # # OASIS
    # names = sorted(glob.glob(datapath + '/*.nii'))

    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # lossall = np.zeros((4, iteration_lvl3 + 1))
    #
    # training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
    #                                      shuffle=True, num_workers=2)
    train_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    stop_criterion = StopCriterion()
    step = 0
    # load_model = False
    # if load_model is True:
    #     model_path = "../Model/LDR_OASIS_NCC_unit_add_reg_3_anti_1_stagelvl3_10000.pth"
    #     print("Loading weight: ", model_path)
    #     step = 10000
    #     model.load_state_dict(torch.load(model_path)['model'])
    #     temp_lossall = np.load("../Model/lossLDR_OASIS_NCC_unit_add_reg_3_anti_1_stagelvl3_10000.npy")
    #     lossall[:, 0:10000] = temp_lossall[:, 0:10000]
    best_loss = 99.
    while step <= iteration_lvl3:
        lossall = []
        model.train()
        for batch, (moving, fixed) in enumerate(train_loader):
            X = moving[0].to(device).float()
            Y = fixed[0].to(device).float()

            # output_disp_e0, warpped_inputx_lvl1_out, y, compose_field_e0_lvl2_compose, lvl1_v, compose_lvl2_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

            # reg2 - use velocity
            _, _, z, y, x = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (x - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (z - 1)
            loss_regulation = loss_smooth(F_xy)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall.append([loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])

            sys.stdout.write(
                "\r" + 'step:batch "{0}:{1}" -> training loss "{2:.4f}" - sim_NCC "{3:4f}" - Jdet "{4:.10f}" -smo "{5:.4f}"'.format(
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
        val_ncc_loss, val_mse_loss = validation(args, model, imgshape, loss_similarity, step)

        # save model
        if val_ncc_loss <= best_loss:
            best_loss = val_ncc_loss
            # modelname = model_dir + '/' + model_name + "{:.4f}_stagelvl3_".format(best_loss) + str(step) + '.pth'
            modelname = model_dir + '/' + model_name + "stagelvl3" + '_{:03d}_'.format(
                step) + '{:.4f}.pth'.format(
                best_loss)
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list, stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        mean_loss = np.mean(np.array(lossall), 0)[0]
        print("\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f" % (
            mean_loss, val_ncc_loss, val_mse_loss))

        stop_criterion.add(val_ncc_loss, val_mse_loss)
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
    smooth = args.smooth
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
    model_name = "{}_NCC_reg_diff_".format(train_time)

    logging.basicConfig(level=logging.INFO,
                        filename=f'Log/log{log_index}.txt',
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    imgshape = (144, 144, 144)
    imgshape_4 = (144 / 4, 144 / 4, 144 / 4)
    imgshape_2 = (144 / 2, 144 / 2, 144 / 2)

    range_flow = 0.4
    train_lvl1()
    train_lvl2()
    train_lvl3()
