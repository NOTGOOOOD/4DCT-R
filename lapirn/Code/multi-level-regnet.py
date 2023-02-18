import os
import sys
import numpy as np
import torch
import torch.utils.data as Data
import logging
import time

from Functions import generate_grid, generate_grid_unit, transform_unit_flow_to_flow_cuda, transform_unit_flow_to_flow
from unet import UNet_lv1, UNet_lv2, UNet_lv3, SpatialTransform_unit
from miccai2020_model_stage import smoothloss, neg_Jdet_loss, multi_resolution_NCC
from utils.datagenerators import Dataset, TestDataset
from utils.config import get_args
from utils.losses import NCC
from utils.scheduler import StopCriterion
from utils.utilize import set_seed, load_landmarks, save_image
from utils.metric import MSE, landmark_loss


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


def validation(args, model, imgshape, loss_similarity, step):
    fixed_folder = os.path.join(args.val_dir, 'fixed')
    moving_folder = os.path.join(args.val_dir, 'moving')
    f_img_file_list = sorted([os.path.join(fixed_folder, file_name) for file_name in os.listdir(fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_img_file_list = sorted([os.path.join(moving_folder, file_name) for file_name in os.listdir(moving_folder) if
                              file_name.lower().endswith('.gz')])

    val_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
    val_loader = Data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    transform = SpatialTransform_unit().cuda()
    transform.eval()

    grid = generate_grid_unit([args.size] * 3)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    scale_factor = args.size / imgshape[0]
    upsample = torch.nn.Upsample(scale_factor=scale_factor, mode="trilinear")
    with torch.no_grad():
        model.eval()  # m_name = "{}_affine.nii.gz".format(moving[1][0][:13])
        losses = []
        for batch, (moving, fixed) in enumerate(val_loader):
            input_moving = moving[0].to('cuda').float()
            input_fixed = fixed[0].to('cuda').float()
            pred = model(input_moving, input_fixed)

            disp_up = pred[0]
            if scale_factor > 1:
                disp_up = upsample(pred[0])

            X_Y_up = transform(input_moving, disp_up.permute(0, 2, 3, 4, 1), grid)

            # F_X_Y_cpu = F_X_Y.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
            # F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

            mse_loss = MSE(X_Y_up, input_fixed)
            # ncc_loss = loss_similarity(X_Y, Y_4x)
            ncc_loss_ori = loss_similarity(X_Y_up, input_fixed)
            losses.append([ncc_loss_ori.item(), mse_loss.item()])

            # save_flow(F_X_Y_cpu, args.output_dir + '/warpped_flow.nii.gz')
            # save_img(X_Y, args.output_dir + '/warpped_moving.nii.gz')
            # m_name = "{}_warped.nii.gz".format(moving[1][0].split('.nii')[0])
            # save_img(X_Y, args.output_dir + '/' + file_name + '_warpped_moving.nii.gz')
            # save_image(X_Y, input_fixed, args.output_dir, m_name)
            # if batch == 0:
            #     m_name = '{0}_{1}.nii.gz'.format(imgshape[0], step)
            #     save_image(pred[1], input_fixed, args.output_dir, m_name)
            #     m_name = '{0}_{1}_up.nii.gz'.format(imgshape[0], step)
            #     save_image(X_Y_up, input_fixed, args.output_dir, m_name)

        mean_loss = np.mean(losses, 0)
        return mean_loss[0], mean_loss[1]


def train_lvl1():
    print("Training lvl1...")
    device = args.device

    model = UNet_lv1(2, 3, depth=depth_lv1, initial_channels=start_channel, is_train=True, imgshape=imgshape_4).to(
        device)

    loss_similarity = NCC(win=3)
    loss_Jdet = neg_Jdet_loss
    loss_smooth = smoothloss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    # names = sorted(glob.glob(datapath + '/*.nii'))

    grid_4 = generate_grid(imgshape_4)
    grid_4 = torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).to(device).float()

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
    best_loss = 99.

    while step <= iteration_lvl1:
        lossall = []
        model.train()
        for batch, (moving, fixed) in enumerate(train_loader):
            X = moving[0].to(device).float()
            Y = fixed[0].to(device).float()

            # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
            F_X_Y, X_Y, Y_4x, F_xy = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

            # reg2 - use velocity
            _, _, z, y, x = F_X_Y.shape
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (z - 1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_X_Y)

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
        val_ncc_loss, val_mse_loss = validation(args, model, imgshape_4, loss_similarity, step)

        # with lr 1e-3 + with bias
        if val_ncc_loss <= best_loss:
            best_loss = val_ncc_loss
            modelname = model_dir + '/' + model_name + "stagelvl1" + '_{:03d}_'.format(step) + '{:.4f}.pth'.format(
                best_loss)
            logging.info("save model:{}".format(modelname))
            torch.save(model.state_dict(), modelname)

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

    model_lvl1 = UNet_lv1(2, 3, depth=depth_lv1, initial_channels=start_channel, is_train=True, imgshape=imgshape_4).to(
        device)

    # model_path = "../Model/Stage/LDR_LPBA_NCC_1_1_stagelvl1_1500.pth"
    model_list = []
    for f in os.listdir('../Model/Stage'):
        if model_name + "stagelvl1" in f:
            model_list.append(os.path.join('../Model/Stage', f))

    model_path = sorted(model_list)[-1]
    model_lvl1.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl1...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = UNet_lv2(2, 3, depth=depth_lv2, initial_channels=start_channel, is_train=True, imgshape=imgshape_2,
                     model_lvl1=model_lvl1).to(device)

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    # names = sorted(glob.glob(datapath + '/*.nii'))

    grid_2 = generate_grid(imgshape_2)
    grid_2 = torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).to(device).float()

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

            # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1 = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            # reg2 - use velocity
            _, _, z, y, x = F_X_Y.shape
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (z - 1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_X_Y)

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
        val_ncc_loss, val_mse_loss = validation(args, model, imgshape_2, loss_similarity, step)

        # with lr 1e-3 + with bias
        if val_ncc_loss <= best_loss:
            best_loss = val_ncc_loss
            modelname = model_dir + '/' + model_name + "stagelvl2" + '_{:03d}_'.format(step) + '{:.4f}.pth'.format(
                best_loss)
            logging.info("save model:{}".format(modelname))
            torch.save(model.state_dict(), modelname)

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

    model_lvl1 = UNet_lv1(2, 3, depth=depth_lv1, initial_channels=start_channel, is_train=True, imgshape=imgshape_4).to(
        device)
    model_lvl2 = UNet_lv2(2, 3, depth=depth_lv2, initial_channels=start_channel, is_train=True, imgshape=imgshape_2,
                          model_lvl1=model_lvl1).to(device)

    model_list = []
    for f in os.listdir('../Model/Stage'):
        if model_name + "stagelvl2" in f:
            model_list.append(os.path.join('../Model/Stage', f))

    model_path = sorted(model_list)[-1]
    model_lvl2.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl2...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = UNet_lv3(2, 3, depth=depth_lv3, initial_channels=start_channel, is_train=True, imgshape=imgshape,
                     model_lvl2=model_lvl2).to(device)

    loss_similarity = multi_resolution_NCC(win=7, scale=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)
    # transform_nearest = SpatialTransformNearest_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    # names = sorted(glob.glob(datapath + '/*.nii'))

    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # lossall = np.zeros((4, iteration_lvl3 + 1))

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
    #     model.load_state_dict(torch.load(model_path))
    #     temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
    #     lossall[:, 0:3000] = temp_lossall[:, 0:3000]
    best_loss = 99.

    while step <= iteration_lvl3:
        lossall = []
        model.train()
        for batch, (moving, fixed) in enumerate(train_loader):
            X = moving[0].to(device).float()
            Y = fixed[0].to(device).float()

            # compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2 = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

            # reg2 - use velocity
            _, _, z, y, x = F_X_Y.shape
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (z - 1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y - 1)
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_X_Y)

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
        val_ncc_loss, val_mse_loss = validation(args, model, imgshape, loss_similarity, step)

        # with lr 1e-3 + with bias
        if val_ncc_loss <= best_loss:
            best_loss = val_ncc_loss
            # modelname = model_dir + '/' + model_name + "{:.4f}_stagelvl3_".format(best_loss) + str(step) + '.pth'
            modelname = model_dir + '/' + model_name + "stagelvl3" + '_{:03d}_'.format(step) + '{:.4f}.pth'.format(
                best_loss)
            logging.info("save model:{}".format(modelname))
            torch.save(model.state_dict(), modelname)

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
    set_seed(1024)

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

    depth_lv1 = 2
    depth_lv2 = 2
    depth_lv3 = 3

    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = "{}_unet_reg_disp_".format(train_time)

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
