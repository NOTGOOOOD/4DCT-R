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
from utils.datagenerators import Dataset, PatientDataset, DirLabDataset
from GDIR.model import regnet
from utils.scheduler import StopCriterion
from utils.utilize import set_seed, save_model, save_image, count_parameters, load_landmarks
from utils.metric import MSE, jacobian_determinant, SSIM, NCC as calc_NCC, landmark_loss


def make_dirs():
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


def test_patient(args, checkpoint, is_save=False):
    model.load_state_dict(torch.load(checkpoint)['model'])
    with torch.no_grad():
        model.eval()
        losses = []
        for batch, (moving, fixed, img_name) in enumerate(test_loader_patient):
            if batch == 6: break
            moving_img = moving.to(args.device).float()
            fixed_img = fixed.to(args.device).float()

            res = model(moving_img,fixed_img)  # b, c, d, h, w  disp, scale_disp, warped
            warped_image, flow = res['warped_img'], res['disp']

            ncc = calc_NCC(fixed_img.cpu().detach().numpy(), warped_image.cpu().detach().numpy())

            # loss_Jacobian = neg_Jdet_loss(y_pred[1].permute(0, 2, 3, 4, 1), grid)
            jac = jacobian_determinant(flow[0].cpu().detach().numpy())

            # MSE
            _mse = MSE(fixed_img, warped_image)
            # SSIM
            _ssim = SSIM(fixed_img.cpu().detach().numpy()[0, 0], warped_image.cpu().detach().numpy()[0, 0])

            losses.append([_mse.item(), jac, _ssim.item(), ncc.item()])
            print('case=%d after warped,MSE=%.5f Jac=%.8f, SSIM=%.5f, NCC=%.5f' % (
                batch + 1, _mse.item(), jac, _ssim.item(), ncc.item()))

            if is_save:
                # Save DVF
                # b,3,d,h,w-> d,h,w,3    (dhw or whd) depend on the shape of image
                m2f_name = img_name[0][:13] + '_warpped_flow_vm.nii.gz'
                save_image(torch.permute(flow[0], (1, 2, 3, 0)), fixed_img[0], args.output_dir,
                           m2f_name)

                # m_name = "{}_warped_lapirn.nii.gz".format(img_name[0][:13])
                # # save_img(X_Y, args.output_dir + '/' + file_name + '_warpped_moving.nii.gz')
                # save_image(X_Y, fixed_img, args.output_dir, m_name)

                m_name = "{}_warped_vm.nii.gz".format(img_name[0][:13])
                save_image(warped_image, fixed_img, args.output_dir, m_name)

    mean_total = np.mean(losses, 0)
    mean_mse = mean_total[0]
    mean_jac = mean_total[1]
    mean_ssim = mean_total[2]
    mean_ncc = mean_total[3]
    # print('mean TRE=%.2f+-%.2f MSE=%.3f Jac=%.6f' % (mean_tre, mean_std, mean_mse, mean_jac))
    print('mean SSIM=%.5f Jac=%.8f MSE=%.5f NCC=%.5f' % (mean_ssim, mean_jac, mean_mse, mean_ncc))


def validation(args, model, loss_similarity):
    transform = SpatialTransform_unit().cuda()
    transform.eval()

    with torch.no_grad():
        model.eval()  # m_name = "{}_affine.nii.gz".format(moving[1][0][:13])
        losses = []
        for batch, (moving, fixed) in enumerate(val_loader):
            input_moving = moving[0].to('cuda').float()
            input_fixed = fixed[0].to('cuda').float()

            res = model(input_moving,input_fixed)  # b, c, d, h, w  disp, scale_disp, warped
            warped_image, flow = res['warped_img'], res['disp']

            mse_loss = MSE(warped_image, input_fixed)
            ncc_loss_ori = loss_similarity(warped_image, input_fixed)
            grad_loss = args.alpha * gradient_loss(flow)

            # F_X_Y_norm = transform_unit_flow_to_flow_cuda(flow.permute(0, 2, 3, 4, 1).clone())

            # loss_Jacobian = neg_Jdet_loss(flow.permute(0, 2, 3, 4, 1), grid)

            loss_sum = ncc_loss_ori + grad_loss

            losses.append([ncc_loss_ori.item(), mse_loss.item(), grad_loss.item(), loss_sum.item()])

        mean_loss = np.mean(losses, 0)
        return mean_loss[0], mean_loss[1], mean_loss[2], mean_loss[3]


def train_unet():
    # load file
    # test_fixed_folder = os.path.join(args.test_dir, 'fixed')
    # test_moving_folder = os.path.join(args.test_dir, 'moving')

    # test_fixed_list = sorted(
    #     [os.path.join(test_fixed_folder, file_name) for file_name in os.listdir(test_fixed_folder) if
    #      file_name.lower().endswith('.gz')])
    # test_moving_list = sorted(
    #     [os.path.join(test_moving_folder, file_name) for file_name in os.listdir(test_moving_folder) if
    #      file_name.lower().endswith('.gz')])

    model = regnet.RegNet_pairwise(3, scale=0.5, depth=5, initial_channels=args.initial_channels, normalization=False)
    model = model.to(device)

    print(count_parameters(model.unet))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    image_loss_func = NCC(win=args.win_size)
    reg_loss = Grad('l2', loss_mult=2).loss

    # load data
    train_dataset = Dataset(moving_files=m_train_list, fixed_files=f_train_list)
    print("Number of training images: ", len(train_dataset))
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
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
            warped_image, flow_m2f = res['warped_img'], res['disp']

            loss_list = []
            sim_loss = image_loss_func(warped_image, input_fixed)
            # grad_loss = args.alpha * reg_loss(None,flow_m2f)  # b*c*h*w*d
            grad_loss = torch.tensor(0., dtype=sim_loss.dtype, device=sim_loss.device)

            # _, _, z, y, x = flow.shape
            # flow[:, 2, :, :, :] = flow[:, 2, :, :, :] * (z - 1)
            # flow[:, 1, :, :, :] = flow[:, 1, :, :, :] * (y - 1)
            # flow[:, 0, :, :, :] = flow[:, 0, :, :, :] * (x - 1)
            # # loss_regulation = smoothloss(flow)

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
            # modelname = model_dir + '/' + model_name + "{:.4f}_stagelvl3_".format(best_loss) + str(step) + '.pth'
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(i) + '{:.4f}best.pth'.format(
                val_total_loss)
            print("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)
        else:
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(i) + '{:.4f}.pth'.format(
                val_total_loss)
            print("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        mean_loss = np.mean(np.array(loss_total), 0)
        print(
            "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                mean_loss, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        test_dirlab(args, model, test_loader_dirlab)

        stop_criterion.add(val_ncc_loss, val_jac_loss, val_total_loss, train_loss=mean_loss)
        if stop_criterion.stop():
            break


if __name__ == "__main__":

    args = get_args()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    model_dir = args.checkpoint_path

    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = "{}_unet_lr{}_".format(train_time, args.lr)

    set_seed(42)
    make_dirs()

    fixed_folder_train = os.path.join(args.train_dir, 'fixed')
    moving_folder_train = os.path.join(args.train_dir, 'moving')
    f_train_list = sorted(
        [os.path.join(fixed_folder_train, file_name) for file_name in os.listdir(fixed_folder_train) if
         file_name.lower().endswith('.gz')])
    m_train_list = sorted(
        [os.path.join(moving_folder_train, file_name) for file_name in os.listdir(moving_folder_train) if
         file_name.lower().endswith('.gz')])
    # img_shape = [144, 192, 160]
    # set gpu
    # landmark_list = load_landmarks(args.landmark_dir)
    device = args.device

    # ===========validation=================
    fixed_folder_val = os.path.join(args.val_dir, 'fixed')
    moving_folder = os.path.join(args.val_dir, 'moving')
    f_img_file_list_vali = sorted(
        [os.path.join(fixed_folder_val, file_name) for file_name in os.listdir(fixed_folder_val) if
         file_name.lower().endswith('.gz')])
    m_img_file_list_vali = sorted([os.path.join(moving_folder, file_name) for file_name in os.listdir(moving_folder) if
                                   file_name.lower().endswith('.gz')])

    val_dataset = Dataset(moving_files=m_img_file_list_vali, fixed_files=f_img_file_list_vali)
    val_loader = Data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # =======================================

    landmark_list = load_landmarks(args.landmark_dir)
    dir_fixed_folder = os.path.join(args.test_dir, 'fixed')
    dir_moving_folder = os.path.join(args.test_dir, 'moving')

    f_dir_file_list = sorted([os.path.join(dir_fixed_folder, file_name) for file_name in os.listdir(dir_fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_dir_file_list = sorted(
        [os.path.join(dir_moving_folder, file_name) for file_name in os.listdir(dir_moving_folder) if
         file_name.lower().endswith('.gz')])
    test_dataset_dirlab = DirLabDataset(moving_files=m_dir_file_list, fixed_files=f_dir_file_list,
                                        landmark_files=landmark_list)
    test_loader_dirlab = Data.DataLoader(test_dataset_dirlab, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_unet()

    # # ==============test====================
    # pa_fixed_folder = r'E:\datasets\registration\test_ori\fixed'
    # # pa_fixed_folder = r'D:\xxf\test_patient\fixed'
    # pa_moving_folder = r'E:\datasets\registration\test_ori\moving'
    # # pa_moving_folder = r'D:\xxf\test_patient\moving'
    #
    # f_patient_file_list = sorted(
    #     [os.path.join(pa_fixed_folder, file_name) for file_name in os.listdir(pa_fixed_folder) if
    #      file_name.lower().endswith('.gz')])
    # m_patient_file_list = sorted(
    #     [os.path.join(pa_moving_folder, file_name) for file_name in os.listdir(pa_moving_folder) if
    #      file_name.lower().endswith('.gz')])
    #
    # test_dataset_patient = PatientDataset(moving_files=m_patient_file_list, fixed_files=f_patient_file_list)
    # test_loader_patient = Data.DataLoader(test_dataset_patient, batch_size=args.batch_size, shuffle=False,
    #                                       num_workers=0)
    #
    # prefix = '2023-04-21-17-47-16'
    # model_dir = args.checkpoint_path
    # model = regnet.RegNet_pairwise(3, scale=0.5, depth=5, initial_channels=args.initial_channels, normalization=False)
    # model = model.to(device)
    #
    # model.load_state_dict(torch.load(os.path.join(model_dir, args.checkpoint_name))['model'])
    # if args.checkpoint_name is not None:
    #     test_dirlab(args, model, test_loader_dirlab)
    #     # test_patient(args, os.path.join(model_dir, args.checkpoint_name), True)
    # else:
    #     checkpoint_list = sorted([os.path.join(model_dir, file) for file in os.listdir(model_dir) if prefix in file])
    #     for checkpoint in checkpoint_list:
    #         print(checkpoint)
    #         test_dirlab(args, checkpoint)
    #         # test_patient(args, checkpoint)
    # # =======================================
