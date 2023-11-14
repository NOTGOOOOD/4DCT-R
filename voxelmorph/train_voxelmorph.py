import os
import warnings
import torch
import numpy as np
import torch.utils.data as Data
from tqdm import tqdm
import logging
import time

from utils.config import get_args
from utils.datagenerators import Dataset, PatientDataset, DirLabDataset, build_dataloader_dirlab
from voxelmorph.vmmodel import vmnetwork
from voxelmorph.vmmodel.losses import Grad, MSE
from utils.losses import NCC as NCC_new
from utils.utilize import set_seed, load_landmarks, save_model, count_parameters
from utils.scheduler import StopCriterion
from utils.metric import get_test_photo_loss, landmark_loss
from utils.Functions import validation_vm, test_dirlab

from GDIR.model import regnet

args = get_args()

# def test_dirlab(args, model):
#     with torch.no_grad():
#         model.eval()
#         losses = []
#         for batch, (moving, fixed, landmarks, img_name) in enumerate(test_loader_dirlab):
#             moving_img = moving.to(args.device).float()
#             fixed_img = fixed.to(args.device).float()
#             landmarks00 = landmarks['landmark_00'].squeeze().cuda()
#             landmarks50 = landmarks['landmark_50'].squeeze().cuda()
#
#             # y_pred = model(moving_img, fixed_img, True)  # b, c, d, h, w warped_image, flow_m2f
#             y_pred = model(moving_img, fixed_img)
#             flow = y_pred['flow']
#
#             crop_range = args.dirlab_cfg[batch + 1]['crop_range']
#             # TRE
#             _mean, _std = landmark_loss(flow[0], landmarks00 - torch.tensor(
#                 [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
#                                         landmarks50 - torch.tensor(
#                                             [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1,
#                                                                                                                   3).cuda(),
#                                         args.dirlab_cfg[batch + 1]['pixel_spacing'],
#                                         fixed_img.cpu().detach().numpy()[0, 0])
#
#             losses.append([_mean.item(), _std.item()])
#             # print('case=%d after warped, TRE=%.2f+-%.2f' % (
#             #     batch + 1, _mean.item(), _std.item()))
#
#     mean_total = np.mean(losses, 0)
#     mean_tre = mean_total[0]
#     mean_std = mean_total[1]
#
#     print('mean TRE=%.2f+-%.2f' % (
#         mean_tre, mean_std))

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def train():
    # set gpu
    # landmark_list = load_landmarks(args.landmark_dir)
    device = args.device
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    model = vmnetwork.VxmDense(
        dim=3,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=args.bidir,
        int_steps=7,
        int_downsize=2
    )
    # model = regnet.RegNet_pairwise(3, scale=0.5, depth=5, initial_channels=args.initial_channels, normalization=False)
    model = model.to(device)
    print(count_parameters(model))
    # Set optimizer and losses
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # prepare image loss
    if args.sim_loss == 'ncc':
        # image_loss_func = NCC([args.win_size]*3).loss
        image_loss_func = NCC_new(win=args.win_size)
    elif args.sim_loss == 'mse':
        image_loss_func = MSE().loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

    # # need two image loss functions if bidirectional
    # if args.bidir:
    #     losses = [image_loss_func, image_loss_func]
    #     weights = [0.5, 0.5]
    # else:
    #     losses = [image_loss_func]
    #     weights = [1]

    # prepare deformation loss
    regular_loss = Grad('l2', loss_mult=2).loss

    # # set scheduler
    # scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.n_iter)
    # stop_criterion = StopCriterion(stop_std=args.stop_std, query_len=args.stop_query_len)


    # test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    stop_criterion = StopCriterion()
    best_loss = 99.
    # Training
    for i in range(0, args.n_iter + 1):
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

            y_true = [input_fixed, input_moving] if args.bidir else [input_fixed, None]
            # y_pred = model(input_moving, input_fixed)  # b, c, d, h, w warped_image, flow_m2f
            # flow, warped_image = y_pred[2], y_pred[0]

            res = model(input_moving, input_fixed)  # b, c, d, h, w  disp, scale_disp, warped
            warped_image, flow = res['warped_img'], res['flow_unit']

            loss_list = []
            r_loss = args.alpha * regular_loss(None, flow)
            sim_loss = image_loss_func(y_true[0], warped_image)

            # _, _, z, y, x = flow.shape
            # flow[:, 2, :, :, :] = flow[:, 2, :, :, :] * (z - 1)
            # flow[:, 1, :, :, :] = flow[:, 1, :, :, :] * (y - 1)
            # flow[:, 0, :, :, :] = flow[:, 0, :, :, :] * (x - 1)
            # # loss_regulation = smoothloss(flow)

            loss = r_loss + sim_loss
            loss_list.append(r_loss.item())
            loss_list.append(sim_loss.item())

            loss_total.append(loss.item())

            moving_name = moving_file[1][0]
            logging.info("img_name:{}".format(moving_name))
            if args.bidir:
                logging.info("iter: %d batch: %d  loss: %.5f  sim: %.5f bisim: %.5f  grad: %.5f" % (
                    i, i_step, loss.item(), loss_list[0], loss_list[1], loss_list[2]))
            else:
                logging.info("iter: %d batch: %d  loss: %.5f  sim: %.5f  grad: %.5f" % (
                    i, i_step, loss.item(), loss_list[0], loss_list[1]))

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f, grad=%.5f)" % (
                i_step, len(train_loader), loss.item(), r_loss.item())
            )
            # Backwards and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # if i % args.n_save_iter == 0:
            #     # save warped image0
            #     m_name = "{}_{}.nii.gz".format(i, moving_name)
            #     save_image(warped_image, input_fixed, args.output_dir, m_name)
            #     print("warped images have saved.")
            #
            #     # Save DVF
            #     # b,3,d,h,w-> w,h,d,3
            #     m2f_name = str(i) + "_dvf.nii.gz"
            #     save_image(torch.permute(flow_m2f[0], (3, 2, 1, 0)), input_fixed, args.output_dir,
            #                m2f_name)
            #     print("dvf have saved.")

        val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss = validation_vm(args, model,
                                                                                 image_loss_func
                                                                                 )
        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            # modelname = model_dir + '/' + model_name + "{:.4f}_stagelvl3_".format(best_loss) + str(step) + '.pth'
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(i) + '{:.4f}best.pth'.format(
                val_total_loss)
        else:
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(i) + '{:.4f}.pth'.format(
                val_total_loss)

        save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list, stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)
        logging.info("save model:{}".format(modelname))
        mean_loss = np.mean(np.array(loss_total), 0)
        mean_tre = test_dirlab(args, model, test_loader_dirlab, is_train=True)

        print(
            "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val mse loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                mean_loss, val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss))

        stop_criterion.add(val_ncc_loss, val_jac_loss, val_total_loss, train_loss=mean_loss)
        if stop_criterion.stop():
            break


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    set_seed(42)
    model_dir = args.checkpoint_path
    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = "{}_ccesingle_".format(train_time)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    log_index = len([file for file in os.listdir(args.log_dir) if file.endswith('.txt')])

    logging.basicConfig(level=logging.INFO,
                        filename=f'Log/log{log_index}.txt',
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    train_loader = build_dataloader_dirlab(args, mode='train')
    test_loader_dirlab = build_dataloader_dirlab(args, mode='test')

    train()
