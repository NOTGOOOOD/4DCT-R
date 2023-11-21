import os
import sys
import numpy as np
import torch
import logging
import time
import torch.nn.functional as F

from utils.datagenerators import build_dataloader_dirlab
from utils.config import get_args
from utils.losses import NCC, gradient_loss as smoothloss
from utils.scheduler import StopCriterion
from utils.utilize import set_seed, save_model, count_parameters, make_dirs
from utils.metric import MSE
from utils.Functions import validation_vm, test_dirlab
from tqdm import tqdm
from CCECor import CCECoNet, CCECoNetDual

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test(model):
    prefix = '2023-04-21-17-47-16'
    model_dir = args.checkpoint_path

    if args.checkpoint_name is not None:
        model.load_state_dict(torch.load(os.path.join(model_dir, args.checkpoint_name))['model'])
        test_dirlab(args, model, test_loader_dirlab, is_train=False)
        # test_patient(args, os.path.join(model_dir, args.checkpoint_name), True)
    else:
        checkpoint_list = sorted([os.path.join(model_dir, file) for file in os.listdir(model_dir) if prefix in file])
        for checkpoint in checkpoint_list:
            print(checkpoint)
            model.load_state_dict(torch.load(checkpoint)['model'])
            test_dirlab(args, model, test_loader_dirlab, is_train=False)

def train(model):
    print("Training CCE_single...")
    device = args.device

    print(count_parameters(model))

    loss_similarity = NCC(win=5)
    loss_smooth = smoothloss

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = args.checkpoint_path

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    stop_criterion = StopCriterion()
    best_loss = 99.

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
            res = model(input_moving, input_fixed)  # b, c, d, h, w  disp, scale_disp, warped
            warped_image, flow = res['warped_img'], res['flow']

            loss_list = []
            r_loss = args.alpha * loss_smooth(flow)
            sim_loss = loss_similarity(y_true[0], warped_image)

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


        val_ncc_loss, val_mse_loss, val_jac_loss, val_total_loss = validation_vm(args, model, loss_similarity)
        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            # modelname = model_dir + '/' + model_name + "{:.4f}_stagelvl3_".format(best_loss) + str(step) + '.pth'
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(i) + '{:.4f}best.pth'.format(
                val_total_loss)
        else:
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(i) + '{:.4f}.pth'.format(
                val_total_loss)

        save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                   stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)
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
    set_seed(48)
    args = get_args()
    make_dirs(args)

    log_index = len([file for file in os.listdir(args.log_dir) if file.endswith('.txt')])

    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = "{}_CCENet_dual_co_lr{}".format(train_time, args.lr)

    logging.basicConfig(level=logging.INFO,
                        filename=f'Log/log{log_index}.txt',
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    lr = args.lr
    antifold = args.antifold
    smooth = args.smooth

    train_loader = build_dataloader_dirlab(args, 'train')
    val_loader = build_dataloader_dirlab(args, 'val')
    test_loader_dirlab = build_dataloader_dirlab(args, 'test')

    model = CCECoNetDual(dim=3).to(args.device)
    # from thop import profile
    # tensor = (torch.randn(1,1,96,144,144).cuda().float(), torch.randn(1,1,96,144,144).cuda().float(),)
    # flops, params = profile(model, tensor)

    train(model)
    # test(model)