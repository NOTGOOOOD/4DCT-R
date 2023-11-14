import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import logging
import time
import torch.utils.data as Data

from utils.config import get_args
from utils.losses import NCC, neg_Jdet_loss, Grad3d, Grad
from utils.datagenerators import Dataset, DirLabDataset
from utils.metric import landmark_loss, jacobian_determinant, SSIM, NCC as mtNCC
from utils.utilize import save_model, load_landmarks, set_seed, count_parameters
from utils.scheduler import StopCriterion
from utils.Functions import SpatialTransformer
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph

set_seed(1024)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


def make_dirs():
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


def test_dirlab(args, model):
    with torch.no_grad():
        losses = []
        # losses_re = []
        model.eval()
        for batch, (moving, fixed, landmarks, img_name) in enumerate(test_loader):
            x = moving.to(args.device).float()
            y = fixed.to(args.device).float()
            landmarks00 = landmarks['landmark_00'].squeeze().cuda()
            landmarks50 = landmarks['landmark_50'].squeeze().cuda()

            x_in = torch.cat((x, y), dim=1)
            flow = model(x_in, True)  # warped,DVF

            x_def = STN(x, flow)

            ncc = mtNCC(y.cpu().detach().numpy(), x_def.cpu().detach().numpy())
            jac = jacobian_determinant(flow.squeeze().cpu().detach().numpy())
            ssim = SSIM(y.cpu().detach().numpy()[0, 0], x_def.cpu().detach().numpy()[0, 0])

            crop_range = args.dirlab_cfg[batch + 1]['crop_range']
            # TRE
            _mean, _std = landmark_loss(flow[0], landmarks00 - torch.tensor(
                [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
                                        landmarks50 - torch.tensor(
                                            [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1,
                                                                                                                  3).cuda(),
                                        args.dirlab_cfg[batch + 1]['pixel_spacing'],
                                        y.cpu().detach().numpy()[0, 0])
            # print('case=%d after warped, TRE=%.2f+-%.2f' % (
            #     batch + 1, _mean.item(), _std.item()))

            # flip moving and fixed images
            # y_in = torch.cat((y, x), dim=1)
            # flow = model(y_in, True)
            # # TRE
            # _mean, _std = landmark_loss(flow[0], landmarks50 - torch.tensor(
            #     [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
            #                             landmarks00 - torch.tensor(
            #                                 [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1,
            #                                                                                                       3).cuda(),
            #                             args.dirlab_cfg[batch + 1]['pixel_spacing'],
            #                             y.cpu().detach().numpy()[0, 0])
            # losses_re.append([_mean.item(), _std.item()])
            # print('case=%d after warped, TRE=%.2f+-%.2f' % (
            #     batch + 1, _mean.item(), _std.item()))
            losses.append([_mean.item(), _std.item(), jac, ncc.item(), ssim.item()])
            # print('case=%d after warped, TRE=%.2f+-%.2f Jac=%.6f ncc=%.6f ssim=%.6f' % (
            #     batch + 1, _mean.item(), _std.item(), jac, ncc.item(), ssim.item()))


    mean_total = np.mean(losses, 0)
    mean_tre = mean_total[0]
    mean_std = mean_total[1]
    mean_jac = mean_total[2]
    mean_ncc = mean_total[3]
    mean_ssim = mean_total[4]
    print('mean TRE=%.2f+-%.2f Jac=%.6f ncc=%.6f ssim=%.6f' % (
        mean_tre, mean_std, mean_jac, mean_ncc, mean_ssim))


def main():
    weights = [1, 1]  # loss weights
    epoch_start = 0
    max_epoch = 50000  # max traning epoch
    cont_training = False  # if continue training

    image_loss_func_NCC = NCC(win=args.win_size)

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    print(count_parameters(model))
    model.cuda()

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    regular_loss = Grad('l2', loss_mult=2).loss
    image_loss_func = image_loss_func_NCC
    # criterion = image_loss_func_NCC
    # criterions = [criterion]
    # criterions += [Grad3d(
    #     penalty='l2')]  # criterions 被定义为一个列表，首先包含了一个均方误差损失函数 nn.MSELoss()，然后又添加了一个基于梯度的正则化损失函数 losses.Grad3d()。这个列表中的损失函数将在模型训练过程中被同时使用，以帮助模型学习更好的特征表示和更稳定的模型。

    stop_criterion = StopCriterion()
    if cont_training:
        checkpoint = r'D:\TransMorph_Transformer_for_Medical_Image_Registration_main\TransMorph\model\2023-05-07-15-12-52_TM__133_1.0887best.pth'
        model.load_state_dict(torch.load(checkpoint)['model'])
        optimizer.load_state_dict(torch.load(checkpoint)['optimizer'])

    # writer = SummaryWriter(log_dir = save_dir)
    best_loss = 99.
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')

        '''
        Training
        '''
        loss_all = AverageMeter()
        idx = 0
        model.train()
        # for data in train_loader:
        # adjust_learning_rate(optimizer, epoch, max_epoch, lr)
        for batch, (moving, fixed) in enumerate(train_loader):
            idx += 1
            x = moving[0].to(device).float()
            y = fixed[0].to(device).float()

            x_in = torch.cat((x, y), dim=1)
            output = model(x_in)

            loss = 0
            loss_vals = []

            sim_loss = image_loss_func(output[0], y)
            loss_vals.append(sim_loss)

            r_loss = args.alpha * regular_loss(None, output[1])
            loss_vals.append(r_loss)

            loss = r_loss + sim_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all.update(loss.item(), y.numel())
            del x_in
            del output
            del sim_loss
            del r_loss

            # flip fixed and moving images
            x_in = torch.cat((y, x), dim=1)
            output = model(x_in)
            loss = 0

            sim_loss = image_loss_func(output[0], x)
            r_loss = args.alpha * regular_loss(None, output[1])
            loss_vals[0] += sim_loss
            loss_vals[1] += r_loss

            loss = r_loss + sim_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item()/2, loss_vals[1].item()/2))
            sys.stdout.write(
                "\r" + 'Transmorph:step:batch "{0}:{1}" -> training loss "{2:.4f}" - sim_NCC "{3:4f}" -smo "{4:.4f}"'.format(
                    epoch, batch, loss.item(), loss_vals[0].item() / 2, loss_vals[1].item() / 2))
            sys.stdout.flush()

            logging.info("img_name:{}".format(moving[1][0]))
            logging.info("TM, epoch: %d  iter: %d batch: %d  loss: %.4f  sim: %.4f  grad: %.4f" % (
                epoch, idx, len(train_loader), loss.item(), loss_vals[0].item() / 2, loss_vals[1].item() / 2))

        print('Train: Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        # validation
        with torch.no_grad():
            # for data in val_loader:
            model.eval()
            losses = []
            for batch, (moving, fixed) in enumerate(val_loader):
                x = moving[0].to('cuda').float()
                y = fixed[0].to('cuda').float()
                x_in = torch.cat((x, y), dim=1)
                output = model(x_in)  # [warped,DVF]

                reg_loss = regular_loss(None, output[1])
                ncc_loss_ori = image_loss_func(output[0], y)

                loss_sum = ncc_loss_ori + args.alpha * reg_loss
                losses.append([ncc_loss_ori.item(), reg_loss.item(), loss_sum.item()])

            mean_loss = np.mean(losses, 0)
            val_ncc_loss, val_jac_loss, val_total_loss = mean_loss

        stop_criterion.add(val_ncc_loss, val_jac_loss, val_total_loss, train_loss=mean_loss)

        if val_total_loss <= best_loss:
            best_loss = val_total_loss
            # modelname = model_dir + '/' + model_name + "{:.4f}_stagelvl3_".format(best_loss) + str(step) + '.pth'
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(epoch) + '{:.4f}best.pth'.format(
                val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)
        else:
            modelname = model_dir + '/' + model_name + '_{:03d}_'.format(epoch) + '{:.4f}.pth'.format(
                val_total_loss)
            logging.info("save model:{}".format(modelname))
            save_model(modelname, model, stop_criterion.total_loss_list, stop_criterion.ncc_loss_list,
                       stop_criterion.jac_loss_list, stop_criterion.train_loss_list, optimizer)

        # mean_loss = np.mean(np.array(loss_total), 0)
        print(
            "\n one epoch pass. train loss %.4f . val ncc loss %.4f . val_jac_loss %.6f . val_total loss %.4f" % (
                loss_all.avg, val_ncc_loss, val_jac_loss, val_total_loss))

        loss_all.reset()

        # test
        # if epoch % 5 == 0:
        test_dirlab(args, model)

        if stop_criterion.stop():
            break


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


if __name__ == '__main__':
    args = get_args()
    lr = args.lr
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()  # GPU_num=1

    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):  # GPU_idx=0
        GPU_name = torch.cuda.get_device_name(GPU_idx)  # GPU_name='NVIDIA GeForce RTX 3060'
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)  # 单卡  使用指定的卡
    GPU_avai = torch.cuda.is_available()  # GPU_avai=true
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))

    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    # train
    train_fixed_folder = os.path.join(args.train_dir, 'fixed')
    train_moving_folder = os.path.join(args.train_dir, 'moving')
    f_train_list = sorted(
        [os.path.join(train_fixed_folder, file_name) for file_name in os.listdir(train_fixed_folder) if
         file_name.lower().endswith('.gz')])
    m_train_list = sorted(
        [os.path.join(train_moving_folder, file_name) for file_name in os.listdir(train_moving_folder) if
         file_name.lower().endswith('.gz')])
    train_dataset = Dataset(moving_files=m_train_list, fixed_files=f_train_list)
    train_loader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    # val
    val_fixed_folder = os.path.join(args.val_dir, 'fixed')
    val_moving_folder = os.path.join(args.val_dir, 'moving')
    f_val_list = sorted([os.path.join(val_fixed_folder, file_name) for file_name in os.listdir(val_fixed_folder) if
                         file_name.lower().endswith('.gz')])
    m_val_list = sorted([os.path.join(val_moving_folder, file_name) for file_name in os.listdir(val_moving_folder) if
                         file_name.lower().endswith('.gz')])
    val_dataset = Dataset(moving_files=m_val_list, fixed_files=f_val_list)
    val_loader = Data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # test
    landmark_list = load_landmarks(args.landmark_dir)
    dir_fixed_folder = os.path.join(args.test_dir, 'fixed')
    dir_moving_folder = os.path.join(args.test_dir, 'moving')

    f_dir_file_list = sorted([os.path.join(dir_fixed_folder, file_name) for file_name in os.listdir(dir_fixed_folder) if
                              file_name.lower().endswith('.gz')])
    m_dir_file_list = sorted(
        [os.path.join(dir_moving_folder, file_name) for file_name in os.listdir(dir_moving_folder) if
         file_name.lower().endswith('.gz')])
    test_dataset = DirLabDataset(moving_files=m_dir_file_list, fixed_files=f_dir_file_list,
                                 landmark_files=landmark_list)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_dir = 'model'
    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = "{}_TM_".format(train_time)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    make_dirs()

    log_index = len([file for file in os.listdir(args.log_dir) if file.endswith('.txt')])
    logging.basicConfig(level=logging.INFO,
                        filename=f'Log/log{log_index}.txt',
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    STN = SpatialTransformer()

    main()
