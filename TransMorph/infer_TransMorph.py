import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph

from utilses.config import get_args
from utilses.datagenerators import PatientDataset, DirLabDataset
import torch.utils.data as Data
from utilses.metric import MSE, SSIM, NCC, jacobian_determinant, landmark_loss
from utilses.utilize import save_image, load_landmarks
from layers import SpatialTransformer


def main(args, checkpoint, is_save=False):
    # test_dir = 'D:\datasets/train_val_test/test_ori'



    # weights = [1, 0.02]
    # model_folder = 'TransMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    # model_dir = 'experiments/' + model_folder
    # dict = utils.process_label()
    # if os.path.exists('experiments/'+model_folder[:-1]+'.csv'):
    #     os.remove('experiments/'+model_folder[:-1]+'.csv')
    # csv_writter(model_folder[:-1], 'experiments/' + model_folder[:-1])
    # line = ''
    # for i in range(46):
    #     line = line + ',' + dict[i]
    # csv_writter(line, 'experiments/' + model_folder[:-1])
    # ckpt = torch.load(checkpoint)['model']







    model.load_state_dict(torch.load(checkpoint)['model'])

    # best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    # print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    # model.load_state_dict(best_model)
    model.cuda()
    # reg_model = utils.register_model((160, 192, 224), 'nearest')
    # reg_model.cuda()
    # test_composed = transforms.Compose([trans.Seg_norm(),
    #                                     trans.NumpyType((np.float32, np.int16)),
    #                                     ])
    # test_set = datasets.JHUBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    # eval_dsc_def = utils.AverageMeter()
    # eval_dsc_raw = utils.AverageMeter()
    # eval_det = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        losses = []
        model.eval()
        # for batch, (moving, fixed, img_name) in enumerate(test_loader_dirlab):
        for batch, (moving, fixed, landmarks, img_name) in enumerate(test_loader_dirlab):
            x = moving.to(args.device).float()
            y = fixed.to(args.device).float()
            landmarks00 = landmarks['landmark_00'].squeeze().cuda()
            landmarks50 = landmarks['landmark_50'].squeeze().cuda()
            # data = [t.cuda() for t in data]
            # x = data[0]
            # y = data[1]
            x_in = torch.cat((x,y),dim=1)
            flow = model(x_in, False)#warped,DVF
            x_def = STN(x, flow)

            ncc = NCC(y.cpu().detach().numpy(),x_def.cpu().detach().numpy())
            jac = jacobian_determinant(flow.squeeze().cpu().detach().numpy())
            _mse = MSE(y, x_def)

            _ssim = SSIM(y.cpu().detach().numpy()[0, 0],x_def.cpu().detach().numpy()[0, 0])

            crop_range = args.dirlab_cfg[batch + 1]['crop_range']
            # TRE
            _mean, _std = landmark_loss(flow[0], landmarks00 - torch.tensor(
                [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
                                        landmarks50 - torch.tensor(
                                            [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1,
                                                                                                                  3).cuda(),
                                        args.dirlab_cfg[batch + 1]['pixel_spacing'],
                                        y.cpu().detach().numpy()[0, 0], is_save)
            losses.append([_mean.item(), _std.item(), _mse.item(), jac, ncc.item(), _ssim.item()])
            print('case=%d after warped, TRE=%.2f+-%.2f MSE=%.5f Jac=%.6f ncc=%.6f ssim=%.6f' % (
                batch + 1, _mean.item(), _std.item(), _mse.item(), jac, ncc.item(), _ssim.item()))

            # def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            # tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            # jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            # line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            # line = line #+','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            # csv_writter(line, 'experiments/' + model_folder[:-1])
            # eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            # print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            # dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            # dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            # print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            # eval_dsc_def.update(dsc_trans.item(), x.size(0))
            # eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            # stdy_idx += 1

            # flip moving and fixed images
            y_in = torch.cat((y, x), dim=1)
            flow = model(y_in,False)
            y_def = STN(y, flow)

            ncc = NCC(x.cpu().detach().numpy(), y_def.cpu().detach().numpy())
            jac = jacobian_determinant(flow.squeeze().cpu().detach().numpy())
            _mse = MSE(x, y_def)
            _ssim = SSIM(x.cpu().detach().numpy()[0, 0], y_def.cpu().detach().numpy()[0, 0])
            crop_range = args.dirlab_cfg[batch + 1]['crop_range']
            # TRE
            _mean, _std = landmark_loss(flow[0], landmarks00 - torch.tensor(
                [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 3).cuda(),
                                        landmarks50 - torch.tensor(
                                            [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1,
                                                                                                                  3).cuda(),
                                        args.dirlab_cfg[batch + 1]['pixel_spacing'],
                                        y.cpu().detach().numpy()[0, 0], is_save)
            losses.append([_mean.item(), _std.item(), _mse.item(), jac, ncc.item(), _ssim.item()])
            print('case=%d after warped, TRE=%.2f+-%.2f MSE=%.5f Jac=%.6f ncc=%.6f ssim=%.6f' % (
                batch + 1, _mean.item(), _std.item(), _mse.item(), jac, ncc.item(), _ssim.item()))
            # def_out = reg_model([y_seg.cuda().float(), flow.cuda()])
            # tar = x.detach().cpu().numpy()[0, 0, :, :, :]

            # jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            # line = utils.dice_val_substruct(def_out.long(), x_seg.long(), stdy_idx)
            # line = line #+ ',' + str(np.sum(jac_det < 0) / np.prod(tar.shape))
            # out = def_out.detach().cpu().numpy()[0, 0, :, :, :]
            # print('det < 0: {}'.format(np.sum(jac_det <= 0)/np.prod(tar.shape)))
            # csv_writter(line, 'experiments/' + model_folder[:-1])
            # eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            # dsc_trans = utils.dice_val(def_out.long(), x_seg.long(), 46)
            # dsc_raw = utils.dice_val(y_seg.long(), x_seg.long(), 46)
            # print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            # eval_dsc_def.update(dsc_trans.item(), x.size(0))
            # eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            # stdy_idx += 1


            # if is_save:
            #     # Save DVF
            #     # b,3,d,h,w-> d,h,w,3    (dhw or whd) depend on the shape of image
            #     m2f_name = img_name[0][:13] + '_flow_vm.nii.gz'
            #     save_image(torch.permute(flow[0], (1, 2, 3, 0)), y[0], args.output_dir,
            #                m2f_name)
            #
            #     m_name = "{}_warped_vm.nii.gz".format(img_name[0][:13])
            #     save_image(x_def, y, args.output_dir, m_name)
    mean_total = np.mean(losses, 0)
    mean_tre = mean_total[0]
    mean_std = mean_total[1]
    mean_mse = mean_total[2]
    mean_jac = mean_total[3]
    mean_ncc = mean_total[4]
    mean_ssim = mean_total[5]
    print('mean TRE=%.2f+-%.2f MSE=%.3f Jac=%.6f ncc=%.6f ssim=%.6f' % (
        mean_tre, mean_std, mean_mse, mean_jac, mean_ncc, mean_ssim))

        # print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
        #                                                                             eval_dsc_def.std,
        #                                                                             eval_dsc_raw.avg,
        #                                                                             eval_dsc_raw.std))
        # print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    # GPU_iden = 1
    # GPU_num = torch.cuda.device_count()
    # print('Number of GPU: ' + str(GPU_num))
    # for GPU_idx in range(GPU_num):
    #     GPU_name = torch.cuda.get_device_name(GPU_idx)
    #     print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    # torch.cuda.set_device(GPU_iden-1)
    # GPU_avai = torch.cuda.is_available()
    # print('Currently using: ' + torch.cuda.get_device_name(GPU_iden-1))
    # print('If the GPU is available? ' + str(GPU_avai))
    #
    args = get_args()
    device = args.device

    STN = SpatialTransformer()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    # pa_fixed_folder = r'D:\xxf\test_patient\fixed'
    # pa_moving_folder = r'D:\xxf\test_patient\moving'
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

    prefix = '2023-05-19-15-32-20'
    model_dir = args.checkpoint_path

    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    print(model.state_dict())


    if args.checkpoint_name is not None:
        # test_dirlab(args, os.path.join(model_dir, args.checkpoint_name), True)
        main(args, os.path.join(model_dir, args.checkpoint_name), True)
    else:
        checkpoint_list = sorted([os.path.join(model_dir, file) for file in os.listdir(model_dir) if prefix in file])
        for checkpoint in checkpoint_list:
            print(checkpoint)
            # test_dirlab(args, checkpoint)
            main(args, checkpoint)

