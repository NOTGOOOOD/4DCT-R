import time

import GDIR.model.regnet, GDIR.model.loss, GDIR.model.util, utils.structure, utils.utilize
import torch, os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from GDIR.process.processing import data_standardization_0_n, imgTomhd, data_standardization_min_max
import random
from utils.metric import SSIM, NCC as mtNCC, jacobian_determinant, landmark_loss, jacobian_determinant

plot_dpi = 300
import numpy as np
import logging, tqdm
from scipy import interpolate
from utils.utilize import save_image, get_project_path, make_dir, count_parameters,make_dirs, save_model
from utils.config import get_args
from GDIR.model.regnet import SpatialTransformer


def calc_tre(calcdisp, disp_i2t, disp_t2i, grid_tuple, landmark_00_converted, landmark_disp, spacing):
    # x' = u(x) + x
    composed_disp = calcdisp.compose_disp(disp_i2t, disp_t2i, mode='all')
    composed_disp_np = composed_disp.cpu().numpy()  # (2, 2, 3, d, h, w)

    inter = interpolate.RegularGridInterpolator(grid_tuple, np.moveaxis(composed_disp_np[0, 1], 0, -1))
    calc_landmark_disp = inter(landmark_00_converted)

    diff = (np.sum(((calc_landmark_disp - landmark_disp) * spacing) ** 2, 1)) ** 0.5
    # diff = (np.sum((calc_landmark_disp - landmark_disp) ** 2, 1)) ** 0.5
    diff = diff[~np.isnan(diff)]

    return np.mean(diff), np.std(diff), diff, composed_disp_np


def preprocess(project_path, cfg):
    for i in range(1, 11):
        case = i
        shape = cfg[i]["orign_size"]
        crop_range = cfg[i]["crop_range"]
        img_path = os.path.join(project_path.split("4DCT-R")[0], f'datasets/dirlab/img/Case{case}Pack/Images')
        save_path = os.path.join(project_path.split("4DCT-R")[0], f'datasets/dirlab/mhd_resample/case{case}')
        make_dir(save_path)

        imgTomhd(img_path, save_path, np.int16, shape, case, crop_range, True)


def show_slice(img_mov, img_ref=None):
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img_mov[:, :, 200], cmap='gray')
    ax[1].imshow(img_mov[:, 200, :], cmap='gray')
    ax[2].imshow(img_mov[50, :, :], cmap='gray')
    plt.show()

    if img_ref:
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(img_ref[:, :, 200], cmap='gray')
        ax[1].imshow(img_ref[:, 200, :], cmap='gray')
        ax[2].imshow(img_ref[50, :, :], cmap='gray')
        plt.show()


def set_seed(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def train(args, case=1):
    # logger.add('{time:YYYY-MM-DD HHmmss}.log', format="{message}", rotation='5 MB', encoding='utf-8')
    states_folder = args.checkpoint_path
    make_dirs(args)

    # d,h,w
    crop_range0 = args.dirlab_cfg[case]["crop_range"][0]
    crop_range1 = args.dirlab_cfg[case]["crop_range"][1]
    crop_range2 = args.dirlab_cfg[case]["crop_range"][2]
    crop_range0_start = crop_range0.start
    crop_range1_start = crop_range1.start
    crop_range2_start = crop_range2.start

    # landmark
    landmark_file = os.path.join(project_path, 'data/dirlab/Case%02d_300_00_50.pt' % (case) )
    landmark_info = torch.load(landmark_file)
    landmark_disp = landmark_info['disp_00_50']  # w, h, d  x,y,z
    landmark_00 = landmark_info['landmark_00']
    landmark_50 = landmark_info['landmark_50']

    landmark_00_converted = np.flip(landmark_00, axis=1) - np.array(
        [crop_range0_start, crop_range1_start, crop_range2_start], dtype=np.float32)

    landmark_50_converted = np.flip(landmark_50, axis=1) - np.array(
        [crop_range0_start, crop_range1_start, crop_range2_start], dtype=np.float32)

    # file
    data_folder = 'D:/xxf/dirlab_1250/case%02d' % case
    image_file_list = sorted([file_name for file_name in os.listdir(data_folder) if file_name.lower().endswith('.gz')])
    image_list = []
    for file_name in image_file_list:
        # xyz W H D
        img_sitk = sitk.ReadImage(os.path.join(data_folder, file_name))
        # zyx D H W
        stkimg = sitk.GetArrayFromImage(img_sitk)

        image_list.append(stkimg)

    input_image = torch.stack([torch.from_numpy(image)[None] for image in image_list], 0)
    input_image = input_image.float()

    if config.group_index_list is not None:
        input_image = input_image[config.group_index_list]

    # normalize [0,1]
    input_image = data_standardization_0_n(1, input_image)

    image_shape = np.array(input_image.shape[2:])  # (d, h, w) z y x
    num_image = input_image.shape[0]  # number of image in the group
    regnet = GDIR.model.regnet.RegNet_single(dim=config.dim, n=num_image, scale=config.scale, depth=config.depth,
                                             initial_channels=config.initial_channels,
                                             normalization=config.normalization, implict_tmp=config.Imp_temp)

    # n = 5
    ncc_loss = GDIR.model.loss.NCC(config.dim, config.ncc_window_size)
    regnet = regnet.to(device)
    input_image = input_image.to(device)
    ncc_loss = ncc_loss.to(device)
    optimizer = torch.optim.Adam(regnet.parameters(), lr=config.learning_rate)
    calcdisp = GDIR.model.util.CalcDisp(dim=config.dim, calc_device='cuda')

    if config.load:
        state_file = os.path.join(states_folder, config.load)
        if os.path.exists(state_file):
            state_file = os.path.join(states_folder, config.load)
            states = torch.load(state_file, map_location=device)
            regnet.load_state_dict(states['model'])
            if config.load_optimizer:
                optimizer.load_state_dict(states['optimizer'])

    grid_tuple = [np.arange(grid_length, dtype=np.float32) for grid_length in image_shape]
    diff_stats = []
    stop_criterion = GDIR.model.util.StopCriterion(stop_std=config.stop_std, query_len=config.stop_query_len)
    pbar = tqdm.tqdm(range(config.max_num_iteration))

    ## %% test landmarks
    # lmk_id = 30
    # # before resampling
    # lm1_mov = landmark_00[lmk_id]
    # lm1_ref = landmark_50[lmk_id]
    #
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(image_list[0][lm1_mov[2]], cmap='gray')
    # ax[0].scatter([lm1_mov[0]], [lm1_mov[1]], 50, color='red')
    # ax[0].set_title('mov')
    # ax[1].imshow(image_list[5][lm1_ref[2]], cmap='gray')
    # ax[1].scatter([lm1_ref[0]], [lm1_ref[1]], 50, color='red')
    # ax[1].set_title('ref')
    # plt.show()
    #
    # # after resampling
    # # crop
    # crop_range0 = cfg[case]["crop_range"][0]
    # crop_range1 = cfg[case]["crop_range"][1]
    # crop_range2 = cfg[case]["crop_range"][2]
    #
    # mov1cc = image_list[0][crop_range0, crop_range1, crop_range2]
    # ref1cc = image_list[5][crop_range0, crop_range1, crop_range2]
    #
    # mov_lmk_int = np.round(np.flip(landmark_00_converted, axis=1)).astype('int32')
    # ref_lmk_int = np.round(np.flip(landmark_50_converted, axis=1) ).astype('int32')
    #
    # lm1_mov0 = mov_lmk_int[lmk_id]
    # lm1_ref0 = ref_lmk_int[lmk_id]
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(mov1cc[lm1_mov0[2]], cmap='gray')
    # ax[0].scatter([lm1_mov0[0]], [lm1_mov0[1]], 50, color='red')
    # ax[0].set_title('mov')
    # ax[1].imshow(ref1cc[lm1_ref0[2]], cmap='gray')
    # ax[1].scatter([lm1_ref0[0]], [lm1_ref0[1]], 50, color='red')
    # ax[1].set_title('ref')
    # plt.show()

    # %%
    diff_ori = (np.sum((landmark_disp * args.dirlab_cfg[case]['pixel_spacing']) ** 2, 1)) ** 0.5
    print("\ncase{0}配准前 diff: {1}({2})".format(case, np.mean(diff_ori), np.std(diff_ori)))

    best_tre = 99.
    for i in pbar:
        res = regnet(input_image)
        total_loss = 0.
        if 'disp_i2t' in res:
            simi_loss = (ncc_loss(res['warped_input_image'], res['template']) + ncc_loss(input_image, res['warped_template'])) / 2.
        else:
            simi_loss = ncc_loss(res['warped_input_image'], res['template'])
        total_loss += simi_loss

        if config.smooth_reg > 0:
            if 'disp_i2t' in res:
                smooth_loss = (GDIR.model.loss.smooth_loss(res['scaled_disp_t2i']) + GDIR.model.loss.smooth_loss(
                    res['scaled_disp_i2t'])) / 2.
            else:
                # smooth_loss = model.loss.smooth_loss(res['scaled_disp_t2i'])
                smooth_loss = GDIR.model.loss.smooth_loss(res['scaled_disp_t2i'], res['scaled_template'])
            total_loss += config.smooth_reg * smooth_loss
            smooth_loss_item = smooth_loss.item()
        else:
            smooth_loss_item = 0

        if config.cyclic_reg > 0:
            if 'disp_i2t' in res:
                # cyclic_loss = (torch.mean((torch.sum(res['scaled_disp_t2i'], 0))**2) + torch.mean((torch.sum(res['scaled_disp_i2t'], 0)))**0.5)/2.
                cyclic_loss = ((torch.mean((torch.sum(res['scaled_disp_t2i'], 0)) ** 2)) ** 0.5 + (
                    torch.mean((torch.sum(res['scaled_disp_i2t'], 0)) ** 2)) ** 0.5) / 2.
            else:
                cyclic_loss = (torch.mean((torch.sum(res['scaled_disp_t2i'], 0)) ** 2)) ** 0.5
            total_loss += config.cyclic_reg * cyclic_loss
            cyclic_loss_item = cyclic_loss.item()
        else:
            cyclic_loss_item = 0

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        stop_criterion.add(simi_loss.item())
        if stop_criterion.stop():
            break

        pbar.set_description(
            f'{i}, totalloss {total_loss:.6f}, simi loss {simi_loss.item():.6f}, smooth loss {smooth_loss_item:.3f}, cyclic loss {cyclic_loss_item:.3f}')

        if i % config.pair_disp_calc_interval == 0:
            if 'disp_i2t' in res:
                disp_i2t = res['disp_i2t'][config.pair_disp_indexes]
            else:
                disp_i2t = calcdisp.inverse_disp(res['disp_t2i'][config.pair_disp_indexes])
            if config.Imp_temp:
                mean, std, diff, compse_disp = calc_tre(calcdisp, disp_i2t, res['disp_t2i'][config.pair_disp_indexes],
                                              grid_tuple, landmark_00_converted, landmark_disp,
                                              args.dirlab_cfg[case]['pixel_spacing'])
            else:
                disp = res['disp_t2i']
                mean, std = landmark_loss(disp[0].detach().cpu(), torch.tensor(landmark_00_converted).flip(1),
                                        torch.tensor(landmark_50_converted).flip(1),
                                        args.dirlab_cfg[case]['pixel_spacing'])
                mean = mean.item()
                std = std.item()

            diff_stats.append([i, mean, std])
            # save_warp(args, np.expand_dims(compse_disp[0, 1], 0), input_image[0].unsqueeze(0), 'case1', 'gdir', spacing=args.dirlab_cfg[case]['pixel_spacing'])

            print(f'\n diff: {mean:.2f}+-{std:.2f}')

            states_file = f'{train_time}_case{case}_{mean:.2f}_scale{config.scale}.pth'
            save_path = os.path.join(states_folder, states_file)
            save_model(save_path,regnet,total_loss=total_loss.item(),simi_loss=simi_loss.item(),reg_loss=smooth_loss_item,train_loss=cyclic_loss_item,optimizer=None)

    if config.Imp_temp:
        disp, warped = get_flow50_00(args, res, input_image[0:1], spacing=args.dirlab_cfg[case]['pixel_spacing'],is_save=False)
        mean, std, diff, _ = calc_tre(calcdisp, disp_i2t, res['disp_t2i'][config.pair_disp_indexes],
                                                grid_tuple, landmark_00_converted, landmark_disp,
                                                args.dirlab_cfg[case]['pixel_spacing'])
    else:
        disp, warped = res['disp_t2i'], res['template'].squeeze().cpu().detach().numpy()
        mean, std = landmark_loss(disp[0].detach().cpu(), torch.tensor(landmark_00_converted).flip(1).cpu(),
                                torch.tensor(landmark_50_converted).flip(1).cpu(),
                                args.dirlab_cfg[case]['pixel_spacing'])
        mean = mean.item()
        std = std.item()

    ncc = mtNCC(warped, input_image[5:6].squeeze().cpu().detach().numpy())
    jac = jacobian_determinant(disp[0].cpu().detach().numpy())
    ssim = SSIM(warped, input_image[5][0].cpu().detach().numpy())
    print(f'\n finally, case{case} diff: {mean:.2f}+-{std:.2f} ncc {ncc:.4f} jac {jac:.8f} ssim {ssim:.4f}')

    states = {'config': config, 'model': regnet.state_dict(), 'optimizer': None,
              'loss_list': stop_criterion.loss_list, 'diff_stats': diff_stats}
    states_file = f'reg_dirlab_case{case}_{mean:.2f}_scale{config.scale}.pth'
    save_path = os.path.join(states_folder, states_file)
    torch.save(states, save_path)
    print(f'save model {states_file} in path:{save_path}')

def test(args, case=1, is_save=False, state_file='', logger=None):
    states_folder = args.checkpoint_path
    make_dirs(args)
    config.load = state_file
    # d,h,w
    crop_range0 = args.dirlab_cfg[case]["crop_range"][0]
    crop_range1 = args.dirlab_cfg[case]["crop_range"][1]
    crop_range2 = args.dirlab_cfg[case]["crop_range"][2]
    crop_range0_start = crop_range0.start
    crop_range1_start = crop_range1.start
    crop_range2_start = crop_range2.start

    # landmark
    landmark_file = os.path.join(project_path, 'data/dirlab/Case%02d_300_00_50.pt' % (case) )
    landmark_info = torch.load(landmark_file)
    landmark_disp = landmark_info['disp_00_50']  # w, h, d  x,y,z
    landmark_00 = landmark_info['landmark_00']
    landmark_50 = landmark_info['landmark_50']

    landmark_00_converted = np.flip(landmark_00, axis=1) - np.array(
        [crop_range0_start, crop_range1_start, crop_range2_start], dtype=np.float32)

    landmark_50_converted = np.flip(landmark_50, axis=1) - np.array(
        [crop_range0_start, crop_range1_start, crop_range2_start], dtype=np.float32)

    # file
    data_folder = 'D:/xxf/dirlab_1250/case%02d' % case
    image_file_list = sorted([file_name for file_name in os.listdir(data_folder) if file_name.lower().endswith('.gz')])
    image_list = []
    for file_name in image_file_list:
        # xyz W H D
        img_sitk = sitk.ReadImage(os.path.join(data_folder, file_name))
        # zyx D H W
        stkimg = sitk.GetArrayFromImage(img_sitk)

        image_list.append(stkimg)

    input_image = torch.stack([torch.from_numpy(image)[None] for image in image_list], 0)
    input_image = input_image.float()

    if config.group_index_list is not None:
        input_image = input_image[config.group_index_list]

    # normalize [0,1]
    input_image = data_standardization_0_n(1, input_image)

    image_shape = np.array(input_image.shape[2:])  # (d, h, w) z y x
    num_image = input_image.shape[0]  # number of image in the group
    regnet = GDIR.model.regnet.RegNet_single(dim=config.dim, n=num_image, scale=config.scale, depth=config.depth,
                                             initial_channels=config.initial_channels,
                                             normalization=config.normalization)
    regnet = regnet.to(device)
    input_image = input_image.to(device)
    calcdisp = GDIR.model.util.CalcDisp(dim=config.dim, calc_device='cuda')

    # from thop import profile
    # tensor = (torch.randn(input_image.shape).cuda().float(), )
    # flops, params = profile(regnet, tensor)

    state_file = os.path.join(states_folder, config.load)
    if os.path.exists(state_file):
        state_file = os.path.join(states_folder, config.load)
        states = torch.load(state_file, map_location=device)
        regnet.load_state_dict(states['model'])
        if logger:
            logger.info("\nload state {}".format(state_file))
        if config.load_optimizer:
            optimizer.load_state_dict(states['optimizer'])

    grid_tuple = [np.arange(grid_length, dtype=np.float32) for grid_length in image_shape]
    diff_stats = []

    # diff_ori = (np.sum((landmark_disp * args.dirlab_cfg[case]['pixel_spacing']) ** 2, 1)) ** 0.5
    # print("\ncase{0}配准前 diff: {1}({2})".format(case, np.mean(diff_ori), np.std(diff_ori)))

    res = regnet(input_image)
    if 'disp_i2t' in res:
        disp_i2t = res['disp_i2t'][config.pair_disp_indexes]
    else:
        disp_i2t = calcdisp.inverse_disp(res['disp_t2i'][config.pair_disp_indexes])

    if config.Imp_temp:
        disp, warped = get_flow50_00(args, res, input_image[0:1], spacing=args.dirlab_cfg[case]['pixel_spacing'],is_save=False)
        mean, std, diff, _ = calc_tre(calcdisp, disp_i2t, res['disp_t2i'][config.pair_disp_indexes],
                                                grid_tuple, landmark_00_converted, landmark_disp,
                                                args.dirlab_cfg[case]['pixel_spacing'])
        disp, warped = get_flow50_00(args, res, input_image[0:1], spacing=args.dirlab_cfg[case]['pixel_spacing'],
                                     is_save=is_save, prefix=f'case{case}', suffix=f'scale{config.scale}_tre{mean:.2f}')
    else:
        disp, warped = res['disp_t2i'], res['template'].squeeze().cpu().detach().numpy()
        mean, std = landmark_loss(disp[0], torch.tensor(landmark_00_converted).flip(1).cuda(),
                                torch.tensor(landmark_50_converted).flip(1).cuda(),
                                args.dirlab_cfg[case]['pixel_spacing'])
        mean = mean.item()
        std = std.item()


    ncc = mtNCC(warped, input_image[5:6].squeeze().cpu().detach().numpy())
    jac = jacobian_determinant(disp[0].cpu().detach().numpy())
    ssim = SSIM(warped, input_image[5][0].cpu().detach().numpy())
    if logger:
        logger.info(f'\n case{case} diff: {mean:.2f}+-{std:.2f} ncc {ncc:.4f} jac {jac:.8f} ssim {ssim:.4f}')
    print(state_file)
    print(f'\n case{case} diff: {mean:.2f}+-{std:.2f} ncc {ncc:.4f} jac {jac:.8f} ssim {ssim:.4f}')
    # return _mean.item(),_std.item(),ncc.item(),jac,ssim
    return mean, std, ncc.item(), jac,ssim

def get_flow50_00(args, res, input_image, spacing, is_save=False, prefix='', suffix=''):
    """

    Parameters
    ----------
    args: dict
    res: dict
    input_image: torch.Tensor :moving image.shape(c,d,h,w)

    Returns
    -------
    torch.Tensor: disp[n,c,d,h,w],
    ndarray: warped[d,h,w]
    """
    calc_disp = GDIR.model.util.CalcDisp(dim=config.dim, calc_device='cuda')
    disp_50_t = calc_disp.inverse_disp(res['disp_t2i'][5:6])
    disp50_00 = calc_disp.compose_disp(disp_50_t, res['disp_t2i'][0:1], mode = 'corr')

    wapred_img = get_warp(args, disp50_00, input_image, prefix, suffix, spacing, is_save)
    return disp50_00, wapred_img


def get_warp(args, disp, moving, prefix, suffix, spacing, is_save=False):
    if type(disp) != torch.Tensor:
        disp = torch.tensor(disp).cuda()
    warped = spatial_transform(moving, disp).detach().cpu().numpy()

    if is_save:
        # Save DVF
        # b,3,d,h,w-> d,h,w,3    (dhw or whd) depend on the shape of image
        m2f_name = '{}_warpped_flow_{}.nii.gz'.format(prefix, suffix)
        save_image(disp[0].permute((1, 2, 3, 0)), args.output_dir, m2f_name, spacing=spacing)

        m_name = '{}_warpped_img_{}.nii.gz'.format(prefix, suffix)
        save_image(warped.squeeze(), args.output_dir, m_name, spacing=spacing)

    return warped.squeeze()


if __name__ == '__main__':
    config = dict(
        dim=3,  # dimension of the input image
        scale=0.5,
        initial_channels=16,  # 4 8 16 32
        depth=4,
        max_num_iteration=300,
        normalization=True,  # whether use normalization layer
        learning_rate=1e-2,
        smooth_reg=1e-3,
        # cyclic_reg=1e-2,
        cyclic_reg=0,
        ncc_window_size=5,
        load=None,
        load_optimizer=False,
        group_index_list=None,
        pair_disp_indexes=[0, 5],
        pair_disp_calc_interval=3,
        stop_std=0.0007,
        stop_query_len=100,
        Imp_temp=False
    )
    config = utils.structure.Struct(**config)
    project_path = get_project_path("4DCT")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spatial_transform = SpatialTransformer(3)
    set_seed(48)
    args = get_args()
    train_time = time.strftime("%Y-%m-%d-%H-%M-%S")

    log_index = len([file for file in os.listdir(args.log_dir) if file.endswith('.txt')])
    logging.basicConfig(level=logging.INFO,
                        filename=f'Log/log{log_index}.txt',
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    # for i in range(1, 11):
    #     ckpt = [file for file in os.listdir(args.checkpoint_path) if f'case{i}_' in file]
    #     for file in ckpt:
    #         test(args, case=i, is_save=False, state_file=file, logger=logging)
    #
    for i in range(6, 11):
        train(args,i)

    # ckpt = [file for file in os.listdir(args.checkpoint_path) if 'case8_' in file]
    # for i in ckpt:
    #     test(args, case=8,is_save=False, state_file=i)

    ckpt_32_list = ['',
                       'reg_dirlab_case1_1.26(0.66)_scale0.5.pth',
                       'reg_dirlab_case2_1.16(0.55)_scale0.5.pth',
                       'reg_dirlab_case3_1.41(0.85)_scale0.5.pth',
                       'reg_dirlab_case4_1.76(1.19)_scale0.5.pth',
                       'reg_dirlab_case5_2.14(1.38)_scale0.5.pth',
                       'reg_dirlab_case6_4.70(4.16)_scale0.5.pth',
                       'reg_dirlab_case7_2.40(1.37)_scale0.5.pth',
                       'reg_dirlab_case8_5.57(4.22)_scale0.5.pth',
                       'reg_dirlab_case9_2.02(1.12)_scale0.5.pth',
                       'reg_dirlab_case10_2.85(2.46)_scale0.5.pth']
    ckpt_32_list_noreg = ['',
                    'noreg/reg_dirlab_case1_1.32_scale0.5.pth',
                    'noreg/reg_dirlab_case2_1.48_scale0.5.pth',
                    'noreg/reg_dirlab_case3_2.80_scale0.5.pth',
                    'noreg/reg_dirlab_case4_3.14_scale0.5.pth',
                    'noreg/reg_dirlab_case5_2.48_scale0.5.pth',
                    'noreg/reg_dirlab_case6_3.34_scale0.5.pth',
                    'noreg/reg_dirlab_case7_2.34_scale0.5.pth',
                    'noreg/reg_dirlab_case8_3.06_scale0.5.pth',
                    'noreg/reg_dirlab_case9_2.61_scale0.5.pth',
                    'noreg/reg_dirlab_case10_3.86_scale0.5.pth']

    ckpt_64_list=['',
                  'reg_dirlab_case1_1.07(0.54)_scale0.5.pth',
                  'reg_dirlab_case2_1.08(0.52)_scale0.5.pth',
                  'reg_dirlab_case3_1.26(0.72)_scale0.5.pth',
                  'reg_dirlab_case4_1.47(0.95)_scale0.5.pth',
                  'reg_dirlab_case5_2.15(1.40)_scale0.5.pth',
                  'reg_dirlab_case6_2.85(2.25)_scale0.5.pth',
                  'reg_dirlab_case7_2.11(1.13)_scale0.5.pth',
                  'reg_dirlab_case8_2.13(2.36)_scale0.5.pth',
                  'reg_dirlab_case9_1.51(0.80)_scale0.5.pth',
                  'reg_dirlab_case10_1.84(1.52)_scale0.5.pth']
    ckpt_64_list_noreg = ['',
                    'noreg/reg_dirlab_case1_1.09_scale0.5.pth',
                    'noreg/reg_dirlab_case2_1.44_scale0.5.pth',
                    'noreg/reg_dirlab_case3_2.37_scale0.5.pth',
                    'noreg/reg_dirlab_case4_1.71_scale0.5.pth',
                    'noreg/reg_dirlab_case5_1.97_scale0.5.pth',
                    'noreg/reg_dirlab_case6_2.65_scale0.5.pth',
                    'noreg/reg_dirlab_case7_2.39_scale0.5.pth',
                    'noreg/reg_dirlab_case8_2.84_scale0.5.pth',
                    'noreg/2023-11-26-17-00-49_case9_2.73_scale0.5.pth',
                    'noreg/2023-11-26-17-00-49_case10_2.25_scale0.5.pth']
    ckpt_128_list=['',
                  '2023-11-25-15-26-01_case1_1.07_scale0.5.pth',#0.99+-1.02 ncc 0.9924 jac 0.00000000 ssim 0.9014
                  '2023-11-25-15-26-01_case2_1.01_scale0.5.pth',#1.00+-1.01 ncc 0.9921 jac 0.00005819 ssim 0.8915
                  '2023-11-25-15-26-01_case3_1.36_scale0.5.pth',#1.43+-1.20 ncc 0.9923 jac 0.00039385 ssim 0.8908
                  '2023-11-25-15-26-01_case4_1.61_scale0.5.pth',#1.62+-1.29 ncc 0.9869 jac 0.00055153 ssim 0.8200
                  '2023-11-25-15-26-01_case5_1.95_scale0.5.pth',#2.11+-1.51 ncc 0.9844 jac 0.00021430 ssim 0.8319
                  '2023-11-25-15-26-01_case6_1.69_scale0.5.pth',#1.85+-1.23 ncc 0.9821 jac 0.00004328 ssim 0.7264
                  '2023-11-25-15-26-01_case7_1.66_scale0.5.pth',# 1.65+-0.97 ncc 0.9822 jac 0.00427638 ssim 0.7493
                  '2023-11-25-15-26-01_case8_1.84_scale0.5.pth',#2.01+-1.73 ncc 0.9692 jac 0.00692394 ssim 0.6433
                  '2023-11-25-15-26-01_case9_1.86_scale0.5.pth',#1.81+-0.94 ncc 0.9765 jac 0.00002052 ssim 0.7634
                  '2023-11-25-15-26-01_case10_1.58_scale0.5.pth']#1.61+-1.30 ncc 0.9792 jac 0.00006889 ssim 0.7774
    ckpt_128bak_list = ['',
                     '2023-11-25-15-26-01_case1_1.08_scale0.5.pth',#1.03+-1.02 ncc 0.9927 jac 0.00000000 ssim 0.9019
                     '2023-11-25-15-26-01_case2_1.03_scale0.5.pth',#0.91+-0.99 ncc 0.9924 jac 0.00004028 ssim 0.8936
                     '2023-11-25-15-26-01_case3_1.42_scale0.5.pth',#1.41+-1.19 ncc 0.9915 jac 0.00000487 ssim 0.8845
                     '2023-11-25-15-26-01_case4_1.54_scale0.5.pth',#1.57+-1.29 ncc 0.9864 jac 0.00022269 ssim 0.8147
                     '2023-11-25-15-26-01_case5_1.92_scale0.5.pth',#2.06+-1.53 ncc 0.9847 jac 0.00003058 ssim 0.8356
                     '2023-11-25-15-26-01_case6_1.71_scale0.5.pth',#1.81+-1.19 ncc 0.9818 jac 0.00003167 ssim 0.7241
                     '2023-11-25-15-26-01_case7_1.58_scale0.5.pth',#1.57+-0.93 ncc 0.9827 jac 0.00445687 ssim 0.7517
                     '2023-11-25-15-26-01_case8_1.77_scale0.5.pth',#1.87+-1.51 ncc 0.9707 jac 0.00687485 ssim 0.6504
                     '2023-11-25-15-26-01_case9_1.83_scale0.5.pth',#1.78+-0.92 ncc 0.9775 jac 0.00000000 ssim 0.7686
                     '2023-11-25-15-26-01_case10_1.59_scale0.5.pth']#1.62+-1.33 ncc 0.9797 jac 0.00000678 ssim 0.7767
    ckpt_128noreg_list = ['',
                        'no_reg/2023-11-26-12-28-18_case1_1.07_scale0.5.pth',# 1.09+-0.63 ncc 0.9883 jac 0.00271166 ssim 0.8912
                        'no_reg/2023-11-26-12-28-18_case2_1.03_scale0.5.pth',# 1.02+-0.54 ncc 0.9879 jac 0.00631040 ssim 0.8721
                        'no_reg/2023-11-26-12-28-18_case3_1.41_scale0.5.pth',# 1.45+-0.79 ncc 0.9891 jac 0.00241249 ssim 0.8802
                        'no_reg/2023-11-26-12-28-18_case4_1.80_scale0.5.pth',# 1.77+-1.13 ncc 0.9719 jac 0.00741816 ssim 0.7809
                        'no_reg/2023-11-26-12-28-18_case5_2.25_scale0.5.pth',#2.27+-1.56 ncc 0.9550 jac 0.01021877 ssim 0.7752
                        'no_reg/2023-11-26-12-28-18_case6_1.97_scale0.5.pth',#1.98+-1.07 ncc 0.9346 jac 0.01585233 ssim 0.6599
                        'no_reg/2023-11-26-12-28-18_case7_2.34_scale0.5.pth',#2.33+-1.35 ncc 0.9749 jac 0.00503225 ssim 0.7144
                        'no_reg/2023-11-26-12-28-18_case8_1.98_scale0.5.pth',#1.95+-1.79 ncc 0.9122 jac 0.02999049 ssim 0.6113
                        'no_reg/2023-11-26-12-28-18_case9_2.15_scale0.5.pth',#2.16+-1.27 ncc 0.8986 jac 0.02446661 ssim 0.6842
                        'no_reg/2023-11-26-12-28-18_case10_2.25_scale0.5.pth']  # 2.27+-2.35 ncc 0.8619 jac 0.05846492 ssim

    ckpt_256_list = ['',
                     '2023-11-25-18-10-16_case1_1.17_scale0.5.pth',  # 1.17+-0.64 ncc 0.9893 jac 0.00000000 ssim 0.8828
                     '2023-11-25-18-10-16_case2_1.10_scale0.5.pth',  # 1.06+-0.52 ncc 0.9921 jac 0.00000000 ssim 0.8912
                     '2023-11-25-18-10-16_case3_1.35_scale0.5.pth',  # 1.40+-0.80 ncc 0.9921 jac 0.00000000 ssim 0.8894
                     '2023-11-25-18-10-16_case4_1.55_scale0.5.pth',  # 1.56+-0.93 ncc 0.9865 jac 0.00000000 ssim 0.8193
                     '2023-11-25-18-10-16_case5_1.63_scale0.5.pth',  # 1.60+-1.35 ncc 0.9877 jac 0.00000000 ssim 0.8587
                     '2023-11-25-18-10-16_case6_1.78_scale0.5.pth',  # 1.75+-0.92 ncc 0.9830 jac 0.00000000 ssim 0.7254
                     '2023-11-25-18-10-16_case7_1.64_scale0.5.pth',  # 1.58+-0.93 ncc 0.9829 jac 0.00000000 ssim 0.7501
                     '2023-11-25-18-10-16_case8_2.67_scale0.5.pth',  # 2.86+-3.01 ncc 0.9605 jac 0.00024556 ssim 0.6080
                     '2023-11-25-18-10-16_case9_1.43_scale0.5.pth',  # 1.42+-0.77 ncc 0.9840 jac 0.00000000 ssim 0.7916
                     '2023-11-25-18-10-16_case10_1.56_scale0.5.pth']  # 1.55+-1.26 ncc 0.9803 jac 0.00000000 ssim 0.7813
    ckpt_256_list_reg = ['',
                         '2023-11-25-18-10-16_case1_1.04_scale0.5.pth',
                         # 1.07+-0.53 ncc 0.9927 jac 0.00000000 ssim 0.9012
                         '2023-11-25-18-10-16_case2_1.04_scale0.5.pth',
                         # 1.03+-0.51 ncc 0.9926 jac 0.00000000 ssim 0.8954
                         '2023-11-25-18-10-16_case3_1.25_scale0.5.pth',
                         # 1.25+-0.71 ncc 0.9931 jac 0.00000000 ssim 0.8966
                         '2023-11-25-18-10-16_case4_1.41_scale0.5.pth',
                         # 1.43+-0.92 ncc 0.9879 jac 0.00000000 ssim 0.8270
                         '2023-11-25-18-10-16_case5_1.61_scale0.5.pth',
                         # 1.65+-1.43 ncc 0.9880 jac 0.00000000 ssim 0.8570
                         '2023-11-25-18-10-16_case6_1.75_scale0.5.pth',
                         # 1.76+-0.91 ncc 0.9825 jac 0.00000000 ssim 0.7238
                         '2023-11-25-18-10-16_case7_1.52_scale0.5.pth',
                         # 1.54+-0.88 ncc 0.9836 jac 0.00000000 ssim 0.7541
                         '2023-11-25-18-10-16_case8_1.91_scale0.5.pth',
                         # 1.84+-1.47 ncc 0.9691 jac 0.00054933 ssim 0.6482
                         '2023-11-25-18-10-16_case9_1.40_scale0.5.pth',
                         # 1.45+-0.78 ncc 0.9840 jac 0.00000000 ssim 0.7924
                         '2023-11-25-18-10-16_case10_1.41_scale0.5.pth']  # 1.46+-1.15 ncc 0.9813 jac 0.00000000 ssim 0.7870
    ckpt_256_list_noreg = ['',
                           'noreg/2023-11-26-13-50-57_case1_1.06_scale0.5.pth',
                           # 1.06+-0.64 ncc 0.9917 jac 0.00162912 ssim 0.8956
                           'noreg/2023-11-26-13-50-57_case2_1.18_scale0.5.pth',
                           # 1.26+-0.76 ncc 0.9879 jac 0.00306258 ssim 0.8602
                           'noreg/2023-11-26-13-50-57_case3_1.45_scale0.5.pth',
                           # 1.43+-0.78 ncc 0.9862 jac 0.01028766 ssim 0.8602
                           'noreg/2023-11-26-13-50-57_case4_1.85_scale0.5.pth',
                           # 1.85+-1.09 ncc 0.9349 jac 0.01736633 ssim 0.7454
                           'noreg/2023-11-26-13-50-57_case5_2.19_scale0.5.pth',
                           # 2.27+-1.51 ncc 0.9765 jac 0.00394123 ssim 0.7998
                           'noreg/2023-11-26-13-50-57_case6_1.98_scale0.5.pth',
                           # 1.98+-1.26 ncc 0.9516 jac 0.01978437 ssim 0.6584
                           'noreg/2023-11-26-13-50-57_case7_2.34_scale0.5.pth',
                           # 2.37+-1.34 ncc 0.9766 jac 0.00267445 ssim 0.7183
                           'noreg/2023-11-26-13-50-57_case8_2.76_scale0.5.pth',
                           # diff: 2.57+-3.03 ncc 0.9226 jac 0.02013422 ssim 0.5883
                           'noreg/2023-11-26-13-50-57_case9_1.86_scale0.5.pth',
                           # 1.95+-1.03 ncc 0.9567 jac 0.01056264 ssim 0.7371
                           'noreg/2023-11-26-13-50-57_case10_1.98_scale0.5.pth']  # 2.01+-1.82 ncc 0.9695 jac 0.00118729 ssim 0.7480

    mean_tre_list = []
    mean_std_list = []
    ncc_list = []
    jac_list = []
    ssim_list = []
    for i in range(1, 11):
        mean_tre,mean_std, ncc, jac, ssim = test(args, i, False, ckpt_64_list_noreg[i])
        mean_tre_list.append(mean_tre)
        mean_std_list.append(mean_std)
        ncc_list.append(ncc)
        jac_list.append(jac)
        ssim_list.append(ssim)

    tre = np.array(mean_tre_list).sum()/10
    std = np.array(mean_std_list).sum() / 10
    ncc = np.array(ncc_list).sum() / 10
    jac = np.array(jac_list).sum() / 10
    ssim = np.array(ssim_list).sum() / 10

    print(f'\n mean diff: {tre:.2f}+-{std:.2f} ncc {ncc:.4f} jac {jac:.8f} ssim {ssim:.4f}')