import time

import GDIR.model.regnet, GDIR.model.loss, GDIR.model.util, utils.structure, utils.utilize
import torch, os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from GDIR.process.processing import data_standardization_0_n, imgTomhd, data_standardization_min_max
import random
from utils.metric import SSIM, NCC as mtNCC, jacobian_determinant

plot_dpi = 300
import numpy as np
import logging, tqdm
from scipy import interpolate
from utils.utilize import save_image, get_project_path, make_dir, count_parameters,make_dirs
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
    torch.backends.cudnn.deterministic = True


config = dict(
    dim=3,  # dimension of the input image
    scale=0.3,
    initial_channels=32,
    depth=4,
    max_num_iteration=300,
    normalization=True,  # whether use normalization layer
    learning_rate=1e-2,
    smooth_reg=1e-3,
    cyclic_reg=1e-2,
    ncc_window_size=5,
    load=None,
    load_optimizer=False,
    group_index_list=None,
    pair_disp_indexes=[0, 5],
    pair_disp_calc_interval=10,
    stop_std=0.0007,
    stop_query_len=200,
)
config = utils.structure.Struct(**config)
project_path = get_project_path("4DCT")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
spatial_transform = SpatialTransformer(3)

def train(case=1):
    args = get_args()
    # logger.add('{time:YYYY-MM-DD HHmmss}.log', format="{message}", rotation='5 MB', encoding='utf-8')
    set_seed()
    states_folder = args.checkpoint_path
    make_dirs(args)
    # log_folder = os.path.join('log/', f'case{case}')
    # 保存固定图像和扭曲图像路径
    # warp_case_path = os.path.join("../result/general_reg/dirlab/warped_image", f"Case{case}")
    # temp_case_path = os.path.join("../result/general_reg/dirlab/template_image", f"Case{case}")
    # make_dir(warp_case_path)
    # make_dir(temp_case_path)
    # make_dir(log_folder)

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

    # crop
    # input_image = input_image[:, :, crop_range0, crop_range1, crop_range2]

    image_shape = np.array(input_image.shape[2:])  # (d, h, w) z y x
    num_image = input_image.shape[0]  # number of image in the group
    regnet = GDIR.model.regnet.RegNet_single(dim=config.dim, n=num_image, scale=config.scale, depth=config.depth,
                                             initial_channels=config.initial_channels,
                                             normalization=config.normalization)
    # 使用tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # test_images = torch.randn(10, 1, 94, 256, 256)
    # writer = SummaryWriter('runs/GDIR')
    # writer.add_graph(regnet, test_images,use_strict_trace=False)
    # writer.close()
    #
    # print("模型参数：", count_parameters(regnet.unet))

    # n = 5
    ncc_loss = GDIR.model.loss.NCC(config.dim, config.ncc_window_size)
    regnet = regnet.to(device)
    input_image = input_image.to(device)
    ncc_loss = ncc_loss.to(device)
    optimizer = torch.optim.Adam(regnet.parameters(), lr=config.learning_rate)
    # optimizer = torch.optim.SGD(regnet.parameters(), lr=config.learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=5e-5)
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
            simi_loss = (ncc_loss(res['warped_input_image'], res['template']) + ncc_loss(input_image,
                                                                                         res[
                                                                                             'warped_template'])) / 2.
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

            mean, std, diff, compse_disp = calc_tre(calcdisp, disp_i2t, res['disp_t2i'][config.pair_disp_indexes],
                                          grid_tuple, landmark_00_converted, landmark_disp,
                                          args.dirlab_cfg[case]['pixel_spacing'])
            diff_stats.append([i, mean, std])
            # save_warp(args, np.expand_dims(compse_disp[0, 1], 0), input_image[0].unsqueeze(0), 'case1', 'gdir', spacing=args.dirlab_cfg[case]['pixel_spacing'])

            print(f'\n diff: {mean:.2f}+-{std:.2f}({np.max(diff):.2f})')

            if mean < best_tre:
                best_tre = mean
                states = {'config': config, 'model': regnet.state_dict(), 'optimizer': None,
                          'loss_list': stop_criterion.loss_list, 'diff_stats': diff_stats}
                states_file = f'reg_dirlab_case{case}_{mean:.2f}({std:.2f}).pth'
                save_path = os.path.join(states_folder, states_file)
                torch.save(states, save_path)
                print(f'save model {states_file} in path:{save_path}')


def test(case=1, is_save=False):
    args = get_args()
    set_seed()
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
                                             normalization=config.normalization)

    ncc_loss = GDIR.model.loss.NCC(config.dim, config.ncc_window_size)
    regnet = regnet.to(device)
    input_image = input_image.to(device)
    ncc_loss = ncc_loss.to(device)
    optimizer = torch.optim.Adam(regnet.parameters(), lr=config.learning_rate)
    calcdisp = GDIR.model.util.CalcDisp(dim=config.dim, calc_device='cuda')


    state_file = os.path.join(states_folder, config.load)
    if os.path.exists(state_file):
        state_file = os.path.join(states_folder, config.load)
        states = torch.load(state_file, map_location=device)
        regnet.load_state_dict(states['model'])
        if config.load_optimizer:
            optimizer.load_state_dict(states['optimizer'])

    grid_tuple = [np.arange(grid_length, dtype=np.float32) for grid_length in image_shape]
    diff_stats = []

    diff_ori = (np.sum((landmark_disp * args.dirlab_cfg[case]['pixel_spacing']) ** 2, 1)) ** 0.5
    print("\ncase{0}配准前 diff: {1}({2})".format(case, np.mean(diff_ori), np.std(diff_ori)))


    res = regnet(input_image)
    if 'disp_i2t' in res:
        disp_i2t = res['disp_i2t'][config.pair_disp_indexes]
    else:
        disp_i2t = calcdisp.inverse_disp(res['disp_t2i'][config.pair_disp_indexes])
    mean, std, diff, compse_disp = calc_tre(calcdisp, disp_i2t, res['disp_t2i'][config.pair_disp_indexes],
                                            grid_tuple, landmark_00_converted, landmark_disp,
                                            args.dirlab_cfg[case]['pixel_spacing'])

    disp, warped = get_flow50_00(args, res, input_image[0:1],spacing=args.dirlab_cfg[case]['pixel_spacing'],is_save=is_save)

    ncc = mtNCC(warped, input_image[5:6].squeeze().cpu().detach().numpy())
    jac = jacobian_determinant(disp[0].cpu().detach().numpy())
    ssim = SSIM(warped, input_image[5][0].cpu().detach().numpy())
    print(f'\n diff: {mean:.2f}+-{std:.2f}({np.max(diff):.2f}) ncc {ncc:.4f} jac {jac:.8f} ssim {ssim:.4f}')



def get_flow50_00(args, res, input_image, spacing, is_save=False):
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

    wapred_img = get_warp(args, disp50_00, input_image, 'case1', 'gdir', spacing, is_save)
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
    # train(1)
    test(1, True)