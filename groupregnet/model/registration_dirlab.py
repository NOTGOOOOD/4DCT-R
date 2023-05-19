import torch
import os
import SimpleITK as sitk
from groupregnet.model import structure
import regnet
import loss
import util
from utils.config import get_args
from utils.utilize import set_seed
set_seed(1024)
plot_dpi = 300
import numpy as np
from scipy import interpolate
from utils.utilize import make_dir, get_project_path
from utils.utilize import show_slice
from utils.metric import NCC, MSE, SSIM, jacobian_determinant
from utils.losses import neg_Jdet_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# landmark_file = f'/data/dirlab/Case1Pack/ExtremePhases/case{case}_00_50.pt'
states_folder = '../result/general_reg/'
make_dir(states_folder)
# data_folder = r'D:\xxf\patient\008'
config = dict(
    dim=3,  # dimension of the input image
    intensity_scale_const=1000.,  # (image - intensity_shift_const)/intensity_scale_const
    intensity_shift_const=1000.,
    # scale = 0.7,
    scale=0.5,
    initial_channels=32,
    depth=4,
    max_num_iteration=30000,
    normalization=True,  # whether use normalization layer
    learning_rate=1e-4,
    smooth_reg=1e-3,
    cyclic_reg=1e-2,
    ncc_window_size=5,
    load=False,
    load_optimizer=False,
    group_index_list=None,
    pair_disp_indexes=[0, 5],
    pair_disp_calc_interval=20,
    stop_std=0.0007,
    stop_query_len=100,
)
config = structure.Struct(**config)

case = 1

project_path = get_project_path("4DCT-R")

args = get_args()

# d,h,w
crop_range0 = args.dirlab_cfg[case]["crop_range"][0]
crop_range1 = args.dirlab_cfg[case]["crop_range"][1]
crop_range2 = args.dirlab_cfg[case]["crop_range"][2]
crop_range0_start = crop_range0.start
crop_range1_start = crop_range1.start
crop_range2_start = crop_range2.start

# landmark
landmark_file = os.path.join(project_path, f'data/dirlab/Case0{case}_300_00_50.pt')
landmark_info = torch.load(landmark_file)
landmark_disp = landmark_info['disp_00_50']  # w, h, d  x,y,z
landmark_00 = landmark_info['landmark_00']
landmark_50 = landmark_info['landmark_50']

landmark_00_converted = np.flip(landmark_00, axis=1) - np.array(
    [crop_range0_start, crop_range1_start, crop_range2_start], dtype=np.float32)

landmark_50_converted = np.flip(landmark_50, axis=1) - np.array(
    [crop_range0_start, crop_range1_start, crop_range2_start], dtype=np.float32)

# file
data_folder = 'D:/xxf/dirlabcase1-10/case%02d' % case
image_file_list = sorted([file_name for file_name in os.listdir(data_folder) if file_name.lower().endswith('.gz')])
image_list = []
for file_name in image_file_list:
    # xyz W H D
    img_sitk = sitk.ReadImage(os.path.join(data_folder, file_name))
    # zyx D H W
    stkimg = sitk.GetArrayFromImage(img_sitk)

    image_list.append(stkimg)

input_image = torch.stack([torch.from_numpy(image)[None] for image in image_list], 0)

if config.group_index_list is not None:
    input_image = input_image[config.group_index_list]

input_image = (input_image - config.intensity_shift_const) / config.intensity_scale_const

image_shape = np.array(input_image.shape[2:])  # (d, h, w)
num_image = input_image.shape[0]  # number of image in the group
regnet = regnet.RegNet_single(dim=config.dim, n=num_image, scale=config.scale, depth=config.depth,
                              initial_channels=config.initial_channels, normalization=config.normalization)

# regnet = regnet.(dim=config.dim, n=num_image, scale=config.scale, depth=config.depth,
#                               initial_channels=config.initial_channels, normalization=config.normalization)

ncc_loss = loss.NCC(config.dim, config.ncc_window_size)
regnet = regnet.to(device)
input_image = input_image.to(device)
ncc_loss = ncc_loss.to(device)
optimizer = torch.optim.Adam(regnet.parameters(), lr=config.learning_rate)
calcdisp = util.CalcDisp(dim=config.dim, calc_device='cuda')

if config.load:
    state_file = os.path.join(states_folder, config.load)
    if os.path.exists(state_file):
        state_file = os.path.join(states_folder, config.load)
        states = torch.load(state_file, map_location=device)
        regnet.load_state_dict(states['model'])
        if config.load_optimizer:
            optimizer.load_state_dict(states['optimizer'])
            print(f'load model and optimizer state {config.load}.pth')
        else:
            print(f'load model state {config.load}.pth')

grid_tuple = [np.arange(grid_length, dtype=np.float32) for grid_length in image_shape]

# stop_criterion = util.StopCriterion(stop_std=config.stop_std, query_len=config.stop_query_len)
# from utils.scheduler import StopCriterion
# stop_criterion = StopCriterion(patient_len=100, min_epoch=10)

import tqdm

diff_stats = []
stop_criterion = util.StopCriterion(stop_std=config.stop_std, query_len=config.stop_query_len)

pbar = tqdm.tqdm(range(config.max_num_iteration))
for i in pbar:
    optimizer.zero_grad()
    res = regnet(input_image)

    total_loss = 0.
    if 'disp_i2t' in res:
        simi_loss = (ncc_loss(res['warped_input_image'], res['template']) + ncc_loss(input_image,
                                                                                     res['warped_template'])) / 2.
    else:
        simi_loss = ncc_loss(res['warped_input_image'], res['template'])
    total_loss += simi_loss

    if config.smooth_reg > 0:
        if 'disp_i2t' in res:
            smooth_loss = (loss.smooth_loss(res['scaled_disp_t2i']) + loss.smooth_loss(
                res['scaled_disp_i2t'])) / 2.
        else:
            # smooth_loss = model.loss.smooth_loss(res['scaled_disp_t2i'])
            smooth_loss = loss.smooth_loss(res['scaled_disp_t2i'], res['scaled_template'])
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

    total_loss.backward()
    optimizer.step()

    stop_criterion.add(simi_loss.item())
    if stop_criterion.stop():
        break

    pbar.set_description(
        f'{i}, total loss {total_loss.item():.4f}, simi. loss {simi_loss.item():.4f}, smooth loss {smooth_loss_item:.3f}, cyclic loss {cyclic_loss_item:.3f}')

    if i % config.pair_disp_calc_interval == 0:
        if 'disp_i2t' in res:
            disp_i2t = res['disp_i2t'][config.pair_disp_indexes]
        else:
            disp_i2t = calcdisp.inverse_disp(res['disp_t2i'][config.pair_disp_indexes])
        composed_disp = calcdisp.compose_disp(disp_i2t, res['disp_t2i'][config.pair_disp_indexes], mode='all')
        composed_disp_np = composed_disp.cpu().numpy()  # (2, 2, 3, d, h, w)

        inter = interpolate.RegularGridInterpolator(grid_tuple, np.moveaxis(composed_disp_np[0, 1], 0, -1))
        calc_landmark_disp = inter(landmark_00_converted)

        diff = (np.sum(((calc_landmark_disp - landmark_disp) * args.dirlab_cfg[case]['pixel_spacing']) ** 2, 1)) ** 0.5
        diff_stats.append([i, np.mean(diff), np.std(diff)])
        print(f'\ndiff: {np.mean(diff):.2f}+-{np.std(diff):.2f}({np.max(diff):.2f})')

if 'disp_i2t' in res:
    disp_i2t = res['disp_i2t'][config.pair_disp_indexes]
else:
    disp_i2t = calcdisp.inverse_disp(res['disp_t2i'][config.pair_disp_indexes])

composed_disp = calcdisp.compose_disp(disp_i2t, res['disp_t2i'][config.pair_disp_indexes], mode='all')
composed_disp_np = composed_disp.cpu().numpy()  # (2, 2, 3, d, h, w)
inter = interpolate.RegularGridInterpolator(grid_tuple, np.moveaxis(composed_disp_np[0, 1], 0, -1))
calc_landmark_disp = inter(landmark_00_converted)

diff = (np.sum(((calc_landmark_disp - landmark_disp) * args.dirlab_cfg[case]['pixel_spacing']) ** 2, 1)) ** 0.5
mean_tre = np.mean(diff)
mean_std = np.mean(diff)
diff_stats.append([i, np.mean(diff), np.std(diff)])
print(f'\ndiff: {np.mean(mean_tre):.2f}+-{np.std(mean_std):.2f}({np.max(diff):.2f})')
diff_stats = np.array(diff_stats)

template = res['template']
# show_slice(template.cpu().detach().numpy())

wapred_moving = res['warped_input_image'][0:1, ...]
disp = res['disp_t2i'][0:1, ...]
img_shape = wapred_moving.shape[2:]

ncc = NCC(template.cpu().detach().numpy(), wapred_moving.cpu().detach().numpy())

ja = jacobian_determinant(disp.cpu().detach().numpy())
# MSE
mse = MSE(wapred_moving, template)
# SSIM
ssim = SSIM(template.cpu().detach().numpy()[0, 0], wapred_moving.cpu().detach().numpy()[0, 0])

print('TRE= %.2f+-.2f, MSE=%.5f Jac=%.6f, SSIM=%.5f, NCC=%.5f' % (
    mean_tre, mean_std, mse.item(), ja, ssim.item(), ncc.item()))

states = {'config': config, 'model': regnet.state_dict(), 'optimizer': optimizer.state_dict(),
          'loss_list': stop_criterion.loss_list, 'diff_stats': diff_stats}
index = len([file for file in os.listdir(states_folder) if file.endswith('pth')])
states_file = f'reg_dirlab_case{case}_{index:03d}_{mean_tre:.2f}({mean_std:.2f}).pth'
torch.save(states, os.path.join(states_folder, states_file))

# plt.figure(dpi = plot_dpi)
# plt.plot(stop_criterion.loss_list, label = 'simi')
# plt.title('similarity loss vs iteration')
