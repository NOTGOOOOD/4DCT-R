import GDIR.model.regnet, GDIR.model.loss, GDIR.model.util, utils.structure, utils.utilize
import torch, os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from process.processing import data_standardization_0_n

plot_dpi = 300
import numpy as np
import logging, tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from scipy import interpolate
from utils.utilize import tre, save_image, get_project_path, make_dir


def calc_tre(pair_disp_indexes):
    # x' = u(x) + x
    composed_disp = calcdisp.compose_disp(disp_i2t, res['disp_t2i'][pair_disp_indexes], mode='all')
    composed_disp_np = composed_disp.cpu().numpy()  # (2, 2, 3, d, h, w)

    inter = interpolate.RegularGridInterpolator(grid_tuple, np.moveaxis(composed_disp_np[0, 1], 0, -1))
    calc_landmark_disp = inter(landmark_00_converted)

    diff = (np.sum(((calc_landmark_disp - landmark_disp) * cfg[case]["pixel_spacing"]) ** 2, 1)) ** 0.5
    diff = diff[~np.isnan(diff)]

    return np.mean(diff), np.std(diff), diff, composed_disp_np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# z y x
cfg = [{},
       {"case": 1,
        "crop_range": [slice(0, 83), slice(43, 200), slice(10, 250)],
        "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32)
        },
       {"case": 2,
        "crop_range": [slice(5, 98), slice(30, 195), slice(8, 243)],
        "pixel_spacing": np.array([1.16, 1.16, 2.5], dtype=np.float32)
        },
       {"case": 3,
        "crop_range": [slice(0, 95), slice(42, 209), slice(10, 248)],
        "pixel_spacing": np.array([1.15, 1.15, 2.5], dtype=np.float32)
        },
       {"case": 4,
        "crop_range": [slice(0, 90), slice(45, 209), slice(11, 242)],
        "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32)
        },
       {"case": 5,
        "crop_range": [slice(0, 90), slice(60, 222), slice(16, 237)],
        "pixel_spacing": np.array([1.10, 1.10, 2.5], dtype=np.float32)
        },
       {"case": 6,
        "crop_range": [slice(10, 107), slice(144, 328), slice(132, 426)],
        "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32)
        },
       {"case": 7,
        "crop_range": [slice(13, 108), slice(141, 331), slice(114, 423)],
        "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32)
        },
       {"case": 8,
        "crop_range": [slice(18, 118), slice(84, 299), slice(113, 390)],
        "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32)
        },
       {"case": 9,
        "crop_range": [slice(0, 70), slice(126, 334), slice(128, 390)],
        "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32)
        },
       {"case": 10,
        "crop_range": [slice(0, 90), slice(119, 333), slice(140, 382)],
        "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32)
        }

       ]

case = 1
project_path = get_project_path("4DCT")
data_folder = os.path.join(project_path.split("4DCT")[0], f'datasets/dirlab/mhd/Case{case}_mhd/')
landmark_file = os.path.join(project_path, f'data/dirlab/Case{case}_300_00_50.pt')
states_folder = os.path.join(project_path, f'result/general_reg/dirlab/')

# 保存固定图像和扭曲图像路径
warp_case_path = os.path.join("../result/general_reg/dirlab/warped_image", f"Case{case}")
temp_case_path = os.path.join("../result/general_reg/dirlab/template_image", f"Case{case}")
dvf_path = os.path.join("../result/general_reg/dirlab/dvf", f"Case{case}")
make_dir(warp_case_path)
make_dir(temp_case_path)
make_dir(dvf_path)

config = dict(
    dim=3,  # dimension of the input image
    scale=0.5,
    initial_channels=32,
    depth=4,
    max_num_iteration=300,
    normalization=True,  # whether use normalization layer
    learning_rate=0.001,
    smooth_reg=1e-3,
    cyclic_reg=1e-2,
    ncc_window_size=5,
    load=False,
    load_optimizer=False,
    group_index_list=None,
    pair_disp_indexes=[0, 5],
    pair_disp_calc_interval=20,
    stop_std=0.0001,
    stop_query_len=300,
)
config = utils.structure.Struct(**config)

image_file_list = sorted([file_name for file_name in os.listdir(data_folder) if file_name.lower().endswith('mhd')])
image_list = []
for file_name in image_file_list:
    # xyz W H D
    img_sitk = sitk.ReadImage(os.path.join(data_folder, file_name))
    # zyx D H W
    stkimg = sitk.GetArrayFromImage(img_sitk)
    image_list.append(stkimg)

# numpy D,H,W  zyx
# simpleitk W,H,D  xyz
# 1(number of group),B,D,H,W
input_image = torch.stack([torch.from_numpy(image)[None] for image in image_list], 0)
if config.group_index_list is not None:
    input_image = input_image[config.group_index_list]

# normalize [0,1]
input_image = data_standardization_0_n(1, input_image)

# crop
crop_range0 = cfg[case]["crop_range"][0]
crop_range1 = cfg[case]["crop_range"][1]
crop_range2 = cfg[case]["crop_range"][2]
try:
    crop_range0_start = cfg[case]["crop_range"][0].start
except:
    crop_range0_start = cfg[case]["crop_range"][0][0].start

crop_range1_start = cfg[case]["crop_range"][1].start

try:
    crop_range2_start = cfg[case]["crop_range"][2].start
except:
    crop_range2_start = cfg[case]["crop_range"][2][0].start

input_image = input_image[:, :, crop_range0, crop_range1, crop_range2]

image_shape = np.array(input_image.shape[2:])  # (d, h, w) z y x
num_image = input_image.shape[0]  # number of image in the group
regnet = GDIR.model.regnet.RegNet_single(dim=config.dim, n=num_image, scale=config.scale, depth=config.depth,
                                         initial_channels=config.initial_channels, normalization=config.normalization)

# n = 5
ncc_loss = GDIR.model.loss.NCC(config.dim, config.ncc_window_size)
regnet = regnet.to(device)
input_image = input_image.to(device)
ncc_loss = ncc_loss.to(device)
optimizer = torch.optim.Adam(regnet.parameters(), lr=config.learning_rate)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0, verbose=True)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=5e-7)
calcdisp = GDIR.model.util.CalcDisp(dim=config.dim, calc_device='cuda')

if config.load:
    state_file = os.path.join(states_folder, config.load)
    if os.path.exists(state_file):
        state_file = os.path.join(states_folder, config.load)
        states = torch.load(state_file, map_location=device)
        regnet.load_state_dict(states['model'])
        if config.load_optimizer:
            optimizer.load_state_dict(states['optimizer'])
            logging.info(f'load model and optimizer state {config.load}.pth')
        else:
            logging.info(f'load model state {config.load}.pth')

grid_tuple = [np.arange(grid_length, dtype=np.float32) for grid_length in image_shape]
diff_stats = []
stop_criterion = GDIR.model.util.StopCriterion(stop_std=config.stop_std, query_len=config.stop_query_len)
pbar = tqdm.tqdm(range(config.max_num_iteration))

# landmark
landmark_info = torch.load(landmark_file)
landmark_disp = landmark_info['disp_00_50']  # w, h, d  x,y,z
landmark_00 = landmark_info['landmark_00']
# d,h,w
landmark_00_converted = np.flip(landmark_00, axis=1) - np.array(
    [crop_range0_start, crop_range1_start, crop_range2_start], dtype=np.float32)

for i in pbar:
    res = regnet(input_image)
    # # 保存前六个阶段图片
    # if i % 10 == 0:
    #     for j in (0, 5):
    #         utils.utilize.plotorsave_ct_scan(res['warped_input_image'][j, 0, :, :, :], "save",
    #                                          epoch=i,
    #                                          head="warped",
    #                                          case=case,
    #                                          phase=j * 10,
    #                                          path=f"../result/general_reg/dirlab/warped_image")
    #
    #     utils.utilize.plotorsave_ct_scan(res['template'][0, 0, :, :, :], "save",
    #                                      epoch=i,
    #                                      head="tem",
    #                                      case=case,
    #                                      phase=50,
    #                                      path=f"../result/general_reg/dirlab/template_image")

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
    # scheduler.step()

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

        mean, std, diff, _ = calc_tre(config.pair_disp_indexes)
        diff_stats.append([i, mean, std])
        print(f'\ndiff: {mean:.2f}+-{std:.2f}({np.max(diff):.2f})')

        # Save images
        phase = 0
        warped_name = str(i) + f"_case{case}_T{phase}0_warped.nii.gz"
        save_image(res['warped_input_image'][phase, 0, :, :, :], input_image[5],
                   warp_case_path + f'/epoch{i}', warped_name)

        m2f_name = f"case{case}_temp.nii.gz"
        save_image(res['template'][0, 0, :, :, :], input_image[5], temp_case_path + f'/epoch{i}', m2f_name)

        # Save DVF
        # n,3,d,h,w-> w,h,d,3
        save_image(torch.permute(disp_i2t[0], (3, 2, 1, 0)), input_image[5], dvf_path + f'/epoch{i}',
                   f'case{case}dvf.nii')

if 'disp_i2t' in res:
    disp_i2t = res['disp_i2t'][config.pair_disp_indexes]
else:
    disp_i2t = calcdisp.inverse_disp(res['disp_t2i'][config.pair_disp_indexes])

mean, std, diff, composed_dis_np = calc_tre(config.pair_disp_indexes)
diff_stats.append([i, mean, std])
print(f'\ndiff: {mean:.2f}+-{std:.2f}({np.max(diff):.2f})')
diff_stats = np.array(diff_stats)

res['composed_disp_np'] = composed_dis_np
states = {'config': config, 'model': regnet.state_dict(), 'optimizer': optimizer.state_dict(),
          'registration_result': res, 'loss_list': stop_criterion.loss_list, 'diff_stats': diff_stats}
index = len([file for file in os.listdir(states_folder) if file.endswith('pth')])
states_file = f'reg_dirlab_case{case}_{index:03d}_{mean:.2f}({std:.2f}).pth'
torch.save(states, os.path.join(states_folder, states_file))

logging.info(f'save model and optimizer state {states_file}')

plt.figure(dpi=plot_dpi)
plt.plot(stop_criterion.loss_list, label='simi')
plt.title('similarity loss vs iteration')

plt.show()
