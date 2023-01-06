import torch
import numpy as np
import os
from scipy import interpolate

from utils.utilize import get_project_path


def NCC(real, predict):
    real_copy = np.copy(real)
    predict_copy = np.copy(predict)
    return np.mean(np.multiply((real_copy - np.mean(real_copy)), (predict_copy - np.mean(predict_copy)))) / (
            np.std(real_copy) * np.std(predict_copy))


def MSE(real_copy, predict_copy):
    if torch.is_tensor(real_copy):
    # return mean_squared_error(real_copy, predict_copy)
        real_copy = real_copy.cuda()
        predict_copy = predict_copy.cuda()
        return torch.mean(torch.square(predict_copy - real_copy))
    else:
        return np.mean(np.square(predict_copy - real_copy))


def calc_dirlab(cfg):
    """
    计算所有dirlab 配准前的位移
    Parameters
    ----------
    cfg

    Returns
    -------

    """
    project_path = get_project_path("4DCT-R")
    diff = [], landmark00 = []
    for case in range(1, 11):
        landmark_file = os.path.join(project_path, f'data/dirlab/Case{case}_300_00_50.pt')
        landmark_info = torch.load(landmark_file)
        landmark_disp = landmark_info['disp_00_50']  # w, h, d  x,y,z
        landmark_00 = landmark_info['landmark_00']
        # landmark_50 = landmark_info['landmark_50']

        diff_ori = (np.sum((landmark_disp * cfg[case]['pixel_spacing']) ** 2, 1)) ** 0.5

        diff[case].append(np.mean(diff_ori), np.std(diff_ori))
        landmark00.append(landmark_00)

    return diff, landmark00


def calc_tre(disp_t2i, landmark_00_converted, landmark_disp, spacing):
    # x' = u(x) + x
    disp = np.array(disp_t2i.cpu())
    landmark_disp = np.array(landmark_disp.cpu())
    # convert -> z,y,x
    landmark_00_converted = np.array(landmark_00_converted[0].cpu())
    landmark_00_converted = np.flip(landmark_00_converted, axis=1)

    image_shape = disp.shape[1:]
    grid_tuple = [np.arange(grid_length, dtype=np.float32) for grid_length in image_shape]
    inter = interpolate.RegularGridInterpolator(grid_tuple, np.moveaxis(disp, 0, -1))
    calc_landmark_disp = inter(landmark_00_converted)

    diff = (np.sum(((calc_landmark_disp - landmark_disp) * spacing) ** 2, 1)) ** 0.5
    diff = diff[~np.isnan(diff)]

    return np.mean(diff), np.std(diff)


def landmark_loss(flow, m_landmarks, f_landmarks, spacing):
    # flow + fixed - moving
    spec = torch.tensor(spacing).cuda()

    all_dist = []
    # zz, yy, xx = flow[0].shape
    # flow[2, :, :, :] = flow[2, :, :, :] * (zz - 1)/2
    # flow[1, :, :, :] = flow[1, :, :, :] * (yy - 1)/2
    # flow[0, :, :, :] = flow[0, :, :, :] * (xx - 1)/2
    for i in range(300):
        # point before warped
        f_point = f_landmarks[i].int()
        # m_point = m_landmarks[i].int()

        # point at flow
        move = flow[:, f_point[2], f_point[1], f_point[0]]
        # point after warped
        ori_point = torch.round(f_point + move)
        dist = ori_point - m_landmarks[i]
        all_dist.append(dist * spec)

    all_dist = torch.stack(all_dist)
    pt_errs_phys = torch.sqrt(torch.sum(all_dist * all_dist, 1))

    return torch.mean(pt_errs_phys), torch.std(pt_errs_phys)

    #     ref_lmk = landmark_50.copy()
    #     for i in range(300):
    #         wi, hi, di = landmark_50[i]
    #         w0, h0, d0 = flow[:, di, hi, wi]
    #         ref_lmk[i] = ref_lmk[i] + [w0, h0, d0]
    #
    #     tre = torch.tensor(ref_lmk - landmark_00).pow(2).sum(1).sqrt()
    #     tre_mean = tre.mean()
    #     tre_std = tre.std()
    #     print("%.2f+-%.2f" % (tre_mean, tre_std))


def get_test_photo_loss(args, logger, model, test_loader):
    with torch.no_grad():
        model.eval()
        losses = []
        for batch, (moving, fixed, landmarks) in enumerate(test_loader):
            m_img = moving[0].to('cuda').float()
            f_img = fixed[0].to('cuda').float()

            landmarks00 = landmarks['landmark_00'].squeeze().cuda()
            # landmarks50 = landmarks['landmark_50'].squeeze().cuda()

            warped_image, flow = model(m_img, f_img, True)
            # warped_image, flow = model(m_img, f_img)
            flow_hr = flow[0]
            index = batch + 1

            crop_range = args.dirlab_cfg[index]['crop_range']

            # TRE
            _mean, _std = calc_tre(flow_hr, landmarks00 - torch.tensor(
                [crop_range[2].start, crop_range[1].start, crop_range[0].start]).view(1, 1, 3).cuda(),
                                   landmarks['disp_00_50'].squeeze(), args.dirlab_cfg[index]['pixel_spacing'])

            # MSE
            _mse = MSE(f_img, warped_image)

            losses.append([_mean.item(), _std.item(), _mse.item()])

            logger.info('case=%d after warped, TRE=%.5f+-%.5f' % (index, _mean.item(), _std.item()))

        return losses
