import os
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import torch

from utils.utilize import plotorsave_ct_scan, get_project_path, make_dir

# import ants

# DIRLAB 4DCT-R 1-10例的 z y x
case_cfg = {
    1: (94, 256, 256),
    2: (112, 256, 256),
    3: (104, 256, 256),
    4: (99, 256, 256),
    5: (106, 256, 256),
    6: (128, 512, 512),
    7: (136, 512, 512),
    8: (128, 512, 512),
    9: (128, 512, 512),
    10: (120, 512, 512),
}

# 重采样后
"""
Case1 [256, 256, 38]

Case2 [256, 256, 45]

Case3 [256, 256, 42]

Case4 [256, 256, 40]

Case5 [256, 256, 42]
 
Case6 [512, 512, 51]

Case7 [512, 512, 54]

Case8 [512, 512, 51]
 
Case9 [512, 512, 51]
 
Case10 [512, 512, 48]

"""


def window(img):
    win_min = -400
    win_max = 1500

    for i in range(img.shape[0]):
        img[i] = 255.0 * (img[i] - win_min) / (win_max - win_min)
        min_index = img[i] < 0
        img[i][min_index] = 0
        max_index = img[i] > 255
        img[i][max_index] = 255
        img[i] = img[i] - img[i].min()
        c = float(255) / img[i].max()
        img[i] = img[i] * c

    return img.astype(np.uint8)


def set_window(img_data, win_width, win_center):
    img_temp = img_data
    min = (2 * win_center - win_width) / 2.0 + 0.5
    max = (2 * win_center + win_width) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    img_temp = ((img_temp - min) * dFactor).astype(np.int)
    img_temp = np.clip(img_temp, 0, 255)
    return img_temp


def imgTomhd(file_folder, save_path, datatype, shape, case, crop_range=None, resample=False):
    for file_name in os.listdir(file_folder):
        file_path = os.path.join(file_folder, file_name)
        file = np.memmap(file_path, dtype=datatype, mode='r')
        if shape:
            file = file.reshape(shape)

        if crop_range:
            file = file[crop_range[0], crop_range[1], crop_range[2]]

        img = sitk.GetImageFromArray(file)
        if resample:
            # 统一采样到1*1*2.5mm
            img = img_resmaple([1, 1, 2.5], ori_img_file=img)

        target_filepath = os.path.join(save_path,
                                       f"case{case}_T" + file_name[file_name.find('T') + 1] + "0.mhd")

        if not os.path.exists(target_filepath):
            sitk.WriteImage(img, target_filepath)

    print("{} convert done".format(file_folder))


def data_standardization_0_n(range, img):
    if torch.is_tensor(img):
        return range * (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    else:
        return range * (img - np.min(img)) / (np.max(img) - np.min(img))


# 最大最小归一化0-1
def data_standardization_min_max(range: list, img):
    img = img.astype("float32")
    min, max = range
    img = (img - min) / (max - min)
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def data_standardization_mean_std(img):
    return (img - np.mean(img)) / np.std(img)


# def affiine(move_img, fix_img, save_path):
#     outs = ants.registration(fix_img, move_img, type_of_transforme='Affine')
#     reg_img = outs['warpedmovout']
#     ants.image_write(reg_img, save_path)


def read_mhd(file_path):
    mhd_file = file_path
    itkimage = sitk.ReadImage(mhd_file)
    ct_value = sitk.GetArrayFromImage(itkimage)  # 这里一定要注意，得到的是[z,y,x]格式
    direction = itkimage.GetDirection()  # mhd文件中的TransformMatrix
    origin = np.array(itkimage.GetOrigin())
    spacing = np.array(itkimage.GetSpacing())  # 文件中的ElementSpacing

    return ct_value, spacing


def img_resmaple(new_spacing, resamplemethod=sitk.sitkLinear, ori_img_file=None, ori_img_path=None):
    """
        @param ori_img_file: sitk.Image
        @param ori_img_path: 原始的itk图像路径，一般为.mhd 两个参数二选一
        @param target_img_file: 保存路径
        @param new_spacing: 目标重采样的spacing，如[0.585938, 0.585938, 0.4]
        @param resamplemethod: itk插值⽅法: sitk.sitkLinear-线性、sitk.sitkNearestNeighbor-最近邻、sitk.sitkBSpline等，SimpleITK源码中会有各种插值的方法，直接复制调用即可
    """
    data = sitk.ReadImage(ori_img_path) if ori_img_file == None else ori_img_file  # 根据路径读取mhd文件
    original_spacing = data.GetSpacing()  # 获取图像重采样前的spacing
    original_size = data.GetSize()  # 获取图像重采样前的分辨率

    # 有原始图像size和spacing得到真实图像大小，用其除以新的spacing,得到变化后新的size
    new_shape = [
        int(np.round(original_spacing[0] * original_size[0] / new_spacing[0])),
        int(np.round(original_spacing[1] * original_size[1] / new_spacing[1])),
        int(np.round(original_spacing[2] * original_size[2] / new_spacing[2])),
    ]
    print("处理后新的分辨率:{}".format(new_shape))

    # 重采样构造器
    resample = sitk.ResampleImageFilter()

    resample.SetOutputSpacing(new_spacing)  # 设置新的spacing
    resample.SetOutputOrigin(data.GetOrigin())  # 原点坐标没有变，所以还用之前的就可以了
    resample.SetOutputDirection(data.GetDirection())  # 方向也未变
    resample.SetSize(new_shape)  # 分辨率发生改变
    resample.SetInterpolator(resamplemethod)  # 插值算法
    data = resample.Execute(data)  # 执行操作

    return data
    # sitk.WriteImage(data, os.path.join(ori_img_file, '_new'))  # 将处理后的数据，保存到一个新的mhd文件中


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkLinear):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    newSpacing = originSpacing * originSize / newSize
    newSize = newSize.astype(np.int)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


if __name__ == '__main__':

    project_folder = get_project_path("4DCT-R").split("4DCT-R")[0]

    # read_mhd(os.path.join(project_folder, f'datasets/dirlab/Case1Pack/Images_mhd'))

    # dirlab数据集img转mhd
    for item in case_cfg.items():
        case = item[0]
        shape = item[1]
        img_path = os.path.join(project_folder, f'datasets/dirlab/img/Case{case}Pack/Images')
        save_path = os.path.join(project_folder, f'datasets/dirlab/mhd/case{case}')
        make_dir(save_path)

        imgTomhd(img_path, save_path, np.int16, shape, case, False)
