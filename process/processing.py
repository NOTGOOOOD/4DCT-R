import os
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import torch

from utils.utilize import plotorsave_ct_scan, get_project_path

# import ants

# DIRLAB 4DCT 1-10例的 z y x
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


def imgTomhd(file_folder, m_path, f_path, datatype, shape, case):
    for file_name in os.listdir(file_folder):
        is_fixed = False
        if 'T50' in file_name:
            is_fixed = True
        file_path = os.path.join(file_folder, file_name)
        file = np.memmap(file_path, dtype=datatype, mode='r')
        if shape:
            file = file.reshape(shape)

        img = sitk.GetImageFromArray(file)
        target_filepath = os.path.join(m_path, file_name.split('.')[0] + "_moving.mhd")
        if is_fixed:
            for i in range(10):
                if i == 5:
                    continue
                new_name = f'dirlab_case{case}_T' + str(i) + '0_fixed.mhd'
                target_filepath = os.path.join(f_path, new_name)
                if not os.path.exists(target_filepath):
                    sitk.WriteImage(img, target_filepath)
            continue

        if not os.path.exists(target_filepath):
            sitk.WriteImage(img, target_filepath)

    print("{} convert done".format(file_folder))


def data_standardization_0_n(range, img):
    if torch.is_tensor(img):
        return range * (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    else:
        return range * (img - np.min(img)) / (np.max(img) - np.min(img))


def data_standardization_mean_std(img):
    return (img - np.mean(img)) / np.std(img)


# def affiine(move_img, fix_img, save_path):
#     outs = ants.registration(fix_img, move_img, type_of_transforme='Affine')
#     reg_img = outs['warpedmovout']
#     ants.image_write(reg_img, save_path)


def read_mhd(mhd_dir):
    for file_name in os.listdir(mhd_dir):
        mhd_file = os.path.join(mhd_dir, file_name)
        itkimage = sitk.ReadImage(mhd_file)
        ct_value = sitk.GetArrayFromImage(itkimage)  # 这里一定要注意，得到的是[z,y,x]格式
        direction = itkimage.GetDirection()  # mhd文件中的TransformMatrix
        origin = np.array(itkimage.GetOrigin())
        spacing = np.array(itkimage.GetSpacing())  # 文件中的ElementSpacing

        img_arr = ct_value.copy()  # ndarray

        level = -200  # 窗位
        window = 1600  # 窗宽

        window_minimum = level - window / 2
        window_maximum = level + window / 2
        img_arr[img_arr < window_minimum] = window_minimum
        img_arr[img_arr > window_maximum] = window_maximum

        plotorsave_ct_scan(ct_value, "plot")
        plotorsave_ct_scan(img_arr, "plot")

        mha_img = sitk.GetImageFromArray(img_arr)
        sitk.WriteImage(mha_img, 'test1.mhd')

        test2 = set_window(ct_value, window, level)
        plotorsave_ct_scan(test2, "plot")
        mha_img = sitk.GetImageFromArray(test2)
        sitk.WriteImage(mha_img, 'test2.mhd')

        plt.show()
        break


def img_resmaple(ori_img_file, new_spacing, resamplemethod=sitk.sitkLinear):
    """
        @param ori_img_file: 原始的itk图像路径，一般为.mhd
        @param target_img_file: 保存路径
        @param new_spacing: 目标重采样的spacing，如[0.585938, 0.585938, 0.4]
        @param resamplemethod: itk插值⽅法: sitk.sitkLinear-线性、sitk.sitkNearestNeighbor-最近邻、sitk.sitkBSpline等，SimpleITK源码中会有各种插值的方法，直接复制调用即可
    """
    data = sitk.ReadImage(ori_img_file)  # 根据路径读取mhd文件
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

    sitk.WriteImage(data, target_img_file)  # 将处理后的数据，保存到一个新的mhdw文件中


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

    project_folder = get_project_path("4DCT").split("4DCT")[0]

    pass

    # read_mhd(os.path.join(project_folder, f'datasets/dirlab/Case1Pack/Images_mhd'))

    # # dirlab数据集img转mhd
    # for item in case_cfg.items():
    #     case = item[0]
    #     shape = item[1]
    #     img_path = os.path.join(project_folder, f'datasets/dirlab/Case{case}Pack/Images')
    #     moving_path = os.path.join(project_folder, f'datasets/registration/moving')
    #     fixed_path = os.path.join(project_folder, f'datasets/registration/fixed')
    #     ut.make_dir(moving_path)
    #     ut.make_dir(fixed_path)
    #
    #     imgTomhd(img_path, moving_path, fixed_path, np.int16, shape, case)

    # import shutil
    #
    # # 10个病人
    # for i in range(10):
    #     case = i + 1
    #     data_folder = os.path.join(project_folder.split("4DCT")[0], f'datasets/dirlab/Case{case}_mhd/')
    #
    #     for file in os.listdir(data_folder):
    #         filename = f'Case{case}'
    #         os.rename(file, filename)
    #         shutil.copy()
    #
    #     print(f"{case} converted")
