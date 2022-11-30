import os
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

from utils.utilize import plotorsave_ct_scan, get_project_path, make_dir

# import ants

# DIRLAB 4DCT 1-10例的 z y x
dirlab_case_cfg = {
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

copd_case_cfg = {
    1: (121, 512, 512),
    2: (102, 512, 512),
    3: (126, 512, 512),
    4: (126, 512, 512),
    5: (131, 512, 512),
    6: (119, 512, 512),
    7: (112, 512, 512),
    8: (115, 512, 512),
    9: (116, 512, 512),
    10: (135, 512, 512),
}

dirlab_crop_range = [{},
                     {"case": 1,
                      "crop_range": [slice(0, 84), slice(43, 199), slice(10, 250)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (94, 256, 256)
                      },
                     {"case": 2,
                      "crop_range": [slice(5, 101), slice(30, 194), slice(8, 244)],
                      "pixel_spacing": np.array([1.16, 1.16, 2.5], dtype=np.float32),
                      "orign_size": (112, 256, 256)
                      },
                     {"case": 3,
                      "crop_range": [slice(0, 96), slice(42, 210), slice(10, 250)],
                      "pixel_spacing": np.array([1.15, 1.15, 2.5], dtype=np.float32),
                      "orign_size": (104, 256, 256)
                      },
                     {"case": 4,
                      "crop_range": [slice(0, 92), slice(42, 210), slice(10, 250)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (99, 256, 256)
                      },
                     {"case": 5,
                      "crop_range": [slice(0, 92), slice(60, 220), slice(10, 250)],
                      "pixel_spacing": np.array([1.10, 1.10, 2.5], dtype=np.float32),
                      "orign_size": (106, 256, 256)
                      },
                     {"case": 6,
                      "crop_range": [slice(10, 102), slice(144, 328), slice(132, 424)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (128, 512, 512)
                      },
                     {"case": 7,
                      "crop_range": [slice(10, 102), slice(144, 328), slice(114, 422)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (136, 512, 512)
                      },
                     {"case": 8,
                      "crop_range": [slice(18, 118), slice(84, 300), slice(113, 389)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (128, 512, 512)
                      },
                     {"case": 9,
                      "crop_range": [slice(0, 72), slice(126, 334), slice(128, 388)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (128, 512, 512)
                      },
                     {"case": 10,
                      "crop_range": [slice(0, 92), slice(119, 335), slice(140, 384)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (120, 512, 512)
                      }]


def crop_resampling_resize_clamp(sitk_img, new_size=None, crop_range=None, resample=None, clamp=None):
    """
    3D volume crop, resampling, resize and clamp
    Parameters
    ----------
    sitk_img: input img
    crop_range: x,y,z tuple[(),(),()]
    resample: x,y,z arr[, , ,]
    new_size: [z,y,x]
    clamp: [min,max]

    Returns: sitk_img
    -------

    """
    # crop
    if crop_range is not None:
        img_arr = sitk.GetArrayFromImage(sitk_img)
        img_arr = img_arr[crop_range[2], crop_range[1], crop_range[0]]
        img = sitk.GetImageFromArray(img_arr)

    else:
        img = sitk_img

    # resampling
    if resample is not None:
        img = img_resmaple(resample, ori_img_file=img)

    # resize and clamp HU[min,max]
    file = sitk.GetArrayFromImage(img)
    file = file.astype('float32')

    if new_size is not None:
        if clamp is not None:
            img_tensor = F.interpolate(torch.tensor(file).unsqueeze(0).unsqueeze(0), size=new_size,
                                       mode='trilinear',
                                       align_corners=False).clamp_(min=clamp[0], max=clamp[1])

        else:
            img_tensor = F.interpolate(torch.tensor(file).unsqueeze(0).unsqueeze(0), size=new_size,
                                       mode='trilinear',
                                       align_corners=False)

        img = sitk.GetImageFromArray(np.array(img_tensor)[0, 0, ...])
    elif clamp is not None:
        img_tensor = torch.tensor(file).clamp_(min=clamp[0], max=clamp[1])
        img = sitk.GetImageFromArray(np.array(img_tensor))

    return img


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
        plotorsave_ct_scan(ct_value, "plot")


def read_dcm_series(dcm_path):
    """
    Parameters
    ----------
    dcm_path

    Returns sitk image
    -------
    """

    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_path)  # 获取该路径下的seriesid的数量
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_path)  # 获取该路径下所有的.dcm文件，并且根据世界坐标从小到大排序
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image_sitk = series_reader.Execute()  # 生成3D图像

    return image_sitk


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


def dirlab_test(file_folder, m_path, f_path, datatype, shape, case):
    for file_name in os.listdir(file_folder):
        if 'T00' in file_name:
            target_path = m_path

        # T50 = fixed image
        elif 'T50' in file_name:
            target_path = f_path

        else:
            continue

        file_path = os.path.join(file_folder, file_name)
        file = np.memmap(file_path, dtype=datatype, mode='r')
        if shape:
            file = file.reshape(shape)

        img = sitk.GetImageFromArray(file)

        img = crop_resampling_resize_clamp(img, None,
                                           dirlab_crop_range[case]['crop_range'][::-1],
                                           [1, 1, 1],
                                           [100, 900])

        target_file_path = os.path.join(target_path,
                                        f"dirlab_case{case}.nii.gz")

        sitk.WriteImage(img, target_file_path)


def dirlab_processing(file_folder, m_path, f_path, datatype, shape, case):
    for file_name in os.listdir(file_folder):
        target_path = m_path
        # T50 = fixed image
        if 'T50' in file_name:
            target_path = f_path

        file_path = os.path.join(file_folder, file_name)
        file = np.memmap(file_path, dtype=datatype, mode='r')
        if shape:
            file = file.reshape(shape)

        img = sitk.GetImageFromArray(file)

        img = crop_resampling_resize_clamp(img, [144, 256, 256],
                                           dirlab_crop_range[case]['crop_range'][::-1],
                                           [0.6, 0.6, 0.6],
                                           [100, 900])

        target_file_path = os.path.join(target_path,
                                        f"dirlab_case{case}_T" + file_name[file_name.find('T') + 1] + "0.nii.gz")
        if target_path == f_path:
            for i in ['0', '1', '2', '3', '4', '6', '7', '8', '9']:
                new_name = f'dirlab_case{case}_T' + i + '0.nii.gz'
                target_file_path = os.path.join(target_path, new_name)
                sitk.WriteImage(img, target_file_path)
        else:
            sitk.WriteImage(img, target_file_path)

    print("{} convert done".format(file_folder))


def copd_processing(img_path, target_path, datatype, shape, case, resample=False):
    file = np.memmap(img_path, dtype=datatype, mode='r')
    if shape:
        file = file.reshape(shape)

    sitk_img = sitk.GetImageFromArray(file)
    img = crop_resampling_resize_clamp(sitk_img, None,
                                       [slice(70, 470), slice(30, 470), slice(None)], [1, 1, 1],
                                       [0, 900])
    # # crop
    # file = file[:, 30:470, 70:470]
    # img = sitk.GetImageFromArray(file)
    #
    # # resampling
    # if resample:
    #     # 采样到x*y*zmm
    #     img = img_resmaple([0.6, 0.6, 0.6], ori_img_file=img)
    #
    # # resize HU[0, 900]
    # file = sitk.GetArrayFromImage(img)
    # file = file.astype('float32')
    # img_tensor = F.interpolate(torch.tensor(file).unsqueeze(0).unsqueeze(0), size=[144, 256, 256], mode='trilinear',
    #                            align_corners=False).clamp_(min=0, max=900)
    #
    # # save
    # img = sitk.GetImageFromArray(np.array(img_tensor)[0, 0, ...])
    make_dir(target_path)
    target_filepath = os.path.join(target_path,
                                   f"copd_case{case}.nii.gz")
    # if not os.path.exists(target_filepath):
    sitk.WriteImage(img, target_filepath)


def learn2reg_processing(fixed_path, moving_path):
    print("learn2reg: ")
    l2r_path = r'E:\datasets\Learn2Reg\all'

    for file_name in os.listdir(l2r_path):
        target_path = moving_path
        file = os.path.join(l2r_path, file_name)
        # exp -> fixed insp -> moving
        if 'exp' in file_name:
            target_path = fixed_path

        # open nii
        img_nii = sitk.ReadImage(file)

        img = crop_resampling_resize_clamp(img_nii, None,
                                           None, [1, 1, 1],
                                           [0, 900])

        # # resampling
        # img_nii = img_resmaple([0.6, 0.6, 0.6], ori_img_file=img_nii)
        #
        # # resize HU[0, 900]
        # img_arr = sitk.GetArrayFromImage(img_nii)
        # img_arr = img_arr.astype('float32')
        # img_tensor = F.interpolate(torch.tensor(img_arr).unsqueeze(0).unsqueeze(0), size=[144, 256, 256],
        #                            mode='trilinear',
        #                            align_corners=False).clamp_(min=0, max=900)
        # img = sitk.GetImageFromArray(np.array(img_tensor)[0, 0, ...])

        # save
        make_dir(target_path)
        case = file_name.split('_')[1]
        target_filepath = os.path.join(target_path,
                                       f"l2r_case{case}.nii.gz")
        # if not os.path.exists(target_filepath):
        sitk.WriteImage(img, target_filepath)
        print('case{} done'.format(case))


def emp10_processing(fixed_path, moving_path):
    print("emp10: ")
    emp_path = r'E:\datasets\emp10\emp30'

    file_list = sorted([file_name for file_name in os.listdir(emp_path) if file_name.lower().endswith('mhd')])

    for file_name in file_list:
        target_path = moving_path
        file = os.path.join(emp_path, file_name)
        # exp -> fixed insp -> moving
        if 'Fixed' in file_name:
            target_path = fixed_path

        # open nii
        img_nii = sitk.ReadImage(file)

        img = crop_resampling_resize_clamp(img_nii, None,
                                           None, [1, 1, 1],
                                           [-900, 500])
        # # resampling
        # img_nii = img_resmaple([0.6, 0.6, 0.6], ori_img_file=img_nii)
        #
        # # resize HU[0, 900]
        # img_arr = sitk.GetArrayFromImage(img_nii)
        # img_arr = img_arr.astype('float32')
        # img_tensor = F.interpolate(torch.tensor(img_arr).unsqueeze(0).unsqueeze(0), size=[144, 256, 256],
        #                            mode='trilinear',
        #                            align_corners=False).clamp_(min=-900, max=500)
        # img = sitk.GetImageFromArray(np.array(img_tensor)[0, 0, ...])

        # save
        make_dir(target_path)
        case = file_name.split('_')[0]
        target_filepath = os.path.join(target_path,
                                       f"emp_case{case}.nii.gz")
        # if not os.path.exists(target_filepath):
        sitk.WriteImage(img, target_filepath)
        print('case{} done'.format(case))


def popi_processing(fixed_path, moving_path):
    print("popi: ")
    # for case in range(1, 7):
    #     popi_path = f'E:/datasets/creatis/case{case}/Images/'
    #     for T in [file_name for file_name in os.listdir(popi_path) if '.gz' not in file_name]:
    #         target_path = moving_path
    #
    #         if case != 1 and T == '50':
    #             target_path = fixed_path
    #
    #         if case == 1 and T == '60':
    #             target_path = fixed_path
    #
    #         # dcm slice -> 3D nii.gz
    #         sitk_img = read_dcm_series(os.path.join(popi_path, T))
    #
    #         if case == 5 or case == 6:
    #             img = crop_resampling_resize_clamp(sitk_img, None,
    #                                                [slice(100, 400), slice(120, 360), slice(None)], [1, 1, 1],
    #                                                [-800, -100])
    #         else:
    #             img = crop_resampling_resize_clamp(sitk_img, None,
    #                                                [slice(70, 460), slice(40, 380), slice(None)], [1, 1, 1],
    #                                                [-800, -100])
    #
    #         # save
    #         make_dir(target_path)
    #
    #         # if this image is fixed image, then copy
    #         if target_path == fixed_path:
    #             if case == 1:
    #                 for t in ['00', '10', '20', '30', '40', '50', '70', '80', '90']:
    #                     target_file_path = os.path.join(target_path, 'popi_case{}_T{}.nii.gz'.format(case, t))
    #                     sitk.WriteImage(img, target_file_path)
    #             else:
    #                 for t in ['00', '10', '20', '30', '40', '60', '70', '80', '90']:
    #                     target_file_path = os.path.join(target_path, 'popi_case{}_T{}.nii.gz'.format(case, t))
    #                     sitk.WriteImage(img, target_file_path)
    #
    #         else:
    #             target_file_path = os.path.join(target_path, 'popi_case{}_T{}.nii.gz'.format(case, T))
    #             sitk.WriteImage(img, target_file_path)
    #
    #     print("case{} done".format(case))

    # case 7 is .mhd
    case = 7
    mhd_path = r'E:\datasets\creatis\case7\Images'
    for T in [file_name for file_name in os.listdir(mhd_path) if '.mhd' in file_name]:
        target_path = moving_path

        if '50' in T:
            target_path = fixed_path

        img_path = os.path.join(mhd_path, T)
        sitk_img = sitk.ReadImage(img_path)
        img = crop_resampling_resize_clamp(sitk_img, None,
                                           [slice(70, 460), slice(25, None, None), slice(None)], [1, 1, 1],
                                           [-900, -100])

        # save
        make_dir(target_path)

        # if this image is fixed image, then copy
        if target_path == fixed_path:
            for t in ['00', '10', '20', '30', '40', '60', '70', '80', '90']:
                target_file_path = os.path.join(target_path, 'popi_case{}_T{}.nii.gz'.format(case, t))
                sitk.WriteImage(img, target_file_path)

        else:
            target_file_path = os.path.join(target_path, 'popi_case{}_T{}.nii.gz'.format(case, T.split('-')[0]))
            sitk.WriteImage(img, target_file_path)

    print('case7 done')


if __name__ == '__main__':
    project_folder = get_project_path("4DCT").split("4DCT")[0]
    target_fixed_path = f'E:/datasets/registration/train_ori/fixed'
    target_moving_path = f'E:/datasets/registration/train_ori/moving'

    target_test_fixed_path = f'E:/datasets/registration/test_ori/fixed'
    target_test_moving_path = f'E:/datasets/registration/test_ori/moving'

    # read_mhd(os.path.join(project_folder, f'datasets/dirlab/Case1Pack/Images_mhd'))
    make_dir(target_moving_path)
    make_dir(target_fixed_path)
    make_dir(target_test_fixed_path)
    make_dir(target_test_moving_path)

    # # dirlab数据集img转mhd
    # for item in dirlab_case_cfg.items():
    #     case = item[0]
    #     shape = item[1]
    #     img_path = os.path.join(project_folder, f'datasets/dirlab/img/Case{case}Pack/Images')
    #     dirlab_processing(img_path, target_moving_path, target_fixed_path, np.int16, shape, case)

    # dirlab for test
    print("dirlab: ")
    for item in dirlab_case_cfg.items():
        case = item[0]
        shape = item[1]
        img_path = os.path.join(project_folder, f'datasets/dirlab/img/Case{case}Pack/Images')
        dirlab_test(img_path, target_test_moving_path, target_test_fixed_path, np.int16, shape, case)

    # # COPD数据集img转nii.gz
    # print("copd: ")
    # for item in copd_case_cfg.items():
    #     case = item[0]
    #     shape = item[1]
    #
    #     fixed_path = f'E:/datasets/copd/copd{case}/copd{case}/copd{case}_eBHCT.img'
    #     moving_path = f'E:/datasets/copd/copd{case}/copd{case}/copd{case}_iBHCT.img'
    #     copd_processing(fixed_path, target_fixed_path, np.int16, shape, case, True)
    #     copd_processing(moving_path, target_moving_path, np.int16, shape, case, True)

    # # learn2reg
    # learn2reg_processing(target_fixed_path, target_moving_path)
    #
    # # emp10
    # emp10_processing(target_fixed_path, target_moving_path)

    # creatis-popi
    # popi_processing(target_fixed_path, target_moving_path)

    # moving_path = os.path.join(project_folder, f'datasets/registration/moving')
    # fixed_path = os.path.join(project_folder, f'datasets/registration/fixed')
    # make_dir(moving_path)
    # make_dir(fixed_path)
    #
    # imgTomhd(img_path, moving_path, fixed_path, np.int16, shape, case, True)

    # # 真实病例
    # img_path = os.path.join(project_folder, f'datasets/4DCT_nii/')
    # for file_name in os.listdir(img_path):
    #     mhd_file = os.path.join(img_path, file_name)
    #     itkimage = sitk.ReadImage(mhd_file)
    #     ct_value = sitk.GetArrayFromImage(itkimage)  # 这里一定要注意，得到的是[z,y,x]格式
    #     direction = itkimage.GetDirection()  # mhd文件中的TransformMatrix
    #     origin = np.array(itkimage.GetOrigin())
    #     spacing = np.array(itkimage.GetSpacing())  # 文件中的ElementSpacing
    #     plotorsave_ct_scan(ct_value, "plot")
