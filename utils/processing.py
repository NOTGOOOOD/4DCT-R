import os
import shutil
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

from utils.utilize import plotorsave_ct_scan, get_project_path, make_dir
from utils.config import get_args

# import ants

# DIRLAB 4DCT-R 1-10例的 z y x
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


def crop_resampling_resize_clamp(sitk_img, new_size=None, crop_range=None, spacing=None, clamp=None):
    """
    3D volume crop, resampling, resize and clamp
    Parameters
    ----------
    sitk_img: input img
    crop_range: x,y,z tuple[(),(),()]
    spacing: x,y,z arr[, , ,]
    new_size: [z,y,x]
    clamp: [min,max]

    Returns: sitk_img
    -------

    """
    origin_spacing = list(sitk_img.GetSpacing()) # tuple -> list
    # # crop
    # if crop_range is not None:
    #     img_arr = sitk.GetArrayFromImage(sitk_img)
    #     img_arr = img_arr[crop_range[2], crop_range[1], crop_range[0]]
    #     img = sitk.GetImageFromArray(img_arr)
    #
    # else:
    #     img = sitk_img

    file = sitk.GetArrayFromImage(sitk_img)
    file = file.astype('float')
    img_tensor = torch.tensor(file)

    # clamp HU[min,max]
    if clamp is not None:
        img_tensor = img_tensor.clamp_(min=clamp[0], max=clamp[1])
    # crop
    if crop_range is not None:
        img_tensor = img_tensor[crop_range[2], crop_range[1], crop_range[0]]

    img = sitk.GetImageFromArray(np.array(img_tensor).squeeze())

    if spacing is None:
        spacing = origin_spacing

    # resample & resize
    img = resample_image(img, origin_spacing=origin_spacing, out_spacing=spacing, restore=True, new_size=new_size)

    return img


def data_standardization_0_n(range, img):
    # return img / 1250.
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

def gaussian_noise(x):
    """
    add random gaussian noise for input image
    Args:
        x: input image array,

    Returns:
        x: standard x
        y: aug x

    """
    # 均值为0，标准差为1，上下限为±0.2
    x = data_standardization_0_n(1, x)
    noise = np.array(torch.clamp(torch.randn_like(torch.from_numpy(x)) * 0.1, -0.05, 0.05))
    y = x + noise
    return x, y


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

def array_to_sitk(sitk_img, origin=None, spacing=None, direction=None, is_vector=False, im_ref=None, new_size=None):
    sitk_img = sitk.GetArrayFromImage(sitk_img)
    if new_size is not None:
        img_tensor = torch.tensor(sitk_img.astype('float'))
        img_tensor = F.interpolate(img_tensor.unsqueeze(0).unsqueeze(0), size=new_size,
                                       mode='trilinear',
                                       align_corners=False)
        sitk_img = np.array(img_tensor).squeeze()

    sitk_img = sitk.GetImageFromArray(sitk_img)
    if origin is None:
        origin = [0, 0, 0]
    if spacing is None:
        spacing = [1, 1, 1]
    else:
        if not isinstance(spacing, list):
            spacing = list(spacing.astype('float'))
    if direction is None:
        direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    sitk_output = sitk_img
    if im_ref is None:
        sitk_output.SetOrigin(origin)
        sitk_output.SetSpacing(spacing)
        sitk_output.SetDirection(direction)
    else:
        sitk_output.SetOrigin(im_ref.GetOrigin())
        sitk_output.SetSpacing(im_ref.GetSpacing())
        sitk_output.SetDirection(im_ref.GetDirection())
    return sitk_output

def resample_image(itk_image, origin_spacing=[1.0,1.0,1.0], restore=True, out_spacing=[1.0, 1.0, 1.0], is_label=False, new_size=None):
    # 不需要插值，只设置spacing
    if origin_spacing==out_spacing:
        return array_to_sitk(itk_image, spacing=out_spacing, new_size=new_size)

    # 重采样 & 插值
    clamp0, clamp1 = sitk.GetArrayFromImage(itk_image).min(), sitk.GetArrayFromImage(itk_image).max()
    original_spacing = origin_spacing
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label: # 如果是mask图像，就选择sitkNearestNeighbor这种插值
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else: # 如果是普通图像，就采用sitkBSpline插值法
        resample.SetInterpolator(sitk.sitkBSpline)
    data = resample.Execute(itk_image)

    ori = data.GetOrigin()
    direction = data.GetDirection()
    spacing = data.GetSpacing()

    file = sitk.GetArrayFromImage(data)
    file = file.astype('float')
    img_tensor = torch.tensor(file)
    img_tensor = img_tensor.clamp_(min=clamp0, max=clamp1)

    # 插值还原原始的尺寸
    if restore:
        img_tensor = F.interpolate(img_tensor.unsqueeze(0).unsqueeze(0), size=[original_size[-1],original_size[-2],original_size[-3]], mode='trilinear', align_corners=True)

    img = sitk.GetImageFromArray(np.array(img_tensor).squeeze())
    data = array_to_sitk(img, origin=ori, direction=direction, spacing=out_spacing, new_size=new_size)

    return data

def img_resmaple(new_spacing, resamplemethod=sitk.sitkLinear, ori_img_file=None, ori_img_path=None):
    """
        @param ori_img_file: sitk.Image
        @param ori_img_path: 原始的itk图像路径，.mhd .nii等 两个参数二选一
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


def resize_image(itkimage, newSize, resamplemethod=sitk.sitkLinear):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)

    newSpacing = [
        int(np.round(originSpacing[0] * originSize[0] / newSize[0])),
        int(np.round(originSpacing[1] * originSize[1] / newSize[1])),
        int(np.round(originSpacing[2] * originSize[2] / newSize[2])),
    ]

    newSize = newSize.astype(np.int)  # size can not be float
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


def dirlab_train(file_folder, m_path, f_path):
    # every case
    for num_fixed, file_name_fixed in enumerate(os.listdir(file_folder)):
        # for file_name_fixed in os.listdir(file_folder):
        # every data as a fiexed [T00-T90]
        # fixed_name = file_name_fixed.split('.nii')[0] + '_%02d' % num_fixed + '.nii.gz'
        if 'T05' in file_name_fixed:
            for num_moving, file_name_moving in enumerate(os.listdir(file_folder)):
                if file_name_moving == file_name_fixed:
                    continue

                new_name = file_name_fixed.split('.nii')[0] + '_T%02d' % num_moving + '.nii.gz'

                fixed_file_path = os.path.join(f_path, new_name)
                moving_file_path = os.path.join(m_path, new_name)

                shutil.copyfile(os.path.join(file_folder, file_name_fixed), fixed_file_path)
                shutil.copyfile(os.path.join(file_folder, file_name_moving), moving_file_path)


def dirlab_processing(args, save_path, file_folder, datatype, shape, case, resize:list =None, isotropic=False):
    """

    Parameters
    ----------
    args
    save_path
    file_folder
    datatype
    shape
    case
    resize z,y,x
    isotropic

    Returns
    -------

    """
    clamp = [None, 1250]
    spacing = args.dirlab_cfg[case]['pixel_spacing']
    crop_range = args.dirlab_cfg[case]['crop_range'][::-1]
    for num, file_name in enumerate(os.listdir(file_folder)):

        file_path = os.path.join(file_folder, file_name)
        file = np.memmap(file_path, dtype=datatype, mode='r')
        if shape:
            file = file.reshape(shape)

        file = file.astype('float')
        img_tensor = torch.tensor(file)

        # clamp HU[min,max]
        if clamp is not None:
            img_tensor = img_tensor.clamp_(min=clamp[0], max=clamp[1])
        # crop
        if crop_range is not None:
            img_tensor = img_tensor[crop_range[2], crop_range[1], crop_range[0]]

        img = sitk.GetImageFromArray(np.array(img_tensor).squeeze())

        img = array_to_sitk(img, spacing=spacing, new_size=resize)

        case_name = 'dirlab_case%02d_T%02d.nii.gz' % (case, num)
        target_file_path = os.path.join(save_path, case_name)

        sitk.WriteImage(img, target_file_path)

def dirlab_test_copy(file_folder, m_path, f_path, case):
    for file_name in os.listdir(file_folder):
        if 'T00' in file_name:
            target_path = m_path

        # T50 = fixed image
        elif 'T50' in file_name:
            target_path = f_path

        else:
            continue

        if case < 10:
            new_name = f'dirlab_case0{case}.nii.gz'
        else:
            new_name = f'dirlab_case{case}.nii.gz'

        file_path = os.path.join(file_folder, file_name)
        target_path = os.path.join(target_path, new_name)

        # move file to targe path
        shutil.copyfile(file_path, target_path)
        print("copy {} to {}".format(file_path, target_path))

def dirlab_test(args, file_folder, m_path, f_path, datatype, shape, case):
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
                                           args.dirlab_cfg[case]['crop_range'][::-1],
                                           args.dirlab_cfg[case]['pixel_spacing'],
                                           [0, 1250])

        case_name = 'dirlab_case%02d.nii.gz' % case
        target_file_path = os.path.join(target_path,
                                        case_name)

        sitk.WriteImage(img, target_file_path)


def copd_processing(img_path, target_path, datatype, shape, case, **cfg):
    file = np.memmap(img_path, dtype=datatype, mode='r')
    if shape:
        file = file.reshape(shape)

    sitk_img = sitk.GetImageFromArray(file)
    img = crop_resampling_resize_clamp(sitk_img, cfg['resize'], cfg['crop']
                                       , cfg['spacing'],
                                       cfg['clamp'])
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
    # file = file.astype('float')
    # img_tensor = F.interpolate(torch.tensor(file).unsqueeze(0).unsqueeze(0), size=[144, 256, 256], mode='trilinear',
    #                            align_corners=False).clamp_(min=0, max=900)
    #
    # # save
    # img = sitk.GetImageFromArray(np.array(img_tensor)[0, 0, ...])
    make_dir(target_path)
    target_filepath = os.path.join(target_path,
                                   "copd_case%02d.nii.gz" % case)
    # if not os.path.exists(target_filepath):
    sitk.WriteImage(img, target_filepath)


def learn2reg_processing(fixed_path, moving_path, **cfg):
    print("learn2reg: ")
    l2r_path = r'E:\datasets\Learn2Reg\all'
    # l2r_path = '/home/cqut/project/xxf/Learn2Reg'

    file_list = sorted([file_name for file_name in os.listdir(l2r_path) if file_name.lower().endswith('.gz')])

    for file_name in file_list:
        target_path = moving_path
        file = os.path.join(l2r_path, file_name)
        # exp -> fixed insp -> moving
        file_prefix = file_name[:8]
        file_suffix = file_name.split('_')[2]

        if 'exp' in file_suffix:
            target_path = fixed_path

        # open nii
        img_nii = sitk.ReadImage(file)

        img = crop_resampling_resize_clamp(img_nii, cfg['resize'], cfg['crop']
                                           , cfg['spacing'],
                                           cfg['clamp'])
        # save
        make_dir(target_path)
        target_filepath = os.path.join(target_path,
                                       "l2r_{}.nii.gz".format(file_prefix))
        # if not os.path.exists(target_filepath):
        sitk.WriteImage(img, target_filepath)
        print('case{} done'.format(file_prefix))


def learn2reg_lungct_processing(fixed_path, moving_path, **cfg):
    print("learn2reg: ")
    l2r_path = r'E:\datasets\LungCT\all'

    file_list = sorted([file_name for file_name in os.listdir(l2r_path) if file_name.lower().endswith('.gz')])

    for file_name in file_list:
        target_path = moving_path
        file = os.path.join(l2r_path, file_name)
        # exp -> fixed insp -> moving
        file_prefix = file_name[:11]
        file_suffix = file_name.split('_')[2]

        if '0000' in file_suffix:
            target_path = fixed_path

        # open nii
        img_nii = sitk.ReadImage(file)

        img = crop_resampling_resize_clamp(img_nii, cfg['resize'], cfg['crop']
                                           , cfg['spacing'],
                                           cfg['clamp'])
        # save
        make_dir(target_path)
        target_filepath = os.path.join(target_path,
                                       "{}.nii.gz".format(file_prefix))
        # if not os.path.exists(target_filepath):
        sitk.WriteImage(img, target_filepath)
        print('case{} done'.format(file_prefix))


def patient_processing(fixed_path, moving_path, **cfg):
    patient_path = r'E:\datasets\patient'
    for patient_name in os.listdir(patient_path):
        patient_folder = os.path.join(patient_path, patient_name)
        file_list = sorted([file_name for file_name in os.listdir(patient_folder) if file_name.lower().endswith('.gz')])
        for file_name in file_list:
            target_path = moving_path
            file = os.path.join(patient_folder, file_name)
            if 'ct_5' in file_name:
                target_path = fixed_path

            # open nii
            img_nii = sitk.ReadImage(file)

            img = crop_resampling_resize_clamp(img_nii, cfg['resize'], cfg['crop']
                                               , cfg['spacing'],
                                               cfg['clamp'])

            make_dir(target_path)
            if target_path == fixed_path:
                for t in ['00', '01', '02', '03', '04', '06', '07', '08', '09']:
                    img_name = '%s_T%s.nii.gz' % (patient_name, t)
                    target_file_path = os.path.join(target_path, img_name)
                    sitk.WriteImage(img, target_file_path)

            else:
                phase = int(file_name.split('_')[1].split('.nii')[0])
                img_name = '%s_T%02d.nii.gz' % (patient_name, phase)
                target_filepath = os.path.join(target_path, img_name)
                # if not os.path.exists(target_filepath):
                sitk.WriteImage(img, target_filepath)


def emp10_processing(fixed_path, moving_path, **cfg):
    print("emp10: ")
    emp_path = r'E:\datasets\emp10\emp30'

    file_list = sorted([file_name for file_name in os.listdir(emp_path) if file_name.lower().endswith('mhd')])

    for file_name in file_list:
        target_path = moving_path
        file = os.path.join(emp_path, file_name)
        if 'Fixed' in file_name:
            target_path = fixed_path

        # open nii
        img_nii = sitk.ReadImage(file)

        img = crop_resampling_resize_clamp(img_nii, cfg['resize'], cfg['crop']
                                           , cfg['spacing'],
                                           cfg['clamp'])
        # save
        make_dir(target_path)
        case = file_name.split('_')[0]
        target_filepath = os.path.join(target_path,
                                       f"emp_case{case}.nii.gz")
        # if not os.path.exists(target_filepath):
        sitk.WriteImage(img, target_filepath)
        print('case{} done'.format(case))


def popi_processing(fixed_path, moving_path):
    # crop_range x y z
    crop_range = {
        1:[slice(60,448), slice(30,380), slice(None)],
        2: [slice(60,430), slice(45,380), slice(None)],
        3: [slice(80,420), slice(120,380), slice(None)],
        4: [slice(60,440), slice(80,380), slice(None)],
        5: [slice(120,390), slice(100,340), slice(None)],
        6: [slice(90,400), slice(110,340), slice(None)],
        7: [slice(None), slice(None), slice(None)],
                  }
    clamp = [-1000,500]
    spacing = None
    resize = [160,224,224]  # z y x

    print("popi: ")
    for case in range(1, 7):
        popi_path = f'E:/datasets/creatis/case{case}/Images/'
        for T in [file_name for file_name in os.listdir(popi_path) if '.gz' not in file_name]:
            target_path = moving_path

            if case != 1 and T == '50':
                target_path = fixed_path

            if case == 1 and T == '60':
                target_path = fixed_path

            # dcm slice -> 3D nii.gz
            sitk_img = read_dcm_series(os.path.join(popi_path, T))


            img = crop_resampling_resize_clamp(sitk_img, new_size=resize,crop_range=crop_range[case],
                                               spacing=spacing,clamp=clamp)

            # save
            make_dir(target_path)

            # if this image is fixed image, then copy
            if target_path == fixed_path:
                if case == 1:
                    for t in ['00', '10', '20', '30', '40', '50', '70', '80', '90']:
                        target_file_path = os.path.join(target_path, 'popi_case{}_T{}.nii.gz'.format(case, t))
                        sitk.WriteImage(img, target_file_path)
                else:
                    for t in ['00', '10', '20', '30', '40', '60', '70', '80', '90']:
                        target_file_path = os.path.join(target_path, 'popi_case{}_T{}.nii.gz'.format(case, t))
                        sitk.WriteImage(img, target_file_path)

            else:
                target_file_path = os.path.join(target_path, 'popi_case{}_T{}.nii.gz'.format(case, T))
                sitk.WriteImage(img, target_file_path)

        print("case{} done".format(case))

    # case 7 is .mhd
    case = 7
    mhd_path = r'E:\datasets\creatis\case7\Images'
    for T in [file_name for file_name in os.listdir(mhd_path) if '.mhd' in file_name]:
        target_path = moving_path

        if '50' in T:
            target_path = fixed_path

        img_path = os.path.join(mhd_path, T)
        sitk_img = sitk.ReadImage(img_path)
        img = crop_resampling_resize_clamp(sitk_img, new_size=resize,crop_range=crop_range[case], spacing=spacing,clamp=clamp)

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


def tcia_processing(fixed_path, moving_path, **cfg):
    tcia_folder = r'D:\project\xxf\datasets\other\4D-Lung'
    # for patien_folder in os.listdir(tcia_folder):
    #     patient_no = patien_folder.split('_')[0]
    #     patien_path = os.path.join(tcia_folder, patien_folder)
    if True:
        patient_no = 116
        patien_path = r'D:\project\xxf\datasets\other\4D-Lung\%d_HM10395' % patient_no

        # for scan_times in os.listdir(patien_path):
        for scan_times in ['05-24-2000-NA-p4-91968', '06-08-2000-NA-p4-87118', '06-22-2000-NA-p4-94897',
                           '06-29-2000-NA-p4-10940', '07-06-2000-NA-p4-38364']:
            scans_path = os.path.join(patien_path, scan_times)

            T = 0
            for dcm_folder in sorted([file_folder for file_folder in os.listdir(scans_path)]):
                target_path = moving_path
                dcm_path = os.path.join(scans_path, dcm_folder)

                # only one .dcm
                if len(os.listdir(dcm_path)) < 2:
                    continue

                sitk_img = read_dcm_series(dcm_path)

                img = crop_resampling_resize_clamp(sitk_img, cfg['resize'], cfg['crop']
                                                   , cfg['spacing'],
                                                   cfg['clamp'])

                if T == 5:
                    target_path = fixed_path

                if target_path == fixed_path:
                    for t in ['00', '10', '20', '30', '40', '60', '70', '80', '90']:
                        target_file_path = os.path.join(target_path,
                                                        'tcia_case{}_time{}_T{}.nii.gz'.format(patient_no,
                                                                                               scan_times[:10], t))
                        if os.path.exists(target_file_path):
                            print('{} already exists!'.format(target_file_path))

                        else:
                            sitk.WriteImage(img, target_file_path)

                else:
                    target_file_path = os.path.join(target_path, 'tcia_case%s_time%s_T%d0.nii.gz' % (
                        patient_no, scan_times[:10], T))

                    if os.path.exists(target_file_path):
                        print('{} already exists!'.format(target_file_path))

                    else:
                        sitk.WriteImage(img, target_file_path)

                T = T + 1

        print("{} done!!".format(scan_times))

    print("%d done!!" % patient_no)


def aug(img_path, save_path):
    # flip
    # for img_name in os.listdir(img_path):
    #     itk_img = sitk.ReadImage(os.path.join(img_path, img_name))
    #     img_arr = sitk.GetArrayFromImage(itk_img)
    #     img_arr_new = img_arr[:, :, ::-1]
    #     img_new = sitk.GetImageFromArray(img_arr_new)
    #     img_new_name = 'flip_' + img_name
    #     sitk.WriteImage(img_new, os.path.join(img_path, img_new_name))

    # add noise
    # for img_name in os.listdir(img_path):
    #     itk_img = sitk.ReadImage(os.path.join(img_path, img_name))
    #     img_arr = sitk.GetArrayFromImage(itk_img)
    #     x, noise_x = gaussian_noise(img_arr)
    #
    #     img_new_name = 'noise_' + img_name
    #
    #     sitk.WriteImage(sitk.GetImageFromArray(x), os.path.join(save_path, img_name))
    #     sitk.WriteImage(sitk.GetImageFromArray(noise_x), os.path.join(save_path, img_new_name))

    # copy
    for img_name in os.listdir(img_path):
        if img_name.endswith('gz'):
            if 'NLST' in img_name:
                continue
            if 'popi' in img_name:
                # 46 copy 1 time to 98
                img_new_name = img_name.split('.nii')[0] + '_copy_1' + '.nii.gz'
                shutil.copyfile(os.path.join(save_path, img_name), os.path.join(save_path, img_new_name))
                print("copy {} to {}".format(os.path.join(save_path, img_name), os.path.join(save_path, img_new_name)))
            elif 'dirlab' in img_name or 'copd' in img_name:
                # 7  copy 14 times to 98
                for i in range(1, 14):
                    img_new_name = img_name.split('.nii')[0] + f'_copy_{i}' + '.nii.gz'
                    shutil.copyfile(os.path.join(save_path, img_name), os.path.join(save_path, img_new_name))
                    print("copy {} to {}".format(os.path.join(save_path, img_name),
                                                 os.path.join(save_path, img_new_name)))
            else:
                # 21 copy 5 to 105
                for i in range(1, 5):
                    img_new_name = img_name.split('.nii')[0] + f'_copy_{i}' + '.nii.gz'
                    shutil.copyfile(os.path.join(save_path, img_name), os.path.join(save_path, img_new_name))
                    print("copy {} to {}".format(os.path.join(save_path, img_name),
                                                 os.path.join(save_path, img_new_name)))

    print('done !')


def NLST_processing(fixed_path, moving_path, nlst_path, **cfg):
    print("NLST: ")

    file_list = sorted([file_name for file_name in os.listdir(nlst_path) if file_name.lower().endswith('.gz')])

    for file_name in file_list:
        target_path = moving_path
        file = os.path.join(nlst_path, file_name)
        # exp -> fixed insp -> moving
        # 0 fixed 1 moving
        file_prefix = file_name[:9]
        file_suffix = file_name.split('_')[2]

        if '0000' in file_suffix:
            target_path = fixed_path

        # open nii
        img_nii = sitk.ReadImage(file)

        img = crop_resampling_resize_clamp(img_nii, cfg['resize'], cfg['crop']
                                           , cfg['spacing'],
                                           cfg['clamp'])
        # save
        make_dir(target_path)
        target_filepath = os.path.join(target_path,
                                       "{}.nii.gz".format(file_prefix))
        # if not os.path.exists(target_filepath):
        sitk.WriteImage(img, target_filepath)
        print('case{} done'.format(file_prefix))

def construct_dirlab_test():
    print("dirlab: ")
    img_folder = f'D:/xxf/dirlab_-1000-500'
    target_test_fixed_path = 'D:/xxf/test_ori_-1000_500/fixed'
    target_test_moving_path = 'D:/xxf/test_ori_-1000_500/moving'
    make_dir(target_test_moving_path)
    make_dir(target_test_fixed_path)
    for item in dirlab_case_cfg.items():
        case = item[0]
        if case < 10:
            img_path = img_folder + f'/case0{case}'
        else:
            img_path = img_folder + f'/case{case}'
        dirlab_test_copy(case=case, m_path=target_test_moving_path, f_path=target_test_fixed_path, file_folder=img_path)

def do_augment():
    pass
    # aug_moving_path = '/home/cqut/project/xxf/val_144/moving'
    # aug_fixed_path = '/home/cqut/project/xxf/val_144/fixed'
    # aug_fixed_path = r'D:\xxf\val_144_192_160_large\fixed'
    # aug_moving_path = r'D:\xxf\val_144_192_160_large\moving'
    #
    # save_path = r'D:\xxf\val_144_192_160_large/fixed'
    # make_dir(save_path)
    # aug(aug_fixed_path, save_path)
    #
    # save_path = r'D:\xxf\val_144_192_160_large/moving'
    # make_dir(save_path)
    # aug(aug_moving_path, save_path)

    ##  test landmarks
    # # load image
    # import SimpleITK as sitk
    #
    # for case in range(1, 11):
    #     print('case %d start' % case)
    #     # m_file_path = '/home/cqut/project/xxf/test_ori/moving/dirlab_case%02d.nii.gz' % case
    #     # f_file_path = '/home/cqut/project/xxf/test_ori/fixed/dirlab_case%02d.nii.gz' % case
    #     m_file_path = r'G:\datasets\registration\test_ori\moving\dirlab_case%02d.nii' % case
    #     f_file_path = r'G:\datasets\registration\test_ori\fixed\dirlab_case%02d.nii' % case
    #
    #     m_file = sitk.GetArrayFromImage(sitk.ReadImage(m_file_path))
    #     f_file = sitk.GetArrayFromImage(sitk.ReadImage(f_file_path))
    #
    #     # load landmark
    #     from utils.utilize import load_landmarks
    #
    #     # landmark_list = load_landmarks('/home/cqut/project/xxf/4DCT-R/data/dirlab')
    #     landmark_list = load_landmarks(r'D:\Project\4DCT-R\data\dirlab')
    #     case_landmark = landmark_list[case-1]
    #     landmark_00 = case_landmark['landmark_00']
    #     landmark_50 = case_landmark['landmark_50']
    #
    #     # crop landmark
    #     crop_range_d = args.dirlab_cfg[case]["crop_range"][0].start
    #     crop_range_h = args.dirlab_cfg[case]["crop_range"][1].start
    #     crop_range_w = args.dirlab_cfg[case]["crop_range"][2].start
    #     landmark_00 = landmark_00 - [crop_range_w, crop_range_h, crop_range_d]
    #     landmark_50 = landmark_50 - [crop_range_w, crop_range_h, crop_range_d]
    #
    #     # mov_lmk_int = np.round(np.flip(landmark_00, axis=1)).astype('int32')
    #     # ref_lmk_int = np.round(np.flip(landmark_50, axis=1)).astype('int32')
    #
    #     # lmk_id = 257
    #     # lm1_mov = landmark_00[lmk_id]
    #     # lm1_ref = landmark_50[lmk_id]
    #     #
    #     # # visual
    #     # from matplotlib import pyplot as plt
    #     #
    #     # fig, ax = plt.subplots(1, 2)
    #     # ax[0].imshow(m_file[lm1_mov[2]], cmap='gray')
    #     # ax[0].scatter([lm1_mov[0]], [lm1_mov[1]], 10, color='red')
    #     # ax[0].set_title('mov')
    #     # ax[1].imshow(f_file[lm1_ref[2]], cmap='gray')
    #     # ax[1].scatter([lm1_ref[0]], [lm1_ref[1]], 10, color='red')
    #     # ax[1].set_title('ref')
    #     # plt.show()
    #
    #     # load flow
    #     # flow_path = '/home/cqut/project/xxf/deformationField.nii.gz'
    #     flow_path = r'G:\datasets\registration\affine\dirlab_case%02d_output' % case
    #     flow_file = os.path.join(flow_path, 'deformationField.nii.gz')
    #
    #     # d,h,w,3
    #     flow_arr = sitk.GetArrayFromImage(sitk.ReadImage(flow_file))
    #     flow = flow_arr.transpose(3, 0, 1, 2)
    #
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
    #
    #     disp_00_50 = (ref_lmk - landmark_00).astype(np.float)
    #     torch.save(disp_00_50, os.path.join(flow_path, 'case%02d_disp_affine.pt' % case))
    #


def construct_dirlab_train():
    target_path = r'D:\xxf\dirlab_train_240'
    make_dir(target_path)
    resize = [96,160,224]
    for item in dirlab_case_cfg.items():
        case = item[0]
        shape = item[1]
        save_path = os.path.join(target_path, 'case%02d' % case)
        make_dir(save_path)

        # crop, resample,resize
        img_path = f'D:/project/xxf/datasets/dirlab/img/Case{case}Pack/Images'
        dirlab_processing(args, save_path, img_path, np.int16, shape, case, resize=resize)

        # make a train set
        m_path = os.path.join(target_path, 'moving')
        f_path = os.path.join(target_path, 'fixed')
        make_dir(m_path)
        make_dir(f_path)
        file_folder = os.path.join(target_path, 'case%02d' % case)
        dirlab_train(file_folder, m_path, f_path)
        print('case %02d done!' % case)

def construct_nlst_data(fixed_path,moving_path):
    make_dir(fixed_path)
    make_dir(moving_path)
    clamp = [-1000, 500]
    crop = None
    spacing = [1.,1.,1.]
    resize = [160,128,160]
    nlst_path = r'E:\datasets\NLST\test'
    NLST_processing(fixed_path, moving_path, nlst_path=nlst_path, resize=resize, crop=crop, clamp=clamp,
                    spacing=spacing)

def construct_emp10(fixed_path,moving_path):
    # emp10
    clamp = [-1000, 500]  # before -900 500
    crop = None
    spacing = None
    resize=[192,192,192]
    emp10_processing(target_fixed_path, target_moving_path, resize=resize, crop=crop, clamp=clamp,
                     spacing=spacing)

if __name__ == '__main__':
    args = get_args()
    project_folder = get_project_path("4DCT").split("4DCT")[0]
    # target_fixed_path = '/home/cqut/project/xxf/train_144/fixed'
    # target_moving_path = '/home/cqut/project/xxf/train_144/moving'
    target_fixed_path = 'D:/xxf/POPI10/fixed'
    target_moving_path = 'D:/xxf/POPI10/moving'
    make_dir(target_moving_path)
    make_dir(target_fixed_path)
    # emp10
    # construct_emp10(target_fixed_path, target_moving_path)

    # #dirlab
    # construct_dirlab_train()
    # construct_dirlab_test()

    # creatis-popi
    popi_processing(target_fixed_path, target_moving_path)
    print()
    # nlst
    # construct_nlst_data(fixed_path=target_fixed_path, moving_path=target_moving_path)
    # aug
    # do_augment()

    # # COPD数据集img转nii.gz
    # print("copd: ")
    # clamp = [-200, 1000]
    # # crop = [slice(70, 470), slice(30, 470), slice(None)]
    # crop = None
    #
    # spacing = [1, 1, 1]
    # for item in copd_case_cfg.items():
    #     case = item[0]
    #     shape = item[1]
    #
    #     fixed_path = f'E:/datasets/copd/copd{case}/copd{case}/copd{case}_eBHCT.img'
    #     moving_path = f'E:/datasets/copd/copd{case}/copd{case}/copd{case}_iBHCT.img'
    #     copd_processing(fixed_path, target_fixed_path, np.int16, shape, case, resize=resize, crop=crop, clamp=clamp,
    #                     spacing=spacing)
    #     copd_processing(moving_path, target_moving_path, np.int16, shape, case, resize=resize, crop=crop, clamp=clamp,
    #                     spacing=spacing)
    #
    # # learn2reg
    # target_fixed_path = r'E:\datasets\registration\l2r_144_192_160\fixed'
    # target_moving_path = r'E:\datasets\registration\l2r_144_192_160\moving'
    # make_dir(target_moving_path)
    # make_dir(target_fixed_path)
    #
    # clamp = [None, 1100]
    # crop = None
    # spacing = [1, 1, 1]
    # learn2reg_processing(target_fixed_path, target_moving_path, resize=resize, crop=crop, clamp=clamp,
    #                      spacing=spacing)
    #
    #

    ## TCIA

    # resize = None
    # spacing = None
    # crop = None
    # clamp = None
    # resize = [144, 256, 256]
    # spacing = [1, 1, 1]
    # crop = [slice(95, 420), slice(120, 400), slice(0,94)]
    # clamp = [-900, 100]
    #
    # tcia_processing(target_fixed_path, target_moving_path, resize=resize, crop=crop, clamp=clamp, spacing=spacing)

    # test TCIA
    # patient_no_list = [114]
    # for patient_no in patient_no_list:
    #     patient_path = r'D:\project\xxf\datasets\other\4D-Lung\%d_HM10395' % patient_no
    #     for scan_time in sorted([sc for sc in os.listdir(patient_path)]):
    #         scan_path = os.path.join(patient_path, scan_time)
    #         dcm_path = os.path.join(scan_path, os.listdir(scan_path)[0])
    #
    #         if len(os.listdir(dcm_path)) < 2:
    #             dcm_path = os.path.join(scan_path, os.listdir(scan_path)[1])
    #
    #         sitk_img = read_dcm_series(dcm_path)
    #
    #         img = crop_resampling_resize_clamp(sitk_img, resize, crop
    #                                            , spacing, clamp)
    #
    #         target_file_path = os.path.join(target_fixed_path,
    #                                         'tcia_case{}_time{}_T{}.nii.gz'.format(patient_no,
    #                                                                                scan_time[:10], 50))
    #         sitk.WriteImage(img, target_file_path)
    #
    # print("TCIA done!!")

    # moving_path = os.path.join(project_folder, f'datasets/registration/moving')
    # fixed_path = os.path.join(project_folder, f'datasets/registration/fixed')
    # make_dir(moving_path)
    # make_dir(fixed_path)
    #
    # imgTomhd(img_path, moving_path, fixed_path, np.int16, shape, case, True)

    # # 真实病例
    # print("patient: ")
    # target_fixed_path = r'E:\datasets\registration\patient_144_192_160\fixed'
    # target_moving_path = r'E:\datasets\registration\patient_144_192_160\moving'
    # make_dir(target_moving_path)
    # make_dir(target_fixed_path)
    #
    # clamp = [None, 500]
    # crop = [slice(70, 430), slice(120, 370), slice(None)]
    # spacing = [1, 1, 1]
    #
    # patient_processing(target_fixed_path, target_moving_path, resize=resize, crop=crop, clamp=clamp,
    #                    spacing=spacing)
    #
    # # learn2reg LungCT
    # target_fixed_path = r'E:\datasets\registration\LungCT_144_192_160\fixed'
    # target_moving_path = r'E:\datasets\registration\LungCT_144_192_160\moving'
    # make_dir(target_moving_path)
    # make_dir(target_fixed_path)
    #
    # clamp = [-1000, 500]
    # crop = None
    # spacing = [1, 1, 1]
    #
    # learn2reg_lungct_processing(target_fixed_path, target_moving_path, resize=resize, crop=crop, clamp=clamp,
    #                             spacing=spacing)
    #
