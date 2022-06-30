import os
import SimpleITK as sitk
import numpy as np
import utils.utilize as ut
from matplotlib import pyplot as plt
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


def imgTomhd(file_folder, datatype, shape):
    for file_name in os.listdir(file_folder):
        file_path = os.path.join(file_folder, file_name)
        file = np.memmap(file_path, dtype=datatype, mode='r')
        if datatype == np.float16:
            file = file.astype('float32')
        if shape:
            file = file.reshape(shape)

        img = sitk.GetImageFromArray(file)

        target_path = os.path.join(os.path.dirname(file_folder), "Images_mhd")
        if not os.path.exists(target_path):
            os.mkdir(os.path.join(target_path))

        target_filename = os.path.join(target_path, file_name.split('.')[0] + ".mhd")
        if not os.path.exists(target_filename):
            sitk.WriteImage(img, target_filename)

    print("{} convert done".format(file_folder))


def data_standardization_0_255(img):
    array = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
    return array
    # ymax = 255
    # ymin = 0
    # xmax = max(map(max, img))  # 进行两次求max值
    # xmin = min(map(min, img))
    # img_standardization_0_255 = np.round((ymax - ymin) * (img - xmin) / (xmax - xmin) + ymin)
    # return img_standardization_0_255


# def affiine(move_img, fix_img, save_path):
#     outs = ants.registration(fix_img, move_img, type_of_transforme='Affine')
#     reg_img = outs['warpedmovout']
#     ants.image_write(reg_img, save_path)

def set_window(img_data, win_width, win_center):
    img_temp = img_data
    min = (2 * win_center - win_width) / 2.0 + 0.5
    max = (2 * win_center + win_width) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    img_temp = ((img_temp - min) * dFactor).astype(np.int)
    img_temp = np.clip(img_temp, 0, 255)
    return img_temp


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

        ut.plotorsave_ct_scan(ct_value, "plot")
        ut.plotorsave_ct_scan(img_arr, "plot")

        mha_img = sitk.GetImageFromArray(img_arr)
        sitk.WriteImage(mha_img, 'test1.mhd')

        test2 = set_window(ct_value, window, level)
        ut.plotorsave_ct_scan(test2, "plot")
        mha_img = sitk.GetImageFromArray(test2)
        sitk.WriteImage(mha_img, 'test2.mhd')

        plt.show()
        break


if __name__ == '__main__':
    print(case_cfg.items())
    project_folder = ut.get_project_path("4DCT").split("4DCT")[0]
    read_mhd(os.path.join(project_folder, f'datasets/dirlab/Case1Pack/Images_mhd'))

    # for item in case_cfg.items():
    #     case = item[0]
    #     shape = item[1]
    #     img_path = os.path.join(project_folder, f'datasets/dirlab/Case{case}Pack/Images')
    #     imgTomhd(img_path, np.float16, shape)
