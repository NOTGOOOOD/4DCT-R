import os
import SimpleITK as sitk
import numpy as np
import utils.utilize as ut

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


if __name__ == '__main__':
    print(case_cfg.items())
    project_folder = ut.get_project_path("4DCT").split("4DCT")[0]
    for item in case_cfg.items():
        case = item[0]
        shape = item[1]
        img_path = os.path.join(project_folder, f'datasets/dirlab/Case{case}Pack/Images')
        imgTomhd(img_path, np.float16, shape)
