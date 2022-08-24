import os
import SimpleITK as sitk
import numpy as np
import utils.utilize as ut
from matplotlib import pyplot as plt
import torch

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


def imgTomhd(file_folder, datatype, shape, case):
    for file_name in os.listdir(file_folder):
        file_path = os.path.join(file_folder, file_name)
        file = np.memmap(file_path, dtype=datatype, mode='r')
        # if datatype != np.float32:
        #     file = file.astype('float32')
        if shape:
            file = file.reshape(shape)

        img = sitk.GetImageFromArray(file)

        target_path = os.path.join(os.path.dirname(os.path.dirname(file_folder)), f"Case{case}_mhd")
        if not os.path.exists(target_path):
            os.mkdir(os.path.join(target_path))

        target_filename = os.path.join(target_path, file_name.split('.')[0] + ".mhd")
        if not os.path.exists(target_filename):
            sitk.WriteImage(img, target_filename)

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
    project_folder = ut.get_project_path("4DCT").split("4DCT")[0]
    # read_mhd(os.path.join(project_folder, f'datasets/dirlab/Case1Pack/Images_mhd'))

    # for item in case_cfg.items():
    #     case = item[0]
    #     shape = item[1]
    #     img_path = os.path.join(project_folder, f'datasets/dirlab/Case{case}Pack/Images')
    #     imgTomhd(img_path, np.int16, shape, case)

    # cfg = [
    #     {"case": 1,
    #      "crop_range": [[slice(8, 33), slice(37, 83)], slice(51, 200), [slice(16, 136), slice(144, 250)]],
    #      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32)
    #      },
    #     {"case": 2,
    #      "crop_range": [slice(6, 93), slice(40, 195), [slice(8, 129), slice(135, 252)]],
    #      "pixel_spacing": np.array([1.16, 1.16, 2.5], dtype=np.float32)
    #      },
    #     {"case": 3,
    #      "crop_range": [[slice(0, 34), slice(39, 97)], slice(53, 209), [slice(10, 123), slice(130, 248)]],
    #      "pixel_spacing": np.array([1.15, 1.15, 2.5], dtype=np.float32)
    #      },
    #     {"case": 4,
    #      "crop_range": [[slice(5, 33), slice(36, 90)], slice(45, 209), [slice(10, 113), slice(119, 242)]],
    #      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32)
    #      },
    #     {"case": 5,
    #      "crop_range": [slice(0, 90), slice(60, 223), slice(16, 244)],
    #      "pixel_spacing": np.array([1.10, 1.10, 2.5], dtype=np.float32)
    #      },
    #     {"case": 6,
    #      "crop_range": [slice(14, 107), slice(190, 328), slice(148, 426)],
    #      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32)
    #      },
    #     {"case": 7,
    #      "crop_range": [slice(13, 108), slice(158, 331), slice(120, 413)],
    #      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32)
    #      },
    #     {"case": 8,
    #      "crop_range": [slice(18, 118), slice(94, 299), slice(120, 390)],
    #      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32)
    #      },
    #     {"case": 9,
    #      "crop_range": [slice(0, 70), slice(153, 323), slice(149, 390)],
    #      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32)
    #      },
    #     {"case": 10,
    #      "crop_range": [slice(0, 90), slice(153, 330), slice(154, 382)],
    #      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32)
    #      }
    #
    # ]

    cfg = [
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
         "crop_range": [slice(14, 107), slice(190, 328), slice(148, 426)],
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

    # 10个病人
    for i in range(10):
        case = i + 1
        data_folder = os.path.join(project_folder.split("4DCT")[0], f'datasets/dirlab/Case{case}_mhd/')
        image_file_list = sorted(
            [file_name for file_name in os.listdir(data_folder) if file_name.lower().endswith('mhd')])
        image_list = []

        # 10个呼吸阶段
        for file_name in image_file_list:
            stkimg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_folder, file_name)))

            # if type(cfg[i]["crop_range"][0]) != slice and len(cfg[i]["crop_range"][0]) == 2:
            #     slicearr = cfg[i]["crop_range"][0]
            #     stkimg = np.concatenate((stkimg[slicearr[0], :, :], stkimg[slicearr[1], :, :]), 0)
            #     cfg[i]["crop_range"][0] = slice(len(stkimg))
            #
            # if type(cfg[i]["crop_range"][2]) != slice and len(cfg[i]["crop_range"][2]) == 2:
            #     slicearr = cfg[i]["crop_range"][2]
            #     stkimg = np.concatenate((stkimg[:, :, slicearr[0]], stkimg[:, :, slicearr[1]]), 2)
            #     cfg[i]["crop_range"][2] = slice(len(stkimg[1, 1]))
            #
            stkimg = stkimg[cfg[i]["crop_range"][0], cfg[i]["crop_range"][1], cfg[i]["crop_range"][2]]
            image_list.append(stkimg)

        for i in range(len(image_list)):
            image_list[i] = data_standardization_0_n(255, image_list[i])
            ut.plotorsave_ct_scan(image_list[i], 'save', epoch=0,
                                  head="input_image",
                                  case=case,
                                  phase=i * 10,
                                  path="../result/general_reg/dirlab/slice")

        print(f"{case} converted")
