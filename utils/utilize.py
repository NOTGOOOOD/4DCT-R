import os
import torch
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import random
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def show_slice(img_mov):
    fig, ax = plt.subplots(1, 1)
    img_mov = img_mov[0, 0]
    img_shape = int(img_mov.shape[0] / 2)
    ax.imshow(img_mov[img_shape, :, :], cmap='gray')
    # ax[1].imshow(img_mov[:, img_shape, :], cmap='gray')
    # ax[2].imshow(img_mov[:, :, img_shape], cmap='gray')
    plt.show()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def save_png(imgs_numpy, save_path, save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # imgs_numpy = process.processing.data_standardization_0_n(255, imgs_numpy)
    cv2.imwrite(os.path.join(save_path, save_name + ".png"), imgs_numpy)


def save_model(args, model, optimizer, scheduler, time, best_tre):
    model_name = "%s_size%d_lr%s_best_tre%.5f.pth" % (time, args.size, args.lr, best_tre)
    model_checkpoint = os.path.join(args.model_dir, model_name)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None
    }
    torch.save(checkpoint, model_checkpoint)
    print("Saved model checkpoint to [DIR: %s]", args.model_dir)


def save_image(img, ref_img, save_path, save_name):
    make_dir(save_path)

    img = sitk.GetImageFromArray(img.cpu().detach().numpy())
    ref_img = sitk.GetImageFromArray(ref_img.cpu().detach().numpy())

    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(save_path, save_name))


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_project_path(project_name):
    project_path = os.path.abspath(os.path.dirname(__file__))
    root_path = project_path[:project_path.find("{}".format(project_name)) + len("{}".format(project_name))]
    return root_path


def loadfile(filename, datatype='float32', shape=None):
    file_array = np.fromfile(filename, dtype=datatype)
    if shape:
        file_array.reshape(shape)
    return file_array


def loadfileFromFolderToarray(file_folder, datatype, shape=None):
    file_name_list = os.listdir(file_folder)
    file_list = []
    for i, file_name in enumerate(file_name_list):
        file_path = os.path.join(file_folder, file_name)
        if os.path.isdir(file_path):
            # 为dcm文件
            # file = data_processing.readDicomSeries(file_path)
            pass
        else:
            file = np.memmap(file_path, dtype=datatype, mode='r')
            if datatype == np.float16:
                file = file.astype('float32')
        if shape:
            file = file.reshape(shape)

        file_list.append(file)
    files_array = np.array(file_list)

    return files_array


def showimg(image: list, cmap='gray'):
    """
    draw single pic every group
    :param image:
    :param cmap:
    :return:
    """
    length = len(image)
    for i in range(length):
        plt.imshow(image[i][:, 90, :], cmap=cmap)  # D*H*W
        plt.show()


def plotorsave_ct_scan(scan, option: "str", **cfg):
    '''
    画出3D-CT 所有的横断面切片
    :param scan: A NumPy ndarray from a SimpleITK Image
    :param option:plot or save
    :param num_column:
    :param jump: 间隔多少画图
    :param cfg: option=save时启用{head, case, phase ,path, epoch} 图像名 epoch_head_Case_Phase_i_slice
    :return:
    '''
    if torch.is_tensor(scan):
        scan_c = scan.clone().cpu()
        scan_c = scan_c.detach().numpy()
        # print(scan_c.dtype)
    else:
        scan_c = scan

    num_slices = len(scan_c)
    if option == 'plot':
        num_column = 4
        jump = 1
        num_row = (num_slices // jump + num_column - 1) // num_column
        f, plots = plt.subplots(num_row, num_column, figsize=(num_column * 5, num_row * 5))
        for i in range(0, num_row * num_column):
            plot = plots[i % num_column] if num_row == 1 else plots[i // num_column, i % num_column]
            plot.axis('off')
            if i < num_slices // jump:
                plot.imshow(scan_c[i * jump], cmap="gray")

    elif option == 'save':
        case_path = os.path.join(cfg["path"], f"Case{cfg['case']}")
        phase_path = os.path.join(case_path, f"T{cfg['phase']}")
        save_path = os.path.join(phase_path, f"epoch{cfg['epoch']}")
        make_dir(save_path)

        for i in range(0, num_slices):
            img_ndarry = scan_c[i, :, :]
            if np.max(img_ndarry) > 0:
                img_name = f"{cfg['epoch']}_{cfg['head']}_Case{cfg['case']}_T{cfg['phase']}_{i}_slice"
                save_png(img_ndarry, save_path, img_name)
    else:
        ValueError("option: {} ,aug error".format(option))


def transform_convert(img, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    img_tensor = img.clone()
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transform.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img


def load_landmarks(landmark_dir):
    landmark_folder = landmark_dir
    landmarks = []
    for i in sorted(
            [os.path.join(landmark_folder, file) for file in os.listdir(landmark_folder) if file.endswith('.pt')]):
        landmarks.append(torch.load(i))

    return landmarks


if __name__ == '__main__':
    pass
    # simpleITK x, y, z
    # numpy z, y, x

    # case = 3
    # shape = (104, 256, 256)
    # img_path = f'/home/cqut-415/project/xxf/datasets/dirlab/Case{case}Pack/Images'
    # img_array = loadfileToarray(img_path, np.float16, shape)
    # # showimg(img_array, cmap='gray')
    # img = sitk.GetImageFromArray(img_array[0])
    # scan = sitk.GetArrayFromImage(img)
    # print("1")
    # dvf_save_nii("4DCT-R", "result/general_reg/dvf/")
