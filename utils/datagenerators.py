import os
import platform
import numpy as np
import SimpleITK as sitk
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.utils import data as Data

from utils.processing import data_standardization_0_n
from utils.utilize import load_landmarks


class Dataset(Data.Dataset):
    def __init__(self, moving_files, fixed_files):
        self.moving_files = moving_files
        self.fixed_files = fixed_files

    def __len__(self):
        return len(self.moving_files)

    def __getitem__(self, index):
        m_img = sitk.GetArrayFromImage(sitk.ReadImage(self.moving_files[index]))[np.newaxis, ...]
        m_img = data_standardization_0_n(1, m_img)
        f_img = sitk.GetArrayFromImage(sitk.ReadImage(self.fixed_files[index]))[np.newaxis, ...]
        f_img = data_standardization_0_n(1, f_img)

        m_name = self.moving_files[index].split('moving\\')[1] if platform.system().lower() == 'windows' else \
            self.moving_files[index].split('moving/')[1]
        f_name = self.fixed_files[index].split('fixed\\')[1] if platform.system().lower() == 'windows' else \
            self.fixed_files[index].split('fixed/')[1]

        # shape dosen't match
        if m_img.shape != f_img.shape:
            img_tensor = F.interpolate(torch.tensor(m_img).unsqueeze(0), size=f_img.shape[1:],
                                       mode='trilinear',
                                       align_corners=False)

            m_img = np.array(img_tensor)[0, ...]

        if m_name != f_name:
            print("=================================================")
            print("{} is not match {}".format(m_name, f_name))
            print("=================================================")
            raise ValueError

        return [[m_img, m_name], [f_img, f_name]]


class DirLabDataset(Data.Dataset):
    def __init__(self, moving_files, fixed_files, landmark_files=None):
        self.moving_files = moving_files
        self.fixed_files = fixed_files
        self.landmark_files = landmark_files

    def __len__(self):
        return len(self.moving_files)

    def __getitem__(self, index):
        m_img = sitk.GetArrayFromImage(sitk.ReadImage(self.moving_files[index]))[np.newaxis, ...]
        m_img = data_standardization_0_n(1, m_img)

        f_img = sitk.GetArrayFromImage(sitk.ReadImage(self.fixed_files[index]))[np.newaxis, ...]
        f_img = data_standardization_0_n(1, f_img)

        m_name = self.moving_files[index].split('moving\\')[1] if platform.system().lower() == 'windows' else \
            self.moving_files[index].split('moving/')[1]
        f_name = self.fixed_files[index].split('fixed\\')[1] if platform.system().lower() == 'windows' else \
            self.fixed_files[index].split('fixed/')[1]

        if m_name != f_name:
            print("=================================================")
            print("{} is not match {}".format(m_name, f_name))
            print("=================================================")
            raise ValueError

        if self.landmark_files is not None:
            return [m_img, f_img, self.landmark_files[index], m_name]
        else:
            return [m_img, f_img, m_name]


class PatientDataset(Data.Dataset):
    def __init__(self, moving_files, fixed_files):
        self.moving_files = moving_files
        self.fixed_files = fixed_files

    def __len__(self):
        return len(self.moving_files)

    def __getitem__(self, index):
        m_img = sitk.GetArrayFromImage(sitk.ReadImage(self.moving_files[index]))[np.newaxis, ...]
        m_img = data_standardization_0_n(1, m_img)

        f_img = sitk.GetArrayFromImage(sitk.ReadImage(self.fixed_files[index]))[np.newaxis, ...]
        f_img = data_standardization_0_n(1, f_img)

        m_name = self.moving_files[index].split('moving\\')[1] if platform.system().lower() == 'windows' else \
            self.moving_files[index].split('moving/')[1]
        f_name = self.fixed_files[index].split('fixed\\')[1] if platform.system().lower() == 'windows' else \
            self.fixed_files[index].split('fixed/')[1]

        if m_name != f_name:
            print("=================================================")
            print("{} is not match {}".format(m_name, f_name))
            print("=================================================")
            raise ValueError

        return [m_img, f_img, m_name]


def build_dataloader_dirlab(args, mode='train', batch_size=1, num_w=0):
    if mode not in ["train", "val", "test"]:
        raise ValueError("mode not in [train, val, test]")

    if mode=="train":
        fixed_folder = os.path.join(args.train_dir, 'fixed')
        moving_folder = os.path.join(args.train_dir, 'moving')
        f_img_file_list = sorted([os.path.join(fixed_folder, file_name) for file_name in os.listdir(fixed_folder) if
                                  file_name.lower().endswith('.gz')])
        m_img_file_list = sorted([os.path.join(moving_folder, file_name) for file_name in os.listdir(moving_folder) if
                                  file_name.lower().endswith('.gz')])
        train_dataset = Dataset(moving_files=m_img_file_list, fixed_files=f_img_file_list)
        return Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_w)

    elif mode=="val":
        val_fixed_folder = os.path.join(args.val_dir, 'fixed')
        val_moving_folder = os.path.join(args.val_dir, 'moving')
        f_val_list = sorted([os.path.join(val_fixed_folder, file_name) for file_name in os.listdir(val_fixed_folder) if
                             file_name.lower().endswith('.gz')])
        m_val_list = sorted(
            [os.path.join(val_moving_folder, file_name) for file_name in os.listdir(val_moving_folder) if
             file_name.lower().endswith('.gz')])

        val_dataset = Dataset(moving_files=m_val_list, fixed_files=f_val_list)
        return Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_w)
    elif mode=="test":
        if len(args.landmark_dir) > 5:
            landmark_list = load_landmarks(args.landmark_dir)
        else:
            landmark_list = None

        dir_fixed_folder = os.path.join(args.test_dir, 'fixed')
        dir_moving_folder = os.path.join(args.test_dir, 'moving')

        f_dir_file_list = sorted(
            [os.path.join(dir_fixed_folder, file_name) for file_name in os.listdir(dir_fixed_folder) if
             file_name.lower().endswith('.gz')])
        m_dir_file_list = sorted(
            [os.path.join(dir_moving_folder, file_name) for file_name in os.listdir(dir_moving_folder) if
             file_name.lower().endswith('.gz')])
        test_dataset_dirlab = DirLabDataset(moving_files=m_dir_file_list, fixed_files=f_dir_file_list,
                                            landmark_files=landmark_list)
        return Data.DataLoader(test_dataset_dirlab, batch_size=batch_size, shuffle=False,
                                             num_workers=num_w)
    else:
        raise ValueError("mode must be train, val or test")