import platform
import numpy as np
import SimpleITK as sitk
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from utils.processing import data_standardization_0_n


class Dataset(Data.Dataset):
    def __init__(self, moving_files, fixed_files):
        self.moving_files = moving_files
        self.fixed_files = fixed_files

    def __len__(self):
        return len(self.moving_files)

    def __getitem__(self, index):
        m_img = sitk.GetArrayFromImage(sitk.ReadImage(self.moving_files[index]))[np.newaxis, ...]
        m_img = data_standardization_0_n(1, m_img)
        # m_img = np.array(F.softsign(torch.from_numpy(m_img)))

        f_img = sitk.GetArrayFromImage(sitk.ReadImage(self.fixed_files[index]))[np.newaxis, ...]
        f_img = data_standardization_0_n(1, f_img)
        # f_img = np.array(F.softsign(torch.from_numpy(f_img)))
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

        return [m_img, f_img, self.landmark_files[index], m_name]


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

