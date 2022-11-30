import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data
from process.processing import data_standardization_0_n

'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''


class Dataset(Data.Dataset):
    def __init__(self, moving_files, fixed_files):
        # 初始化
        self.moving_files = moving_files
        self.fixed_files = fixed_files

    def __len__(self):
        # 返回数据集的大小
        return len(self.moving_files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        m_img = sitk.GetArrayFromImage(sitk.ReadImage(self.moving_files[index]))[np.newaxis, ...]
        m_img = data_standardization_0_n(1, m_img)

        f_img = sitk.GetArrayFromImage(sitk.ReadImage(self.fixed_files[index]))[np.newaxis, ...]
        f_img = data_standardization_0_n(1, f_img)

        return [m_img, self.moving_files[index]], [f_img, self.fixed_files[index]]


class TestDataset(Data.Dataset):
    def __init__(self, moving_files, fixed_files, landmark_files):
        # 初始化
        self.moving_files = moving_files
        self.fixed_files = fixed_files
        self.landmark_files = landmark_files

    def __len__(self):
        # 返回数据集的大小
        return len(self.moving_files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        m_img = sitk.GetArrayFromImage(sitk.ReadImage(self.moving_files[index]))[np.newaxis, ...]
        m_img = data_standardization_0_n(1, m_img)

        f_img = sitk.GetArrayFromImage(sitk.ReadImage(self.fixed_files[index]))[np.newaxis, ...]
        f_img = data_standardization_0_n(1, f_img)

        return [m_img, self.moving_files[index]], [f_img, self.fixed_files[index]], self.landmark_files[index]
