import matplotlib as mpl
from matplotlib import pylab as plt
import matplotlib.colors as mcolors
import torch
# matplotlib.use('TkAgg')  # If use this line, the show of image will generate a new window out of jupyter.
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pylab

import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D


# pylab.rcParams['figure.figsize'] = (5.0, 3.0)


class MidpointNormalize(mpl.colors.Normalize):
    ## class from the mpl docs:
    # https://matplotlib.org/users/colormapnorms.html

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def data_standardization_0_n(range, img):
    if torch.is_tensor(img):
        return range * (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    else:
        return range * (img - np.min(img)) / (np.max(img) - np.min(img))


fixed_file = r'E:\datasets\registration\patient\fixed\009-4DCT_niii_T00.nii.gz'
img1 = nib.load(fixed_file)
width, height, queue = img1.dataobj.shape
img1 = data_standardization_0_n(1, img1.dataobj)

warped_folder = r'C:\Users\admin\Desktop\experiment\cal_dm\009'
# example2_filename = r'C:\Users\admin\Desktop\experiment\cal_dm\4DCT_nii_T00_lapirn_warped_lv3.nii.gz'
warped_file = [os.path.join(warped_folder, file) for file in os.listdir(warped_folder) if '009' in file]

# 007
# D = 21  # int D dims
# H = 100
# W = 100

# 008
# D = 21  # int D dims
# H = 180
# W = 130

# 009
D = 50  # int D dims
H = 180
W = 130
num = 1
plt.figure(figsize=(15, 10))
for warped in warped_file:
    img2 = nib.load(warped)

    img_arr1 = img1[:, :, D]
    img_arr2 = img2.dataobj[:, :, D, 0, 0] if img2.dataobj.ndim > 3 else img2.dataobj[:, :, D]
    img_arr_diff = img_arr1 - img_arr2

    plt.subplot(6, 1, num)
    img_arr_diff = np.rot90(img_arr_diff, 1)
    plt.title(warped.split('\\')[-1])
    plt.imshow(img_arr_diff, cmap='bwr', norm=MidpointNormalize(vmin=-0.4, vmax=0.4, midpoint=0))
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(fraction=0.046, pad=0.04)

    # plt.subplot(6, 1, num)
    # img_arr_diff = np.rot90(img_arr2, 1)
    # plt.title(warped.split('\\')[-1])
    # plt.imshow(img_arr_diff, 'gray')
    # plt.xticks([])
    # plt.yticks([])

    num += 1

plt.show()
