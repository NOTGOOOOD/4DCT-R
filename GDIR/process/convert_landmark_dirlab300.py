import numpy as np
import torch
import os

from utils.utilize import get_project_path


def convert_landmark(project_path):
    for i in range(1, 11):
        case = i
        rigid_disp_filename = 'dirlab_case%02d_disp_rigid.pt' % case
        affine_disp_filename = 'dirlab_case%02d_disp_affine.pt' % case

        landmark_path = os.path.join(project_path, 'datasets/dirlab/Case%dPack/ExtremePhases' % case)
        landmark_00_file = os.path.join(landmark_path, 'Case%d_300_T00_xyz.txt' % case)
        landmark_50_file = os.path.join(landmark_path, 'Case%d_300_T50_xyz.txt' % case)
        landmark_00 = np.genfromtxt(landmark_00_file, dtype=np.int64) - 1  # change to 0-based indexing
        landmark_50 = np.genfromtxt(landmark_50_file,
                                    dtype=np.int64) - 1  # (n, 3), (w, h, d) order in the last dimension

        disp_00_50 = (landmark_50 - landmark_00).astype(np.float32)  # (n, 3)
        disp_rigid = torch.load(os.path.join(landmark_path, rigid_disp_filename))
        disp_affine = torch.load(os.path.join(landmark_path, affine_disp_filename))
        landmark = {'landmark_00': landmark_00, 'landmark_50': landmark_50, 'disp_00_50': disp_00_50,
                    'disp_rigid': disp_rigid, 'disp_affine': disp_affine}

        torch.save(landmark, '../../data/dirlab/Case%02d_300_00_50.pt' % case)

    print("done")


if __name__ == '__main__':
    # case = 1
    project_path = get_project_path("4DCT").split("4DCT")[0]
    convert_landmark(project_path)
    # landmark_file = os.path.join(project_path, f'data/dirlab/Case{case}_300_00_50.pt')
    # landmark_info = torch.load(landmark_file)
    # landmark_disp = landmark_info['disp_00_50']  # w, h, d  x,y,z
    # landmark_00 = landmark_info['landmark_00']
    # landmark_50 = landmark_info['landmark_50']
    #
    # res = tre(landmark_00, landmark_50)
    # print(res)
