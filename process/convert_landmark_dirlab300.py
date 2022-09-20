import numpy as np
import torch
import os

from utils.utilize import get_project_path, tre


def convert_landmark(project_path):
    for i in range(1, 11):
        case = i
        landmark_path = os.path.join(project_path, f'datasets/dirlab/Case{case}Pack/ExtremePhases')
        landmark_00_file = os.path.join(landmark_path, f'Case{case}_300_T00_xyz.txt')
        landmark_50_file = os.path.join(landmark_path, f'Case{case}_300_T50_xyz.txt')
        landmark_00 = np.genfromtxt(landmark_00_file, dtype=np.int64) - 1  # change to 0-based indexing
        landmark_50 = np.genfromtxt(landmark_50_file,
                                    dtype=np.int64) - 1  # (n, 3), (w, h, d) order in the last dimension

        disp_00_50 = (landmark_50 - landmark_00).astype(np.float32)  # (n, 3)

        landmark = {'landmark_00': landmark_00, 'landmark_50': landmark_50, 'disp_00_50': disp_00_50}
        torch.save(landmark, f'../data/dirlab/Case{case}_300_00_50.pt')

    print("done")


if __name__ == '__main__':
    case = 1
    project_path = get_project_path("4DCT").split("4DCT")[0]
    # landmark_file = os.path.join(project_path, f'data/dirlab/Case{case}_300_00_50.pt')
    # landmark_info = torch.load(landmark_file)
    # landmark_disp = landmark_info['disp_00_50']  # w, h, d  x,y,z
    # landmark_00 = landmark_info['landmark_00']
    # landmark_50 = landmark_info['landmark_50']
    #
    # res = tre(landmark_00, landmark_50)
    # print(res)
