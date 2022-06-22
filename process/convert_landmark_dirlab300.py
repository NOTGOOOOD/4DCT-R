import numpy as np
import torch
import os
import utils.utilize as ut

project_path = ut.get_project_path("4DCT").split("4DCT-R")[0]
for i in range(1, 11):
    case = i
    landmark_path = os.path.join(project_path, f'datasets/dirlab/Case{case}Pack/ExtremePhases')
    landmark_00_file = os.path.join(landmark_path, f'Case{case}_300_T00_xyz.txt')
    landmark_50_file = os.path.join(landmark_path, f'Case{case}_300_T50_xyz.txt')
    landmark_00 = np.genfromtxt(landmark_00_file, dtype=np.int64) - 1  # change to 0-based indexing
    landmark_50 = np.genfromtxt(landmark_50_file, dtype=np.int64) - 1  # (n, 3), (w, h, d) order in the last dimension
    disp_00_50 = (landmark_50 - landmark_00).astype(np.float32)  # (n, 3)

    landmark = {'landmark_00': landmark_00, 'landmark_50': landmark_50, 'disp_00_50': disp_00_50}
    torch.save(landmark, f'../data/dirlab/Case{case}_300_00_50.pt')

print("done")

