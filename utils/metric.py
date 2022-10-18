import torch
import numpy as np


def tre(mov_lmk, ref_lmk, spacing=1):
    # TRE, unit: mm

    diff = (ref_lmk - mov_lmk) * spacing
    diff = torch.Tensor(diff)
    tre = diff.pow(2).sum(1).sqrt()
    mean, std = tre.mean(), tre.std()
    return mean, std, diff


def NCC(real, predict):
    real_copy = np.copy(real)
    predict_copy = np.copy(predict)
    return np.mean(np.multiply((real_copy - np.mean(real_copy)), (predict_copy - np.mean(predict_copy)))) / (
            np.std(real_copy) * np.std(predict_copy))


def MSE(real_copy, predict_copy):
    # return mean_squared_error(real_copy, predict_copy)
    return np.mean(np.square(predict_copy - real_copy))
