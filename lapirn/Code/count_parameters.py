import os
from argparse import ArgumentParser

from CRegNet import CRegNet_lv1, \
    CRegNet_lv2, CRegNet_lv3

parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='../Model/LapIRN_disp_fea7.pth',
                    help="frequency of saving models")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='../Result',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,
                    help="number of start channels")
parser.add_argument("--fixed", type=str,
                    dest="fixed", default='../Data/image_A.nii',
                    help="fixed image")
parser.add_argument("--moving", type=str,
                    dest="moving", default='../Data/image_B.nii',
                    help="moving image")
opt = parser.parse_args()

savepath = opt.savepath
fixed_path = opt.fixed
moving_path = opt.moving
if not os.path.isdir(savepath):
    os.mkdir(savepath)

imgshape_4 = (160 / 4, 192 / 4, 144 / 4)
imgshape_2 = (160 / 2, 192 / 2, 144 / 2)
imgshape = (160, 192, 144)
range_flow = 0.4

start_channel = opt.start_channel

model_lvl1 = CRegNet_lv1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                         range_flow=range_flow).cuda()
model_lvl2 = CRegNet_lv2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                         range_flow=range_flow, model_lvl1=model_lvl1).cuda()

model = CRegNet_lv3(2, 3, start_channel, is_train=False, imgshape=imgshape,
                    range_flow=range_flow, model_lvl2=model_lvl2).cuda()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total number of parameters: ", count_parameters(model))
