import argparse
import numpy as np
import torch

dirlab_crop_range = [{},
                     {"case": 1,
                      "crop_range": [slice(0, 84), slice(43, 199), slice(10, 250)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (94, 256, 256)
                      },
                     {"case": 2,
                      "crop_range": [slice(5, 101), slice(30, 194), slice(8, 244)],
                      "pixel_spacing": np.array([1.16, 1.16, 2.5], dtype=np.float32),
                      "orign_size": (112, 256, 256)
                      },
                     {"case": 3,
                      "crop_range": [slice(0, 96), slice(42, 210), slice(10, 250)],
                      "pixel_spacing": np.array([1.15, 1.15, 2.5], dtype=np.float32),
                      "orign_size": (104, 256, 256)
                      },
                     {"case": 4,
                      "crop_range": [slice(0, 92), slice(42, 210), slice(10, 250)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (99, 256, 256)
                      },
                     {"case": 5,
                      "crop_range": [slice(0, 92), slice(60, 220), slice(10, 250)],
                      "pixel_spacing": np.array([1.10, 1.10, 2.5], dtype=np.float32),
                      "orign_size": (106, 256, 256)
                      },
                     {"case": 6,
                      "crop_range": [slice(10, 102), slice(144, 328), slice(132, 424)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (128, 512, 512)
                      },
                     {"case": 7,
                      "crop_range": [slice(10, 102), slice(144, 328), slice(114, 422)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (136, 512, 512)
                      },
                     {"case": 8,
                      "crop_range": [slice(18, 118), slice(84, 300), slice(113, 389)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (128, 512, 512)
                      },
                     {"case": 9,
                      "crop_range": [slice(0, 72), slice(126, 334), slice(128, 388)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (128, 512, 512)
                      },
                     {"case": 10,
                      "crop_range": [slice(0, 92), slice(119, 335), slice(140, 384)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (120, 512, 512)
                      }]


def get_args():
    parser = argparse.ArgumentParser()

    # common param
    parser.add_argument("--gpu", type=str, help="gpu id",
                        dest="gpu", default='0')
    parser.add_argument("--model", type=str, help="select model",
                        dest="model", choices=['vm', 'gdir', 'dault-prnet'], default='vm')
    parser.add_argument("--result_dir", type=str, help="results folder",
                        dest="result_dir", default='./result/vm')
    parser.add_argument("--size", type=int, dest="size", default='256')
    parser.add_argument("--initial_channels", type=int, dest="initial_channels", default='16')
    parser.add_argument("--bidir", action='store_true')

    # train param
    parser.add_argument("--train_dir", type=str, help="data folder with training",
                        dest="train_dir", default=r"C:\datasets\registration\train")
    parser.add_argument("--lr", type=float, help="learning rate",
                        dest="lr", default=4e-4)
    parser.add_argument("--n_iter", type=int, help="number of iterations",
                        dest="n_iter", default=500)
    parser.add_argument("--warmup_steps", type=int, dest="warmup_steps", default=20)
    parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc",
                        dest="sim_loss", default='ncc')
    parser.add_argument("--alpha", type=float, help="regularization parameter",
                        dest="alpha", default=1)  # recommend 1.0 for ncc, 0.01 for mse
    parser.add_argument("--batch_size", type=int, help="batch_size",
                        dest="batch_size", default=1)
    parser.add_argument("--n_save_iter", type=int, help="frequency of model saves",
                        dest="n_save_iter", default=100)
    parser.add_argument("--model_dir", type=str, help="models folder",
                        dest="model_dir", default='./Checkpoint')
    parser.add_argument("--log_dir", type=str, help="logs folder",
                        dest="log_dir", default='./Log')
    parser.add_argument("--output_dir", type=str, help="output folder with dvf and warped image",
                        dest="output_dir", default='./output')
    parser.add_argument("--win_size", type=int, help="window size for ncc",
                        dest="win_size", default='5')
    parser.add_argument("--stop_std", type=float, help="early stop",
                        dest="stop_std", default='0.001')
    parser.add_argument("--stop_query_len", type=int, help="early stop",
                        dest="stop_query_len", default='15')

    # test时参数
    parser.add_argument("--test_dir", type=str, help="test data directory",
                        dest="test_dir", default=r'C:\datasets\registration\test_ori')
    parser.add_argument("--landmark_dir", type=str, help="landmark directory",
                        dest="landmark_dir", default=r'D:\project\4DCT\data\dirlab')
    parser.add_argument("--checkpoint_path", type=str, help="model weight file",
                        dest="checkpoint_path", default="./Checkpoint/DIRLAB.pth")

    args = parser.parse_args()
    args.dirlab_cfg = dirlab_crop_range
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    return args
