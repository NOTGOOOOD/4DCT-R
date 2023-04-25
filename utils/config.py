import argparse
import numpy as np
import torch

dirlab_crop_range = [{},
                     {"case": 1,
                      "crop_range": [slice(0, 88), slice(40, 200), slice(10, 250)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (94, 256, 256)
                      },
                     {"case": 2,
                      "crop_range": [slice(5, 101), slice(24, 200), slice(8, 248)],
                      "pixel_spacing": np.array([1.16, 1.16, 2.5], dtype=np.float32),
                      "orign_size": (112, 256, 256)
                      },
                     {"case": 3,
                      "crop_range": [slice(0, 96), slice(42, 210), slice(10, 250)],
                      "pixel_spacing": np.array([1.15, 1.15, 2.5], dtype=np.float32),
                      "orign_size": (104, 256, 256)
                      },
                     {"case": 4,
                      "crop_range": [slice(0, 96), slice(42, 210), slice(10, 250)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (99, 256, 256)
                      },
                     {"case": 5,
                      "crop_range": [slice(0, 96), slice(60, 220), slice(10, 250)],
                      "pixel_spacing": np.array([1.10, 1.10, 2.5], dtype=np.float32),
                      "orign_size": (106, 256, 256)
                      },
                     {"case": 6,
                      "crop_range": [slice(8, 104), slice(144, 328), slice(130, 426)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (128, 512, 512)
                      },
                     {"case": 7,
                      "crop_range": [slice(8, 104), slice(144, 328), slice(112, 424)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (136, 512, 512)
                      },
                     {"case": 8,
                      "crop_range": [slice(16, 120), slice(84, 300), slice(112, 424)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (128, 512, 512)
                      },
                     {"case": 9,
                      "crop_range": [slice(0, 96), slice(126, 334), slice(126, 390)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (128, 512, 512)
                      },
                     {"case": 10,
                      "crop_range": [slice(0, 96), slice(119, 335), slice(138, 386)],
                      "pixel_spacing": np.array([0.97, 0.97, 2.5], dtype=np.float32),
                      "orign_size": (120, 512, 512)
                      }]


def get_args():
    parser = argparse.ArgumentParser()

    # common param
    parser.add_argument("--gpu", type=str, help="gpu id",
                        dest="gpu", default='0')
    parser.add_argument("--model", type=str, help="select model",
                        dest="model", default='vm')
    parser.add_argument("--result_dir", type=str, help="results folder",
                        dest="result_dir", default='./result/vm')
    parser.add_argument("--size", type=int, dest="size", default='144')
    parser.add_argument("--initial_channels", type=int, dest="initial_channels", default=16)  # default 16
    parser.add_argument("--bidir", action='store_true')
    parser.add_argument("--val_dir", type=str, help="data folder with validation",
                        dest="val_dir", default=r"C:\datasets\registration\val")

    # train param
    parser.add_argument("--train_dir", type=str, help="data folder with training",
                        dest="train_dir", default=r"C:\datasets\registration\train")
    parser.add_argument("--lr", type=float, help="learning rate",
                        dest="lr", default=4e-4)
    parser.add_argument("--n_iter", type=int, help="number of iterations",
                        dest="n_iter", default=5000)
    parser.add_argument("--warmup_steps", type=int, dest="warmup_steps", default=50)
    parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc",
                        dest="sim_loss", default='ncc')
    parser.add_argument("--alpha", type=float, help="regularization parameter",
                        dest="alpha", default=1)  # recommend 1.0 for ncc, 0.01 for mse
    parser.add_argument("--batch_size", type=int, help="batch_size",
                        dest="batch_size", default=1)
    parser.add_argument("--n_save_iter", type=int, help="frequency of model saves",
                        dest="n_save_iter", default=1)
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
    parser.add_argument("--checkpoint_path", type=str, help="model weight folder",
                        dest="checkpoint_path", default="./Checkpoint")
    parser.add_argument("--checkpoint_name", type=str, help="model weight name",
                        dest="checkpoint_name", default=None)

    # LapIRN
    parser.add_argument("--iteration_lvl1", type=int,
                        dest="iteration_lvl1", default=10000,
                        help="number of lvl1 iterations")
    parser.add_argument("--iteration_lvl2", type=int,
                        dest="iteration_lvl2", default=10000,
                        help="number of lvl2 iterations")
    parser.add_argument("--iteration_lvl3", type=int,
                        dest="iteration_lvl3", default=10000,
                        help="number of lvl3 iterations")
    parser.add_argument("--antifold", type=float,
                        dest="antifold", default=100,
                        help="Anti-fold loss: suggested range 0 to 1000")
    parser.add_argument("--smooth", type=float,
                        dest="smooth", default=1.0,
                        help="Gradient smooth loss: suggested range 0.1 to 10, diff use 3.5 ")
    parser.add_argument("--freeze_step", type=int,
                        dest="freeze_step", default=2000,
                        help="Number step for freezing the previous level")

    args = parser.parse_args()
    args.dirlab_cfg = dirlab_crop_range
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    return args
