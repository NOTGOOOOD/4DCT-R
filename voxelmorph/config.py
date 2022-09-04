import argparse

parser = argparse.ArgumentParser()

# 公共参数
parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='0')
parser.add_argument("--landmark_file", type=str, help="landmark file",
                    dest="landmark_file", default='data/dirlab/Case0_300_00_50.pt')
parser.add_argument("--fixed_img", type=str, dest="fixed_img")
parser.add_argument("--model", type=str, help="select model",
                    dest="model", choices=['vm', 'gdir', 'dault-prnet'], default='vm')
parser.add_argument("--result_dir", type=str, help="results folder",
                    dest="result_dir", default='./result/vm')

# train时参数
parser.add_argument("--train_dir", type=str, help="data folder with training vols",
                    dest="train_dir", default="../../Dataset/LPBA40_delineation/delineation_l_norm/train")
parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=5e-3)
parser.add_argument("--n_iter", type=int, help="number of iterations",
                    dest="n_iter", default=1500)
parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc",
                    dest="sim_loss", default='ncc')
parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=0.001)  # recommend 1.0 for ncc, 0.01 for mse
parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=1)
parser.add_argument("--n_save_iter", type=int, help="frequency of model saves",
                    dest="n_save_iter", default=500)
parser.add_argument("--model_dir", type=str, help="models folder",
                    dest="model_dir", default='./Checkpoint')
parser.add_argument("--log_dir", type=str, help="logs folder",
                    dest="log_dir", default='./Log')

# test时参数
parser.add_argument("--test_dir", type=str, help="test data directory",
                    dest="test_dir", default='../../Dataset/LPBA40_delineation/delineation_l_norm/test')
parser.add_argument("--label_dir", type=str, help="label data directory",
                    dest="label_dir", default='../../Dataset/LPBA40_delineation/label')
parser.add_argument("--checkpoint_path", type=str, help="model weight file",
                    dest="checkpoint_path", default="./Checkpoint/LPBA40.pth")

args = parser.parse_args()