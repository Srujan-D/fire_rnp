import os
import argparse
import datetime
import torch
import multiprocessing
n_workers = multiprocessing.cpu_count()-10

def get_fire_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file-path', default='fire_data/fire_data/fire_map')
    parser.add_argument('--normalize-y', action='store_true', help='Normalize the target between 0 and 1')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the data')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--num-workers', default=n_workers, type=int, help='Number of workers for data loader')
    parser.add_argument('--num-epochs', default=200, type=int, help='Number of epochs to train for')
    parser.add_argument('--epoch', default=-1, type=int, help='load from this epoch.pt file (-2 means gen data and scratch train, -1 means only scratch train')

    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--save-every', default=5, type=int, help='Save every ... epochs')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--load-model', default=None, help='Load model from a .pth file')
    parser.add_argument('--save-dir', default='models', help='Directory to save models')
    parser.add_argument('--take-points-dir', default='fire_logs/mgp_rnp_logs_10epochs_500samples/', help='Directory to save logs')        # for reusing stmgp points of 500 fire data
    # parser.add_argument('--take-points-dir', default='fire_logs', help='Directory to save logs')
    parser.add_argument('--logdir', default='fire_logs', help='Directory to save logs')
    parser.add_argument('--logid', default=None, type=str, help='unique id for each experiment')

    parser.add_argument('--runid', default=0, type=int, help='unique run id for each experiment')

    parser.add_argument('--log-file', default=None, help='Log file name')
    parser.add_argument('--log-every', default=100, type=int, help='Log every ... batches')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--cuda', action='store_true', help='Use GPU')
    parser.add_argument('--model', default='conv', help='Model type: conv or lstm')
    parser.add_argument('--seq-length', default=7, type=int, help='Sequence length')
    parser.add_argument('--future', default=0, type=int, help='Predicting future')

    parser.add_argument('--hidden-size', default=64, type=int, help='Hidden size of LSTM')
    parser.add_argument('--num-layers', default=2, type=int, help='Number of layers in LSTM')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout ratio')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM')
    parser.add_argument('--num-samples', default=500, type=int, help='Number of samples to generate')

    args = parser.parse_args()

    logid = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") if args.logid is None else str(args.logid)
    args.logdir = os.path.join(args.logdir, str(logid))
    args.runid = str(args.runid)

    if args.log_file is None:
        args.log_file = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')

    if args.cuda and not torch.cuda.is_available():
        print('CUDA is not available, using CPU')
        args.cuda = False

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    return args

def get_args():
    parser = argparse.ArgumentParser()

    # DATASET
    data_file = 'noaa_datasets/air.sig995.mon.mean.nc'
    # data_file = 'animal_data.mat'
    parser.add_argument('--data-file', default=data_file)
    parser.add_argument('--random-roi', action='store_true', help='A random region of interest is sampled at every instant')
    parser.add_argument('--roi-size', default=30, type=int, help='size of region of interest')
    # these are indices, check the data file to see the actual values of them
    parser.add_argument('--lat-min', default=50, type=int, help='minimum latitude index')
    parser.add_argument('--lat-max', default=80, type=int, help='maximum latitude index')
    parser.add_argument('--lon-min', default=25, type=int, help='minimum longtitude index')
    parser.add_argument('--lon-max', default=45, type=int, help='maximum longtitude index')

    # TRAINING
    parser.add_argument('--normalize-y', action='store_true', help='set target value between 0 and 1')
    parser.add_argument('--seq-length', default=6, type=int, help='length of training sequence (default 6)')
    # parser.add_argument('--eval-timesteps', default=12, type=int, help='evaluate on these timesteps')
    parser.add_argument('--num-epochs', default=50*2, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--num-samples', default=10, type=int, help='number of points in each timestep')
    parser.add_argument('--future', default=0, type=int, help='train on 1 + future timesteps')
    parser.add_argument('--epoch', default=-2, type=int, help='load from this epoch.pt file (-2 means gen data and scratch train, -1 means only scratch train')

    # LOGGING
    parser.add_argument('--save-every', default=1, type=int, help='save results every ... epochs')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--logdir', default='logs')
    parser.add_argument('--logid', default=None, type=str, help='unique id for each experiment')
    parser.add_argument('--runid', default=0, type=int, help='unique run id for each experiment')
    
    args = parser.parse_args()
    # setup logdir
    logid = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") if args.logid is None else str(args.logid)
    args.logdir = os.path.join(args.logdir, str(logid))
    args.runid = str(args.runid)

    return args

