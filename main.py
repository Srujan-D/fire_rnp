import numpy as np
import utils
from arguments import get_args
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import tensorboard_logger
from train import Trainer
import os

import scipy.io as sio


class STDataset(Dataset):
    def __init__(self, data, params):
        super(STDataset, self).__init__()
        self.seq_length = params['seq_length']
        self.future = params['future']
        self.random_roi = params['random_roi']
        self.roi_size = params['roi_size']

        if not self.random_roi:
            data = data[:, params['lat_min']:params['lat_max'], params['lon_min']:params['lon_max']]

        # transforming data to (time, channels=1, height, width)
        self.data = np.expand_dims(data, 1)
        self.size = len(self.data)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp = self.data[idx: idx+self.seq_length]
        target = self.data[idx+self.seq_length: idx+self.seq_length+1+self.future]
        if self.random_roi:
            r1,r2,c1,c2 = utils.get_random_roi(*self.data.shape[-2:], self.roi_size)
            inp = inp[:, :, r1:r2, c1:c2]
            target = target[:, :, r1:r2, c1:c2]
        # print('inp.shape, target.shape', inp.shape, target.shape)
        # inp.shape, target.shape (6, 1, 45, 21) (1, 1, 45, 21)
        return idx, inp, target


def get_dataloaders(args):
    # return train and test dataloaders for conv model
    data = None
    if args.data_file != "animal_data.mat":
        print('RUNNING WSPD WEATHER DATA')
        data = utils.load_nc_data(args.data_file, variable='air')
    else:
        print('RUNNING ANIMAL DATA')
        mat_contents = sio.loadmat("animal_data.mat")
        data = mat_contents['formatted_Fss'].astype(np.float32)
        data = np.rollaxis(data, 2)
    print("--------data--------")
    print(data.shape)
    # (894, 73, 144)
    # normalize data between 0 and 1
    if args.normalize_y:
        data = (data - data.min())/(data.max() - data.min())
    data_mean = data.mean(0)
    data = data - data_mean


    # train test split
    #TODO: index some of the data (some timesteps) by adding a time index here
    train_size = int(.8*len(data))
    offset = 24
    train_indices = np.arange(train_size)
    test_indices = np.arange(train_size+offset, len(data)-offset)
    eval_data = data[train_size+offset:len(data)-offset]
    if not args.random_roi:
        eval_data = eval_data[:, args.lat_min:args.lat_max, args.lon_min:args.lon_max]
        data_mean = data_mean[args.lat_min:args.lat_max, args.lon_min:args.lon_max]

    shuffle = False
    if shuffle:
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_batch_size = 64
    test_batch_size = 64

    dataset = STDataset(data=data, params=vars(args))
    train_dataloader = DataLoader(dataset, batch_size=train_batch_size, num_workers=12, sampler=train_sampler)
    test_dataloader = DataLoader(dataset, batch_size=test_batch_size, num_workers=12, sampler=test_sampler)
    print("--------train and test data--------")
    # (894, 1, 45, 21) (894, 1, 45, 21)

    # print(train_dataloader.dataset.data.shape, test_dataloader.dataset.data.shape)
    return train_dataloader, test_dataloader, eval_data, data_mean


if __name__ == '__main__':
    args = get_args()
    if not args.test:
        tensorboard_logger.configure(os.path.join(args.logdir, str(args.runid)))
        print('Logging to {}'.format(args.logdir))

    # first variant - for every timeframe, select k points 
    # later work - work with different k 
    
    train_dataloader, test_dataloader, eval_data, data_mean = get_dataloaders(args)
    trainer = Trainer(train_dataloader, test_dataloader, args=args, eval_data=eval_data, data_mean=data_mean)

    '''
    For training further a pre-trained RNP
    '''
    
    # trainer.checkpoint = torch.load('/home/srujan/CMU/STMGP/crnp/logs/air temp/air_temp_new_model/0/99.pt', map_location=torch.device('cpu'))
    # trainer.net.load_state_dict(trainer.checkpoint['model_state_dict'])
    # trainer.optimizer.load_state_dict(trainer.checkpoint['optimizer_state_dict'])

    trainer.train(args.num_epochs, args.epoch)# + 1)