import numpy as np
import utils
from arguments import get_fire_args
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import tensorboard_logger
from train import Trainer
import os

import matplotlib.pyplot as plt

import scipy.io as sio

# write data loader for fire data: data/fire_{i}.npy where i is the fire number, each fire has 8 timesteps, so size of each file is 8*N*N where N is the number of gridpoints
# no params needed as its just npy file

class FireDataset(Dataset):
    """
    Dataset class for handling fire data.

    Args:
    - data (numpy.ndarray): The input data containing 8 time stamps.
    """

    def __init__(self, data, params):
        super(FireDataset, self).__init__()
        self.data = data
        self.seq_length = params['seq_length']
        self.future = params['future']
        self.factor = self.seq_length+self.future+1

        self.data = np.expand_dims(data, 1)
        self.size = len(self.data)//(self.seq_length + self.future + 1)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        """
        Args:
        - idx (int): Index of the sample.

        Returns:
        - Tuple[int, torch.Tensor, torch.Tensor]: A tuple containing index, input, and target.
        """
        actual_idx = idx*self.factor
        inp = self.data[actual_idx: actual_idx + self.seq_length]
        target = self.data[actual_idx + self.seq_length: actual_idx + self.seq_length + 1 + self.future]

        ## if input is 0, set it to random number between 100 and 300
        for i in range(inp.shape[0]):
            for j in range(inp.shape[2]):
                for k in range(inp.shape[3]):
                    if inp[i, 0, j, k] == 0:
                        inp[i, 0, j, k] = np.random.rand() * 0.1 + 0.05
                        # inp[i, 0, j, k] = np.random.randint(100, 300)

        ## if target is 0, set it to random number between 100 and 300
        for i in range(target.shape[2]):
            for j in range(target.shape[3]):
                if target[0, 0, i, j] == 0:
                    target[0, 0, i, j] = np.random.rand() * 0.1 + 0.05
                    # target[0, 0, i, j] = np.random.randint(100, 300)

        fire_mean = self.data[actual_idx: actual_idx + self.seq_length + 1 + self.future].mean(0)

        ## plot the input and target
        # for i in range(inp.shape[0]):
        #     plt.imshow(inp[i, 0, :, :])
        #     plt.savefig(f'fire_testing/input_{idx}_{i}.png')
        # plt.imshow(target[0, 0, :, :])
        # plt.savefig(f'fire_testing/input_{idx}_{i+1}.png')
        # plt.close()

        inp = inp.astype(np.float32)
        target = target.astype(np.float32)

        # print('inp.shape, target.shape', inp.shape, target.shape)
        # inp.shape, target.shape (7, 1, 30, 30) (1, 1, 30, 30)

        return idx, inp, target, fire_mean

 
def get_fire_dataloaders(args):
    data = args.file_path
    files = os.listdir(data)
    files = [os.path.join(data, f) for f in files]
    fires = [np.load(f) for f in files]
    # # normalize each fire
    # whole_fire_mean = []
    fire_means = []
    for i, fire in enumerate(fires):
        fire_means.append(fire.mean(0))
        min = np.min(fire)
        max = np.max(fire)
        fires[i] = (fire - min)/(max - min)

    fire_means = np.array(fire_means)
    print(fire_means.shape)
    # for i in range(fires[0].shape[0]):
    #     whole_fire_mean.append(fire_means[i:i+8].mean(0))

    data = np.concatenate(fires, axis=0)
    data_mean = fire_means
    print('data_mean.shape', data_mean.shape)

    # data = np.concatenate([np.load(f) for f in files], axis=0)
    # for i, fire in enumerate(files):
    #     f = np.load(fire)
    #     for j in range(f.shape[0]):
    #         plt.imshow(f[j, :, :])
    #         plt.savefig(f'fire_testing/fire_{i}_{j}.png')
        
    print('data.shape', data.shape, len(data))  # data.shape (800, 30, 30) 800

    # normalize data between 0 and 1
    # print('min max', np.min(data), np.max(data))
    # print('---------normalizing----------')
    # data_mean = data.mean(0)

    # print(data_mean.shape)
    # # if args.normalize_y:
        
    # min = np.min(data)
    # max = np.max(data)
    # data = (data - min)/(max - min)
    # print('min max', np.min(data), np.max(data))

    # print('data_mean', data_mean)
        

    seq_length = args.seq_length
    future = args.future + 1
    train_size = int(0.8 * len(data)//(seq_length + future))
    # offset = 16//(seq_length + future)

    # train_indices = np.arange(train_size)
    train_data = data[:train_size*(seq_length + future)]
    # plot training data
    # print(train_data.shape)
    # for i in range(train_data.shape[0]):
    #     for j in range(8):
    #         plt.imshow(train_data[i*8 + j, :, :])
    #         plt.savefig(f'fire_testing/train_{i}_{j}.png')
    #         plt.close()

    test_data = data[train_size*(seq_length + future):]

    dataset_train = FireDataset(train_data, params=vars(args))
    dataset_test = FireDataset(test_data, params=vars(args))

    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=1)

    print("--------train and test data--------")
    print(train_dataloader.dataset.data.shape, test_dataloader.dataset.data.shape)
    # (800, 1, 30, 30) (800, 1, 30, 30)
    
    return train_dataloader, test_dataloader, test_data, data_mean

if __name__ == '__main__':
    args = get_fire_args()
    if not args.test:
        tensorboard_logger.configure(os.path.join(args.logdir, str(args.runid)))
        print('Logging to {}'.format(args.logdir))

    # first variant - for every timeframe, select k points 
    # later work - work with different k 
    
    train_dataloader, test_dataloader, eval_data, data_mean = get_fire_dataloaders(args)
    trainer = Trainer(train_dataloader, test_dataloader, args=args, eval_data=eval_data, data_mean=data_mean)

    '''
    For training further a pre-trained RNP
    '''
    
    # trainer.checkpoint = torch.load('fire_logs/2023-11-28-22-13-26-362563/0/499.pt', map_location=torch.device('cpu'))
    # trainer.net.load_state_dict(trainer.checkpoint['model_state_dict'])
    # trainer.optimizer.load_state_dict(trainer.checkpoint['optimizer_state_dict'])

    trainer.train(args.num_epochs, args.epoch)# + 1)
