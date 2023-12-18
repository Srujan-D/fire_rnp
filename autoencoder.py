import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer

class FireTransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(FireTransformerModel, self).__init__()
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        self.linear = nn.Linear(d_model, 1)

    def forward(self, src):
        output = self.transformer(src, src)
        output = self.linear(output[-1, :, :])  # Assuming output[-1] is the last step prediction
        return output

# use transformers to predict how fire will look in the future. for 8 timesteps of available fire data, take x timesteps of fire data and predict the next y timesteps of fire data. so x+y=8.

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
        self.size = len(self.data)//self.factor

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
        # print('inp', inp.shape)
        target = self.data[actual_idx + self.seq_length: actual_idx + self.factor]

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
        fire_mean = np.zeros(fire_mean.shape)

        ## plot the input and target
        # for i in range(inp.shape[0]):
        #     plt.imshow(inp[i, 0, :, :])
        #     plt.savefig(f'fire_testing/input_{idx}_{i}.png')
        # plt.imshow(target[0, 0, :, :])
        # plt.savefig(f'fire_testing/input_{idx}_{i+1}.png')
        # plt.close()

        inp = inp.astype(np.float32)
        target = target.astype(np.float32)

        inp = inp.view(self.seq_length, -1).T  # Reshape for transformer input
        target = target.view(self.future, -1).T

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


def train_transformer_model(model, train_dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for idx, inp, target, _ in train_dataloader:
            optimizer.zero_grad()
            output = model(inp)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    

# Define the autoencoder class
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=3, padding=1)

        # Decoder
        self.conv3 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Reshape the input to have 4 dimensions (batch_size * depth, channels, height, width)
        x = x.view(-1, 1, 30, 30)  # Assuming 8 frames are stacked along the depth dimension

        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Decoder
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))

        # output should have 8 frames along the depth dimension
        x = x.view(-1, 1, 8, 30, 30)

        return x


# Define the training function

def train(model, train_loader, num_epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)

    outputs = []
    for epoch in range(num_epochs):
        for (img, _) in train_loader:
            img = img.to(device)
            recon = model(img)
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, float(loss)))
        outputs.append((epoch, img, recon))


# Define the test function

def test(model, test_loader, device):
    criterion = nn.MSELoss()
    test_loss = 0.0
    num_images = 0
    outputs = []
    with torch.no_grad():
        for (img, _) in test_loader:
            img = img.to(device)
            recon = model(img)
            loss = criterion(recon, img)
            test_loss += loss.item() * img.size(0)
            num_images += img.size(0)
            outputs.append((img, recon))

    test_loss = test_loss / num_images
    print('Test loss: {:.4f}\n'.format(test_loss))

    print(len(outputs))
    return outputs


# Define the plotting function

def plot(outputs, num_images=1, model=None, device=None):
    # outputs is a list of tuples (img, recon), and img/recon is of shape (100, 1, 8, 30, 30)
    print('in plot fires')
    for k in range(num_images):
        plt.figure(figsize=(9, 2))
        # print(outputs[0][0].shape)
        # print(outputs[0][1].shape)
    
        imgs = outputs[0][0][k].detach().cpu().numpy()
        recon = outputs[0][1][k].detach().cpu().numpy()

        # print(imgs.shape)
        # print(recon.shape)

        # reshape from 1,8,30,30 to 8,30,30
        imgs = imgs.reshape(8, 30, 30)
        recon = recon.reshape(8, 30, 30)
        
        for i, item in enumerate(imgs):
            if i >= 8: break
            plt.subplot(2, 8, i + 1)
            plt.imshow(item)
            plt.subplot(2, 8, i + 1).set_title('Input')

        for i, item in enumerate(recon):
            if i >= 8: break
            plt.subplot(2, 8, 8 + i + 1)
            plt.imshow(item)
            plt.subplot(2, 8, 8 + i + 1).set_title('Recon')

    plt.savefig('fire_recon.png')

# Define the main function

def main():
    # Hyperparameters
    num_epochs = 10
    batch_size = 128
    learning_rate = 1e-3

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('----device--------', device)

    # load fire data from fire_data/fire_data/fire_map. each fire_i.npy file is 8,30,30 array of fire data for 8 time steps.
    fire_data = []
    for i in range(500):
        fire_data.append(np.load("fire_data/fire_data/fire_map/fire_" + str(i) + ".npy"))

    # divide into train and test
    train_fire_data = fire_data[:400]
    test_fire_data = fire_data[400:]

    # convert to torch tensor
    train_fire_data = torch.stack([torch.from_numpy(arr) for arr in train_fire_data])
    test_fire_data = torch.stack([torch.from_numpy(arr) for arr in test_fire_data])

    # reshape to 1,8,30,30
    train_fire_data = train_fire_data.reshape(400, 1, 8, 30, 30)
    test_fire_data = test_fire_data.reshape(100, 1, 8, 30, 30)

    # convert to float
    train_fire_data = train_fire_data.float()
    test_fire_data = test_fire_data.float()

    # train_loader
    train_loader = torch.utils.data.DataLoader(dataset=list(zip(train_fire_data, torch.zeros(len(train_fire_data)))),  # Assuming you don't need labels
                                            batch_size=batch_size,
                                            shuffle=True)

    # test_loader
    test_loader = torch.utils.data.DataLoader(dataset=list(zip(test_fire_data, torch.zeros(len(test_fire_data)))),  # Assuming you don't need labels
                                            batch_size=batch_size,
                                            shuffle=False)


    # Initialize the model
    model = AutoEncoder().to(device)

    # Train the model
    train(model, train_loader, num_epochs, learning_rate, device)

    # Test the model
    outputs = test(model, test_loader, device)

    # Plot the first 5 input images and then the reconstructed images
    plot(outputs)

if __name__ == '__main__':
    args = get_fire_args()
    if not args.test:
        tensorboard_logger.configure(os.path.join(args.logdir, str(args.runid)))
        print('Logging to {}'.format(args.logdir))
    
    train_dataloader, test_dataloader, eval_data, data_mean = get_fire_dataloaders(args)
    
    dataset_train = FireTransformerDataset(train_dataloader, params=vars(args))
    dataset_test = FireTransformerDataset(test_dataloader, params=vars(args))

    transformer_model = FireTransformerModel(d_model=128, nhead=4, num_layers=2)
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_transformer_model(transformer_model, train_dataloader, optimizer, criterion, epochs=10)

    '''
    For training further a pre-trained RNP
    '''
    
    # trainer.checkpoint = torch.load('fire_logs/2023-11-28-22-13-26-362563/0/499.pt', map_location=torch.device('cpu'))
    # trainer.net.load_state_dict(trainer.checkpoint['model_state_dict'])
    # trainer.optimizer.load_state_dict(trainer.checkpoint['optimizer_state_dict'])

    # trainer.train(args.num_epochs, args.epoch)# + 1)
