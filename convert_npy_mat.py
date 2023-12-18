import numpy as np
import os
import scipy.io as sio

# load .npy files from fire_data/fire_data/fire_map and convert to .mat files in fire_data/fire_data/fire_map_mat

# # load .npy files from fire_data/fire_data/fire_map
# fire_path = 'fire_data/fire_data/fire_map/'
# fire_files = os.listdir(fire_path)
# fire_files.sort()

# # convert to .mat files in fire_data/fire_data/fire_map_mat
fire_mat_path = 'fire_data/fire_data/fire_map_mat/'

# # create fire_data/fire_data/fire_map_mat if not exist
# if not os.path.exists(fire_mat_path):
#     os.makedirs(fire_mat_path)

# for file in fire_files:
#     fire = np.load(fire_path + file)
#     fire = fire.astype(np.float32)
#     sio.savemat(fire_mat_path + file[:-4] + '.mat', {'fire': fire})
#     print(file[:-4] + '.mat saved')

# load one .mat file to check
fire_mat = sio.loadmat(fire_mat_path + 'fire_1.mat')
print(fire_mat['fire'].shape)
print(fire_mat['fire'])