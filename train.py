from model import CRNP
import torch
from tensorboard_logger import log_value
import numpy as np
import utils as ut
import os
import ipdb

import matplotlib.pyplot as plt

import concurrent.futures
import matlab.engine
from joblib import Parallel, delayed

class Trainer:
    def __init__(self, train_dataloader, test_dataloader, args, eval_data, data_mean, data_name='fire'):
        np.random.seed()

        self.data_name = data_name
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.eval_data = eval_data
        self.data_mean = data_mean
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.matlab_engine = matlab.engine.start_matlab()

        self.input_size = 2
        self.output_size = 1
        self.net = CRNP(input_size=self.input_size, output_size=self.output_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), args.lr)
        self.args = args
        if self.args.epoch > -1:
            self.checkpoint = torch.load(os.path.join(self.args.logdir, str(self.args.epoch)+'.pt'))
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.min_val = test_dataloader.dataset.data.min()
        self.sz1, self.sz2 = test_dataloader.dataset.data.shape[-2:]
        self.N = self.sz1 * self.sz2
        self.x_grid = ut.generate_grid(self.sz1, self.sz2)
        #self.eng = matlab.engine.start_matlab()
        #self.eng.init()

    def select_rand_robot(self, inp, return_idx=False):
        # inp - num_batches x self.args.seq_length x 1 x self.sz1 x self.sz2
        sz = inp.shape
        # print('inp shape is', sz)
        inp_flat = inp.view(*sz[:-2], -1)
        j = 0
        while j < sz[1]:
            if j > 0:
                idxs_copy = idxs
            idxs = np.random.randint(0, self.N, (sz[0], 1, sz[2], 3))
            agents = np.array([0, 1, 2])
            curr_agent = 0
            i = idxs.shape[-1]
            while i < self.args.num_samples:
                curr_idx = idxs[:,:,:,agents[curr_agent]]
                curr_idx = curr_idx + np.random.randint(-10, 11, (sz[0], 1, sz[2])) + 21*np.random.randint(-22, 23, (sz[0], 1, sz[2]))
                idxs = np.concatenate((idxs, np.expand_dims(curr_idx, axis=-1)), axis=-1)
                agents[curr_agent] = i
                i = i + 1
                curr_agent = curr_agent + 1
                if curr_agent >= len(agents):
                    curr_agent = 0
            idxs = np.clip(idxs, 0, self.N-1)
            if j > 0:
                idxs = np.concatenate((idxs_copy, idxs), axis=1)
            j = j + 1
        idxs = torch.tensor(idxs)
        ys = torch.gather(inp_flat, -1, idxs).permute([0,1,3,2])
        x_all = self.x_grid.unsqueeze(1).expand(sz[0], sz[1], -1, -1)
        xs = x_all.gather(-2, idxs.permute([0,1,3,2]).expand(-1,-1,-1,self.input_size))
        print(xs.shape, ys.shape)
        if return_idx:
            return xs.to(self.device), ys.to(self.device), idxs
        return xs.to(self.device), ys.to(self.device)
    
    def select(self, inp, return_idx=False):
        # # plot input fire intensity for all timesteps
        # for i in range(inp.shape[0]):
        #     for j in range(inp.shape[1]):
        #         _, ax = plt.subplots()
        #         ax.imshow(inp[i,j,0,:,:].cpu().numpy())
        #         plt.savefig('fire_testing/inp_{}_{}.png'.format(i,j))
        #         plt.close()

        # inp - num_batches x self.args.seq_length x 1 x self.sz1 x self.sz2

        # sample self.args.num_samples in each timestep
        sz = inp.shape
        inp_flat = inp.view(*sz[:-2], -1)
        idxs = torch.tensor(np.random.randint(0, self.N, (*sz[:-2], self.args.num_samples)))
        ys = torch.gather(inp_flat, -1, idxs).permute([0,1,3,2])
        x_all = self.x_grid.unsqueeze(1).expand(sz[0], sz[1], -1, -1)
        xs = x_all.gather(-2, idxs.permute([0,1,3,2]).expand(-1,-1,-1,self.input_size))
        if return_idx:
            return xs.to(self.device), ys.to(self.device), idxs
        return xs.to(self.device), ys.to(self.device)
    

    def close_matlab_engine(self):
        self.matlab_engine.quit()

    def matlab_call(self, input, in_shape):
        return np.array(self.matlab_engine.parallel_function(matlab.double(input.tolist()), matlab.double([in_shape])))

    def matlab_select_with_one_engine(self, inp, return_idx=False):
        sz = inp.shape
        inp_flat = inp.view(*sz[:-2], -1)
        matlab_input = np.array(inp_flat)
        matlab_input = matlab_input.reshape(matlab_input.shape[0], matlab_input.shape[1], matlab_input.shape[3])

        results = Parallel(n_jobs=-1)(delayed(self.matlab_call)(matlab_input[i], matlab_input.shape[1]) for i in range(matlab_input.shape[0]))

        results = np.array(results, dtype=np.int64).reshape(matlab_input.shape[0], matlab_input.shape[1], 1, self.args.num_samples)
        idxs = torch.tensor(results)
        ys = torch.gather(inp_flat, -1, idxs).permute([0,1,3,2])
        x_all = self.x_grid.unsqueeze(1).expand(sz[0], sz[1], -1, -1)
        xs = x_all.gather(-2, idxs.permute([0,1,3,2]).expand(-1,-1,-1,self.input_size))

        if return_idx:
            return xs.to(self.device), ys.to(self.device), idxs
        return xs.to(self.device), ys.to(self.device)

    def cleanup(self):
        self.close_matlab_engine()

    
    def matlab_select(self, inp, return_idx=False):
        # inp - num_batches x self.args.seq_length x 1 x self.sz1 x self.sz2
        def matlab_call(input, in_shape):
            eng = matlab.engine.start_matlab()
            idx = eng.main_bot_distribute_window(c)
            eng.quit()
            return np.array(idx)
        # sample self.args.num_samples in each timestep
        sz = inp.shape
        inp_flat = inp.view(*sz[:-2], -1)
        matlab_input = np.array(inp_flat)
        # print('matlab input shape is', matlab_input.shape)
        matlab_input = matlab_input.reshape(matlab_input.shape[0], matlab_input.shape[1], matlab_input.shape[3])
        # print('post reshaping matlab input shape is', matlab_input.shape)

        results = Parallel(n_jobs=-11)(delayed(matlab_call)(matlab_input[i], matlab_input.shape[1]) for i in range(matlab_input.shape[0]))
        # print('------------results---------', np.array(results, dtype=np.int64).shape)
        # print('----------reshaping to---------', matlab_input.shape[0], matlab_input.shape[1], 1, self.args.num_samples)

        # for i in range(matlab_input.shape[0]):
        #     idx = self.eng.main_bot_distribute_window(matlab.double(matlab_input[i].tolist()), matlab.double(matlab_input.shape[1]))
        results = np.array(results, dtype=np.int64).reshape(matlab_input.shape[0], matlab_input.shape[1], 1, self.args.num_samples)
        idxs = torch.tensor(results)
        #idxs = torch.tensor(np.random.randint(0, self.N, (*sz[:-2], self.args.num_samples)))
        ys = torch.gather(inp_flat, -1, idxs).permute([0,1,3,2])
        x_all = self.x_grid.unsqueeze(1).expand(sz[0], sz[1], -1, -1)
        xs = x_all.gather(-2, idxs.permute([0,1,3,2]).expand(-1,-1,-1,self.input_size))
        if return_idx:
            return xs.to(self.device), ys.to(self.device), idxs
        return xs.to(self.device), ys.to(self.device)

    def save_model(self, epoch):
        checkpoint = {'model_state_dict': self.net.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'epoch': epoch,
                      }
        torch.save(checkpoint, os.path.join(self.args.logdir, str(self.args.runid), str(epoch)+'.pt'))

    def train(self, num_epochs=50, curr_epoch=0):
        gen_data_train = True
        if curr_epoch >= -1:
            gen_data_train = False
            curr_epoch = curr_epoch + 1
        if curr_epoch == -2:
            curr_epoch = 0
        step = 0
        for epoch in range(curr_epoch, num_epochs):
            if epoch > 0:

                gen_data_train = False
            
            # for batch_idx, (idx, inp, target) in enumerate(self.train_dataloader):
            for batch_idx, (idx, inp, target, fire_mean) in enumerate(self.train_dataloader):
                self.data_mean = fire_mean

                if gen_data_train == True:
                    # print('input shape', inp.shape)
                    x_context, y_context = self.select(inp)
                    # x_context, y_context = self.select_rand_robot(inp)
                    # x_context, y_context = self.matlab_select(inp)
                    for i, j in enumerate(idx):
                        torch.save(x_context[i], os.path.join(self.args.logdir,'train_xs_' + str(j) + '.pt'))
                        torch.save(y_context[i], os.path.join(self.args.logdir,'train_ys_' + str(j) + '.pt'))
                    continue
                else:
                    # print('going in else')
                    step += 1
                    x_context = []
                    y_context = []
                    for i, j in enumerate(idx):
                        # x_context.append(torch.load(os.path.join(self.args.logdir,'train_xs_' + str(j) + '.pt'), self.device))
                        # y_context.append(torch.load(os.path.join(self.args.logdir,'train_ys_' + str(j) + '.pt'), self.device))
                        
                        ## for resuing Stmgp selected points:
                        x_context.append(torch.load(os.path.join(self.args.take_points_dir,'train_xs_' + str(j) + '.pt'), self.device))
                        y_context.append(torch.load(os.path.join(self.args.take_points_dir,'train_ys_' + str(j) + '.pt'), self.device))
                    x_context = torch.stack(x_context, 0)
                    y_context = torch.stack(y_context, 0)
                x_target = self.x_grid.expand(len(inp),-1,-1).to(self.device)
                y_target = target.view(target.shape[0], -1, 1).to(self.device)
                x_target = x_target.repeat(1, (self.args.future+1), 1)

                self.optimizer.zero_grad()
                mu, sigma, dist, loss = self.net(x_context, y_context, x_target, y_target)
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    mae = torch.sqrt(torch.mean((mu - y_target)**2))
                    real_mae = torch.mean(torch.abs(mu - y_target))

                print('Epoch {:d}/{:d} Batch {:d} Loss {:.3f} RMSE {:.3f} MAE {:.3f}'.format(
                      epoch, self.args.num_epochs, batch_idx, loss.item(), mae.item(), real_mae.item()))
                
                if not self.args.test:
                    log_value('train_rmse', mae, step)
                    log_value('train_loss', loss.item(), step)
                    log_value('train_mae', real_mae, step)

            save = (epoch+1)%self.args.save_every == 0
            self.test(epoch, save=save, gen_data_train=gen_data_train)

    def test(self, epoch, save=False, gen_data_train=True):
        gen_data_test = True
        if epoch > 0 or gen_data_train == False:
            gen_data_test = False
        abs_err = 0
        real_abs_err = 0
        count = 0
        if save:
            print('Saving model and results...')
            self.save_model(epoch)

        with torch.no_grad():
            # for batch_idx, (idx, inp, target) in enumerate(self.train_dataloader):
            for batch_idx, (idx, inp, target, fire_mean) in enumerate(self.test_dataloader):
                self.data_mean = fire_mean
                if gen_data_test == True:
                    x_context, y_context, idxs = self.select(inp, return_idx=True)
                    # x_context, y_context, idxs = self.select_rand_robot(inp, return_idx=True)
                    # x_context, y_context, idxs = self.matlab_select(inp, return_idx=True)
                    for i, j in enumerate(idx):
                        torch.save(x_context[i], os.path.join(self.args.logdir,'test_xs_' + str(j) + '.pt'))
                        torch.save(y_context[i], os.path.join(self.args.logdir,'test_ys_' + str(j) + '.pt'))
                        torch.save(idxs[i], os.path.join(self.args.logdir,'test_idxs_' + str(j) + '.pt'))
                    continue
                else:
                    x_context = []
                    y_context = []
                    idxs = []
                    for i, j in enumerate(idx):
                        # x_context.append(torch.load(os.path.join(self.args.logdir,'test_xs_' + str(j) + '.pt'), self.device))
                        # y_context.append(torch.load(os.path.join(self.args.logdir,'test_ys_' + str(j) + '.pt'), self.device))
                        # idxs.append(torch.load(os.path.join(self.args.logdir,'test_idxs_' + str(j) + '.pt')))
                        
                        # ## for resuing Stmgp selected points:
                        x_context.append(torch.load(os.path.join(self.args.take_points_dir,'test_xs_' + str(j) + '.pt'), self.device))
                        y_context.append(torch.load(os.path.join(self.args.take_points_dir,'test_ys_' + str(j) + '.pt'), self.device))
                        idxs.append(torch.load(os.path.join(self.args.take_points_dir,'test_idxs_' + str(j) + '.pt')))
                    x_context = torch.stack(x_context, 0)
                    y_context = torch.stack(y_context, 0)
                    idxs = torch.stack(idxs, 0)
                x_target = self.x_grid.expand(len(inp),-1,-1).to(self.device)
                y_target = target.view(target.shape[0], -1, 1).to(self.device)
                x_target = x_target.repeat(1, (self.args.future+1), 1)
                
                mu, sigma, dist, loss = self.net(x_context, y_context, x_target, y_target)
                abs_err += torch.sum((mu - y_target)**2)
                real_abs_err += torch.sum(torch.abs(mu - y_target))
                count += len(mu)
                if save:
                    t = np.random.randint(0, len(inp))
                    # t = len(inp) - 1
                    if self.args.future == 0:
                        true = y_target[t].cpu().numpy().squeeze().reshape(self.sz1, self.sz2) + self.data_mean[t].cpu().numpy().squeeze().reshape(self.sz1, self.sz2)
                        pred = mu[t].cpu().numpy().squeeze().reshape(self.sz1, self.sz2) + self.data_mean[t].cpu().numpy().squeeze().reshape(self.sz1, self.sz2)
                        fn = os.path.join(self.args.logdir, self.args.runid, 'epoch_' + str(epoch) + '_' + str(batch_idx) + '.png')
                        inps = self.reconstruct(y_context[t].cpu().numpy(), idxs[t].cpu().numpy(), t)
                        ut.save_image(inps, true, pred, fn, var=None)
                    else:
                        true = y_target[t].cpu().numpy().squeeze().reshape((self.args.future+1), self.sz1, self.sz2) #+ self.data_mean[t].cpu().numpy().squeeze().reshape(self.sz1, self.sz2)
                        pred = mu[t].cpu().numpy().squeeze().reshape((self.args.future+1), self.sz1, self.sz2) #+ self.data_mean[t].cpu().numpy().squeeze().reshape(self.sz1, self.sz2)
                        fn = os.path.join(self.args.logdir, self.args.runid, 'epoch_' + str(epoch) + '_' + str(batch_idx) + '.png')
                        inps = self.reconstruct(y_context[t].cpu().numpy(), idxs[t].cpu().numpy(), t)
                        ut.save_multi_futures(inps, true, pred, fn, var=None)

        if gen_data_test != True:
            test_mae = torch.sqrt(abs_err / (count * self.N))
            real_test_mae = real_abs_err / (count * self.N)
            print('Test RMSE: {:.6f}'.format(test_mae))
            print('Test MAE: {:.6f}'.format(real_test_mae))
            log_value('test_rmse', test_mae, epoch)
            log_value('test_mae', real_test_mae, epoch)

    def reconstruct(self, y_context, idxs, t):
        nb = len(y_context)
        # print('nb', nb)
        canvas = [np.ones(self.N)*self.min_val for _ in range(nb)]
        mean = self.data_mean[t]
        for i in range(nb):
            idx = idxs[i].squeeze()
            canvas[i][idx] = y_context[i].squeeze() + mean.cpu().numpy().squeeze().flatten()[idx]
            canvas[i] = canvas[i].reshape(self.sz1, self.sz2) 
        return canvas
    
    def just_test(self, epoch, save=False, gen_data_train=True, testing_save_results=True, gen_data_test=True):
        gen_data_test = gen_data_test
        if epoch > 0 or gen_data_train == False:
            gen_data_test = False
        abs_err = 0
        real_abs_err = 0
        count = 0
        if save:
            print('Saving model and results...')
            self.save_model(epoch)

        with torch.no_grad():
            for batch_idx, (idx, inp, target, fire_mean) in enumerate(self.test_dataloader):
                self.data_mean = fire_mean
                # print(fire_mean)
                # x_context, y_context, idxs = self.select(inp, return_idx=True)
                # x_context, y_context, idxs = self.matlab_select(inp, return_idx=True)
                if gen_data_test == True:
                    x_context, y_context, idxs = self.select(inp, return_idx=True)
                    # x_context, y_context, idxs = self.select_rand_robot(inp, return_idx=True)
                    # x_context, y_context, idxs = self.matlab_select(inp, return_idx=True)
                    # x_context, y_context, idxs = self.matlab_select_with_one_engine(inp, return_idx=True)
                    if testing_save_results != True:
                        for i, j in enumerate(idx):
                            torch.save(x_context[i], os.path.join(self.args.logdir,'test_xs_' + str(j) + '.pt'))
                            torch.save(y_context[i], os.path.join(self.args.logdir,'test_ys_' + str(j) + '.pt'))
                            torch.save(idxs[i], os.path.join(self.args.logdir,'test_idxs_' + str(j) + '.pt'))
                        continue
                else:
                    if testing_save_results != True:
                        x_context = []
                        y_context = []
                        idxs = []
                        for i, j in enumerate(idx):
                            x_context.append(torch.load(os.path.join(self.args.logdir,'test_xs_' + str(j) + '.pt'), self.device))
                            y_context.append(torch.load(os.path.join(self.args.logdir,'test_ys_' + str(j) + '.pt'), self.device))
                            idxs.append(torch.load(os.path.join(self.args.logdir,'test_idxs_' + str(j) + '.pt')))
                        x_context = torch.stack(x_context, 0)
                        y_context = torch.stack(y_context, 0)
                        idxs = torch.stack(idxs, 0)
                x_target = self.x_grid.expand(len(inp),-1,-1).to(self.device)
                y_target = target.view(target.shape[0], -1, 1).to(self.device)
                
                mu, sigma, dist, loss = self.net(x_context, y_context, x_target, y_target)
                abs_err += torch.sum((mu - y_target)**2)
                real_abs_err += torch.sum(torch.abs(mu - y_target))
                count += len(mu)
                if save:
                    t = np.random.randint(0, len(inp))
                    true = y_target[t].cpu().numpy().squeeze().reshape(self.sz1, self.sz2) + self.data_mean[t].cpu().numpy().squeeze().reshape(self.sz1, self.sz2)
                    pred = mu[t].cpu().numpy().squeeze().reshape(self.sz1, self.sz2) + self.data_mean[t].cpu().numpy().squeeze().reshape(self.sz1, self.sz2)
                    fn = os.path.join(self.args.logdir, self.args.runid, 'epoch_' + str(epoch) + '_' + str(batch_idx) + '.png')
                    inps = self.reconstruct(y_context[t].cpu().numpy(), idxs[t].cpu().numpy(), t)
                    ut.save_image(inps, true, pred, fn, var=None)

        if gen_data_test != True or testing_save_results == True:
            test_mae = torch.sqrt(abs_err / (count * self.N))
            real_test_mae = real_abs_err / (count * self.N)
            print('Test RMSE: {:.6f}'.format(test_mae))
            print('Test MAE: {:.6f}'.format(real_test_mae))
            log_value('test_rmse', test_mae, epoch)
            log_value('test_mae', real_test_mae, epoch)
