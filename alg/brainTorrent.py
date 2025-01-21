import copy
import cvxpy as cp
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from alg.fedavg import fedavg

class brainTorrent(fedavg):
    def __init__(self, args, server_model=None, client_model=None, client_weight=None):
        super(brainTorrent, self).__init__(args, server_model, client_model, client_weight)
        self.client_model = client_model
        self.server_model = server_model
        self.witghts = client_weight
        self.version = [[0] * args.n_parties for _ in range(args.n_parties)]
        self.optimizers = [torch.optim.SGD(params=self.client_model[idx].parameters(), lr=args.lr) 
                           for idx in range(args.n_parties)]
        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.args = args
    
    def client_train(self, cur_model, train_local_dl, optimizer):            
        cur_model.to(self.args.device)
        cur_model.train()    
        iterator = iter(train_local_dl)
        for iteration in range(self.args.local_iters):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)
            if x.size(0) == 1:
                continue
            x, target = x.to(self.args.device), target.to(self.args.device)
            optimizer.zero_grad()
            target = target.long()
            out = cur_model(x)
            loss = self.loss_fun(out, target)
            loss.backward()
            optimizer.step()

    def execute(self, SAVE_PATH, train_loaders, test_loaders, val_loaders=None):
        best_epoch = 0
        best_val_acc = [0] * self.args.n_parties
        best_test_acc = [0] * self.args.n_parties        
        save_parties_acc_loss ={} 
        
        for round in range(self.args.global_iters):
            print(f'---------------- Round {round} -----------------')
            # local training
            # random select one clients to update
            cur_id = np.random.randint(0, self.args.n_parties)
            old_v = self.version[cur_id]
            self.version[cur_id][cur_id] += 1            
            new_v = [self.version[i][i] for i in range(self.args.n_parties)]
            
            # aggregate 
            aggre_idx = [i for i in range(self.args.n_parties) if new_v[i] > old_v[i]]
            total_data_points = sum([len(train_loaders[k]) for k in aggre_idx])
            fed_avg_freqs = {k: len(train_loaders[k]) / total_data_points for k in aggre_idx}
            
            with torch.no_grad():
                for key in self.client_model[cur_id].state_dict().keys():
                    if 'num_batches_tracked' in key:
                        pass
                    else:
                        temp = torch.zeros_like(self.client_model[cur_id].state_dict()[key])
                        for i in aggre_idx:
                            temp += self.client_model[i].state_dict()[key] * fed_avg_freqs[i]
                        self.client_model[cur_id].state_dict()[key].data.copy_(temp)

            # local training
            cur_model = self.client_model[cur_id]
            self.client_train(cur_model, train_loaders[cur_id], self.optimizers[cur_id])                   
            
            # update version
            self.version[cur_id] = new_v

            # evaluation 
            if self.args.is_val:
                save_dict = self.eval_print_with_valid(self.args.n_parties, train_loaders, val_loaders, 
                                                    test_loaders, best_val_acc, best_test_acc, best_epoch, round)
                best_epoch = save_dict['best_epoch']
                best_val_acc = save_dict['best_val_acc']
                best_test_acc = save_dict['best_test_acc']
            else:
                save_dict = self.eval_print(self.args.n_parties, train_loaders, test_loaders)         
            
            # save the results
            save_parties_acc_loss[round] = save_dict.copy() 
            save_acc_loss = SAVE_PATH + '_acc_loss'
            with open(save_acc_loss, 'wb') as f:
                pickle.dump(save_parties_acc_loss, f)

