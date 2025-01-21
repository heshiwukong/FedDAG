import copy
import cvxpy as cp
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from alg.fedavg import fedavg

class ditto(fedavg):
    def __init__(self, args, server_model=None, client_model=None, client_weight=None):
        super(ditto, self).__init__(args, server_model, client_model, client_weight)
        self.client_model = client_model
        self.server_model = server_model
        self.client_weight = client_weight
        with torch.no_grad():
            self.w_models = [copy.deepcopy(self.server_model) for _ in range(args.n_parties)]
        
        self.args = args
        self.optimizers = [torch.optim.SGD(params=self.w_models[idx].parameters(), lr=args.lr) 
                           for idx in range(args.n_parties)]
        self.p_optimizers = [torch.optim.SGD(params=self.client_model[idx].parameters(), lr=args.lr) 
                           for idx in range(args.n_parties)]
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_fun = torch.nn.CrossEntropyLoss()
    
    def client_train(self, nets_this_round, train_loaders):
        for net_id, net in nets_this_round.items():
            vnet = self.client_model[net_id]
            net = self.w_models[net_id]
            train_local_dl = train_loaders[net_id]
            # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.args.lr, momentum=0.9, weight_decay=self.args.reg)
            # poptimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, vnet.parameters()), lr = self.args.lr, momentum=0.9, weight_decay=self.args.reg)
            poptimizer = self.p_optimizers[net_id]
            optimizer = self.optimizers[net_id]
            
            net.to(self.args.device)
            net.train()
            
            vnet.to(self.args.device)
            vnet.train()
            
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

                out = vnet(x)
                loss = self.criterion(out, target)
                for param_p, param in zip(vnet.parameters(), net.parameters()):
                    loss += ((self.args.lamda / 2) * torch.norm((param - param_p)) ** 2)
                loss.backward()
                poptimizer.step()

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

                out = net(x)
                loss = self.criterion(out, target)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                # for param_p, param in zip(net.parameters(), self.server_model.parameters()):
                #     param_p = param_p - param 
                for key in net.state_dict().keys():
                    if 'num_batches_tracked' in key:
                        net.state_dict()[key] = self.server_model.state_dict()[key]
                    else:
                        temp = torch.zeros_like(net.state_dict()[key])
                        temp = net.state_dict()[key] - self.server_model.state_dict()[key]
                        net.state_dict()[key].data.copy_(temp)


    def execute(self, SAVE_PATH, train_loaders, test_loaders, val_loaders=None):
        best_epoch = 0
        best_val_acc = [0] * self.args.n_parties
        best_test_acc = [0] * self.args.n_parties        
        save_parties_acc_loss ={} 
        total_data_points = sum([len(train_loaders[k]) for k in range(self.args.n_parties)])
        fed_avg_freqs = {k: len(train_loaders[k]) / total_data_points for k in range(self.args.n_parties)}
        self.client_weight = [fed_avg_freqs[k] for k in range(self.args.n_parties)]

        for round in range(self.args.global_iters):
            print(f'---------------- Round {round} -----------------')
            nets_this_round = {k: self.w_models[k] for k in range(self.args.n_parties)}
            # local training
            self.client_train(nets_this_round, train_loaders)                        
            # aggregate: sever_model + dw
            with torch.no_grad(): 
                for key in self.server_model.state_dict().keys():
                    if 'num_batches_tracked' in key:
                        pass
                    else:
                        temp = torch.zeros_like(self.w_models[0].state_dict()[key])
                        for i in range(self.args.n_parties):
                            temp += self.w_models[i].state_dict()[key] * self.client_weight[i]
                        temp += self.server_model.state_dict()[key]
                        self.server_model.state_dict()[key].data.copy_(temp)
                for i in range(self.args.n_parties):
                    self.w_models[i].load_state_dict(self.server_model.state_dict())
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