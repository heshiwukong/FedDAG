import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from backbone import utils as backbone_utils

from alg.utils import train, test, communication


class fedavg(torch.nn.Module): 
    def __init__(self, args, server_model=None, client_model=None, client_weight=None):
        super(fedavg, self).__init__()
        self.server_model = server_model
        self.client_model = client_model
        self.client_weight = client_weight
        if server_model is None:
            self.server_model, self.client_model, self.client_weight = backbone_utils.model_select(args, args.device)
        self.optimizers = [optim.SGD(params=self.client_model[idx].parameters(
        ), lr=args.lr) for idx in range(args.n_parties)]
        self.loss_fun = nn.CrossEntropyLoss()
        self.args = args

    def client_train(self, c_idx, dataloader, **kwargs):
        train_loss, train_acc = train(self.client_model[c_idx], 
                                      dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        return train_loss, train_acc

    def server_aggre(self, server_model=None, client_model=None, client_weight=None): 
        if server_model is not None and client_model is not None and client_weight is not None:
            server_model, client_model = communication(self.args, server_model, client_model, client_weight)    
        else:
            self.server_model, self.client_model = communication(
            self.args, self.server_model, self.client_model, self.client_weight)

    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc = test(
            self.client_model[c_idx], dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc

    def server_eval(self, dataloader):
        train_loss, train_acc = test(
            self.server_model, dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc

    def execute(self, SAVE_PATH, train_loaders, test_loaders, val_loaders=None):
        # train the model
        best_epoch = 0
        best_val_acc = [0] * self.args.n_parties
        best_test_acc = [0] * self.args.n_parties        
        save_parties_acc_loss ={}
        for t in range(self.args.global_iters):
            print(f"============ Train round {t} ============")
            for client_idx in range(self.args.n_parties):
                for _ in range(self.args.local_iters):
                    self.client_train(client_idx, train_loaders[client_idx], round=t)
            self.server_aggre()
            if self.args.is_val:
                save_dict = self.eval_print_with_valid(self.args.n_parties, train_loaders, val_loaders, 
                                                    test_loaders, best_val_acc, best_test_acc, best_epoch, t)
                best_epoch = save_dict['best_epoch']
                best_val_acc = save_dict['best_val_acc']
                best_test_acc = save_dict['best_test_acc']
            else:
                save_dict = self.eval_print(self.args.n_parties, train_loaders, test_loaders,t==self.args.global_iters-1)
            
            save_parties_acc_loss[t] = save_dict.copy()               
            save_acc_loss = SAVE_PATH + '_acc_loss'
            with open(save_acc_loss, 'wb') as f:
                pickle.dump(save_parties_acc_loss, f)
        
    def eval_print(self, n_parties, train_loaders, test_loaders, is_last=False): 
        # evaluation on training data
        save_acc_loss = { 
            'tra_acc_list': [], 
            'tra_loss_list': [],
            'test_acc_list': [],                                      
        }        
        tra_acc_list = [None] * n_parties
        tra_loss_list = [None] * n_parties
        for client_idx in range(n_parties):
            train_loss, train_acc = self.client_eval(client_idx, train_loaders[client_idx])
            tra_acc_list[client_idx] = train_acc
            tra_loss_list[client_idx] = train_loss
            print(f' Site-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        save_acc_loss['tra_acc_list'] = tra_acc_list
        save_acc_loss['tra_loss_list'] = tra_loss_list
        # if is_last:
        test_acc_list = []
        for client_idx in range(n_parties):
            test_loss, test_acc = self.client_eval(client_idx, test_loaders[client_idx])
            test_acc_list.append(test_acc)
            print(f' Test site-{client_idx:02d} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}') 
        print(f'Test acc average: {np.mean(test_acc_list):.4f}') 
        save_acc_loss['test_acc_list'] = test_acc_list          
        return save_acc_loss

    def eval_print_with_valid(self, n_parties, train_loaders, val_loaders, test_loaders, best_val_acc, best_test_acc, best_epoch, global_iter):
        save_acc_loss = { 
            'epoch': global_iter,
            'tra_acc_list': [], 
            'tra_loss_list': [],                   
            'val_acc_list': [],
            'val_loss_list': [],
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc, 
            'best_test_acc': best_test_acc,                     
        }    

        tra_acc_list = [None] * n_parties
        tra_loss_list = [None] * n_parties
        for client_idx in range(n_parties):
            train_loss, train_acc = self.client_eval(client_idx, train_loaders[client_idx])
            tra_acc_list[client_idx] = train_acc
            tra_loss_list[client_idx] = train_loss
            print(f' Site-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        save_acc_loss['tra_acc_list'] = tra_acc_list
        save_acc_loss['tra_loss_list'] = tra_loss_list

        val_acc_list = [None] * n_parties
        val_loss_list = [None] * n_parties
        for client_idx in range(n_parties):
            val_loss, val_acc = self.client_eval(client_idx, val_loaders[client_idx])
            val_acc_list[client_idx] = val_acc
            val_loss_list[client_idx] = val_loss
            print(f' Site-{client_idx:02d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        save_acc_loss['val_acc_list'] = val_acc_list
        save_acc_loss['val_loss_list'] = val_loss_list

        if np.mean(val_acc_list) > np.mean(best_val_acc):
            for client_idx in range(n_parties):
                best_val_acc[client_idx] = val_acc_list[client_idx]            
            best_epoch = global_iter
            save_acc_loss['best_epoch'] = best_epoch
            save_acc_loss['best_val_acc'] = best_val_acc
            
            for client_idx in range(n_parties):
                _, test_acc = self.client_eval(client_idx, test_loaders[client_idx])
                print(f' Test site-{client_idx:02d} | Epoch:{best_epoch} | Test Acc: {test_acc:.4f}')               
                best_test_acc[client_idx] = test_acc
            save_acc_loss['best_test_acc'] = best_test_acc
        print(f'Best Test acc average: {np.mean(best_test_acc):.4f} | Best Epoch: {best_epoch}')
        return save_acc_loss
