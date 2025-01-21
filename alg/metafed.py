import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from alg.utils import *
from backbone import utils as backbone_utils


class metafed(torch.nn.Module):
    def __init__(self, args, server_model=None, client_model=None, client_weight=None):
        super(metafed, self).__init__()
        self.server_model = server_model
        self.client_model = client_model
        self.client_weight = client_weight
        if server_model is None:
            self.server_model, self.client_model, self.client_weight = backbone_utils.model_select(args, args.device)
        self.optimizers = [optim.SGD(params=self.client_model[idx].parameters(
        ), lr=args.lr) for idx in range(args.n_parties)]
        self.loss_fun = nn.CrossEntropyLoss()
        args.sort = '' 
        for i in range(args.n_parties):  
            args.sort += '%d-' % i  
        args.sort = args.sort[:-1]  
        self.args = args
        self.csort = [int(item) for item in args.sort.split('-')]       

    def init_model_flag(self, train_loaders, val_loaders):
        self.flagl = []
        client_num = self.args.n_parties
        for _ in range(client_num):
            self.flagl.append(False)
        optimizers = [optim.SGD(params=self.client_model[idx].parameters(
        ), lr=self.args.lr) for idx in range(client_num)]
        for idx in range(client_num):
            client_idx = idx
            model, train_loader, optimizer, tmodel, val_loader = self.client_model[
                client_idx], train_loaders[client_idx], optimizers[client_idx], None, val_loaders[idx]
            for _ in range(30):
                _, _ = trainwithteacher(
                    model, train_loader, optimizer, self.loss_fun, self.args.device, tmodel, 1, self.args, False)
            _, val_acc = test(model, val_loader,
                              self.loss_fun, self.args.device)
            if val_acc > self.args.threshold:
                self.flagl[idx] = True
        if self.args.dataset in ['vlcs', 'pacs']:
            self.thes = 0.4
        elif 'medmnist' in self.args.dataset:
            self.thes = 0.5
        elif 'pamap' in self.args.dataset:
            self.thes = 0.5
        else:
            self.thes = 0.5

    def update_flag(self, val_loaders):
        for client_idx, model in enumerate(self.client_model):
            _, val_acc = test(
                model, val_loaders[client_idx], self.loss_fun, self.args.device)
            if val_acc > self.args.threshold:
                self.flagl[client_idx] = True

    def client_train(self, c_idx, dataloader, round):
        client_idx = self.csort[c_idx]
        tidx = self.csort[c_idx-1]
        model, train_loader, optimizer, tmodel = self.client_model[
            client_idx], dataloader, self.optimizers[client_idx], self.client_model[tidx]
        if round == 0 and c_idx == 0:
            tmodel = None
        for _ in range(self.args.local_iters):
            train_loss, train_acc = trainwithteacher(
                model, train_loader, optimizer, self.loss_fun, self.args.device, tmodel, self.args.lam, self.args, self.flagl[client_idx])
        return train_loss, train_acc

    def personalization(self, c_idx, dataloader, val_loader):
        client_idx = self.csort[c_idx]
        model, train_loader, optimizer, tmodel = self.client_model[
            client_idx], dataloader, self.optimizers[client_idx], copy.deepcopy(self.client_model[self.csort[-1]])

        with torch.no_grad():
            _, v1a = test(model, val_loader, self.loss_fun, self.args.device)
            _, v2a = test(tmodel, val_loader, self.loss_fun, self.args.device)

        if v2a <= v1a and v2a < self.thes:
            lam = 0
        else:
            lam = (10**(min(1, (v2a-v1a)*5)))/10*self.args.lam

        for _ in range(self.args.local_iters):
            train_loss, train_acc = trainwithteacher(
                model, train_loader, optimizer, self.loss_fun, self.args.device, tmodel, lam, self.args, self.flagl[client_idx])
        return train_loss, train_acc

    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc = test(
            self.client_model[c_idx], dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc

    def execute(self, SAVE_PATH, train_loaders, test_loaders, val_loaders=None):
        best_epoch = 0
        best_val_acc = [0] * self.args.n_parties
        best_test_acc = [0] * self.args.n_parties        
        save_parties_acc_loss ={}
        # initialize client models by local training
        self.init_model_flag(train_loaders, val_loaders)
        print('Common knowledge accumulation stage')
        for a_iter in range(0, self.args.global_iters-1):
            print(f"============ Train round {a_iter} ============")   
            for c_idx in range(self.args.n_parties):
                self.client_train(c_idx, train_loaders[self.csort[c_idx]], a_iter)
            self.update_flag(val_loaders)
            save_dict = self.eval_print_with_valid(self.args.n_parties, train_loaders, 
                                                   val_loaders, test_loaders, best_val_acc, 
                                                   best_test_acc, best_epoch, a_iter)
            best_epoch = save_dict['best_epoch']
            best_val_acc = save_dict['best_val_acc']
            best_test_acc = save_dict['best_test_acc']
            save_parties_acc_loss[a_iter] = save_dict.copy()
            save_acc_loss = SAVE_PATH + '_acc_loss'
            with open(save_acc_loss, 'wb') as f:
                pickle.dump(save_parties_acc_loss, f)

        print('Personalization stage')
        for c_idx in range(self.args.n_parties):
            self.personalization(c_idx, train_loaders[self.csort[c_idx]], val_loaders[self.csort[c_idx]])

        save_dict = self.eval_print_with_valid(self.args.n_parties, train_loaders, 
                                                val_loaders, test_loaders, best_val_acc, 
                                                best_test_acc, best_epoch, self.args.global_iters)
        best_epoch = save_dict['best_epoch']
        best_val_acc = save_dict['best_val_acc']
        best_test_acc = save_dict['best_test_acc']
        save_parties_acc_loss[self.args.global_iters] = save_dict.copy()
        save_acc_loss = SAVE_PATH + '_acc_loss'
        with open(save_acc_loss, 'wb') as f:
            pickle.dump(save_parties_acc_loss, f)
        
        s = 'Personalized test acc for each client: '
        for item in best_test_acc:
            s += f'{item:.4f},'
        mean_acc_test = np.mean(np.array(best_test_acc))
        s += f'\nAverage accuracy: {mean_acc_test:.4f}'
        print(s)

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
    