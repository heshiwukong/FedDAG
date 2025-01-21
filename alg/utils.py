import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ALGORITHMS = [
    'fedavg',
    'fedprox',
    'fedbn',
    'metafed'
    'feddng'
    'dittow',
    'brainTorrent',
]

def train(model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        if data.size(0) == 1:
            continue
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total


def test(model, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            output = model(data)
            loss = loss_fun(output, target)
            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

        return loss_all / len(data_loader), correct/total


def train_prox(args, model, server_model, data_loader, optimizer, loss_fun, device):
    '''
    fedprox方法的本地训练
    '''
    model.train()  
    loss_all = 0  
    total = 0  
    correct = 0  
    for step, (data, target) in enumerate(data_loader):
        if data.size(0) == 1:
            continue
        data = data.to(device).float()  
        target = target.to(device).long()  

        output = model(data)  
        loss = loss_fun(output, target)  

        if step > 0:
            w_diff = torch.tensor(0., device = device)  
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)

            w_diff = torch.sqrt(w_diff)  
            loss += args.mu / 2. * w_diff  

        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  

        loss_all += loss.item()  
        total += target.size(0)  
        pred = output.data.max(1)[1] 
        correct += pred.eq(target.view(-1)).sum().item()  

    return loss_all / len(data_loader), correct / total  

def trainwithteacher(model, data_loader, optimizer, loss_fun, device, tmodel, lam, args, flag):
    model.train()
    if tmodel:
        tmodel.eval()
        if not flag:
            with torch.no_grad():
                for key in tmodel.state_dict().keys():
                    if 'num_batches_tracked' in key:
                        pass
                    elif args.nosharebn and 'bn' in key:
                        pass
                    else:
                        model.state_dict()[key].data.copy_(
                            tmodel.state_dict()[key])
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        if data.size(0) == 1:
            continue
        optimizer.zero_grad()

        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        f1 = model.get_sel_fea(data, args.plan)
        loss = loss_fun(output, target)
        if flag and tmodel:
            f2 = tmodel.get_sel_fea(data, args.plan).detach()
            loss += (lam*F.mse_loss(f1, f2))
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total

def communication(args, server_model, models, client_weights):
    client_num=len(models)
    with torch.no_grad():
        if args.alg.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.alg.lower()=='fedap':
            tmpmodels=[]
            for i in range(client_num):
                tmpmodels.append(copy.deepcopy(models[i]).to(args.device))
            with torch.no_grad():
                for cl in range(client_num):
                    for key in server_model.state_dict().keys():
                        temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                        for client_idx in range(client_num):
                            temp += client_weights[cl,client_idx] * tmpmodels[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                        if 'bn' not in key:
                            models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models
