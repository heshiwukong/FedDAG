import copy
import cvxpy as cp
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from alg.fedavg import fedavg

class pfedgraph(fedavg):
    def __init__(self, args, server_model=None, client_model=None, client_weight=None):
        super(pfedgraph, self).__init__(args, server_model, client_model, client_weight)
        self.client_model = client_model
        self.aggre_model = server_model
        self.args = args
        self.optimizers = [torch.optim.SGD(params=self.client_model[idx].parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg) 
                           for idx in range(args.n_parties)]
        if args.optimizer == 'adam':
            self.optimizers = [torch.optim.SGD(params=self.client_model[idx].parameters(), lr=args.lr, weight_decay=args.reg) 
                           for idx in range(args.n_parties)]
        elif args.optimizer == 'amsgrad':
            self.optimizers = [torch.optim.SGD(params=self.client_model[idx].parameters(), lr=args.lr, weight_decay=args.reg, amsgrad=True) 
                           for idx in range(args.n_parties)]
        elif args.optimizer == 'sgd':
            self.optimizers = [torch.optim.SGD(params=self.client_model[idx].parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg) 
                           for idx in range(args.n_parties)]
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_fun = torch.nn.CrossEntropyLoss()

        self.graph_matrix = torch.ones(len(self.client_model), len(self.client_model)) / (len(self.client_model)-1) # Collaboration Graph
        self.graph_matrix[range(len(self.client_model)), range(len(self.client_model))] = 0        

    def local_train_pfedgraph(self, round, nets_this_round, cluster_models, train_local_dls):
        
        for net_id, net in nets_this_round.items():
            train_local_dl = train_local_dls[net_id]
            optimizer = self.optimizers[net_id]
            
            if round > 0:
                cluster_model = cluster_models[net_id].to(self.args.device)
            net.to(self.args.device)
            net.train()
            iterator = iter(train_local_dl)
            
            for iteration in range(self.args.local_iters):
                try:
                    x, target = next(iterator)
                except StopIteration:
                    iterator = iter(train_local_dl)
                    x, target = next(iterator)
                x, target = x.to(self.args.device).float(), target.to(self.args.device).long()
                if x.size(0) == 1:
                    continue
                optimizer.zero_grad()
                target = target.long()

                out = net(x)
                loss = self.criterion(out, target)
            
                if round > 0:
                    flatten_model = []
                    for param in net.parameters():
                        flatten_model.append(param.reshape(-1))                    
                    flatten_model = torch.cat(flatten_model)
                    loss2 = self.args.lam_pfedgraph * torch.dot(cluster_model, flatten_model) / torch.linalg.norm(flatten_model)
                    # loss2.backward()
                    loss += loss2
                    
                loss.backward()
                optimizer.step()
    
    def cal_model_cosine_difference(self, nets_this_round, initial_global_parameters, dw, similarity_matric):
        model_similarity_matrix = torch.zeros((len(nets_this_round),len(nets_this_round)))
        index_clientid = list(nets_this_round.keys())
        for i in range(len(nets_this_round)):
            model_i = nets_this_round[index_clientid[i]].state_dict()
            for key in dw[index_clientid[i]]:
                dw[index_clientid[i]][key] =  model_i[key] - initial_global_parameters[key]
        for i in range(len(nets_this_round)):
            for j in range(i, len(nets_this_round)):
                if similarity_matric == "all":
                    diff = - torch.nn.functional.cosine_similarity(weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0), weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0))
                    if diff < - 0.9:
                        diff = - 1.0
                    model_similarity_matrix[i, j] = diff
                    model_similarity_matrix[j, i] = diff
                elif  similarity_matric == "fc":
                    diff = - torch.nn.functional.cosine_similarity(weight_flatten(dw[index_clientid[i]]).unsqueeze(0), weight_flatten(dw[index_clientid[j]]).unsqueeze(0))
                    if diff < - 0.9:
                        diff = - 1.0
                    model_similarity_matrix[i, j] = diff
                    model_similarity_matrix[j, i] = diff
        return model_similarity_matrix

    def optimizing_graph_matrix_neighbor(self, graph_matrix, index_clientid, model_difference_matrix, lamba, fed_avg_freqs):
        n = model_difference_matrix.shape[0]
        p = np.array(list(fed_avg_freqs.values()))
        P = lamba * np.identity(n)
        P = cp.atoms.affine.wraps.psd_wrap(P)
        G = - np.identity(n)
        h = np.zeros(n)
        A = np.ones((1, n))
        b = np.ones(1)
        for i in range(model_difference_matrix.shape[0]):
            model_difference_vector = model_difference_matrix[i]
            d = model_difference_vector.numpy()
            q = d - 2 * lamba * p
            x = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                    [G @ x <= h,
                    A @ x == b]
                    )
            prob.solve()

            graph_matrix[index_clientid[i], index_clientid] = torch.Tensor(x.value)
        return graph_matrix

    def update_graph_matrix_neighbor(self, graph_matrix, nets_this_round, initial_global_parameters, dw, fed_avg_freqs, lambda_1, similarity_matric): 
        # index_clientid = torch.tensor(list(map(int, list(nets_this_round.keys()))))     # for example, client 'index_clientid[0]'s model difference vector is model_difference_matrix[0] 
        index_clientid = list(nets_this_round.keys())
        model_difference_matrix = self.cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric)
        graph_matrix = self.optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_difference_matrix, lambda_1, fed_avg_freqs)
        return graph_matrix
    
    def aggregation_by_graph(self, graph_matrix, nets_this_round, global_w):
        with torch.no_grad():
            # 获得模型聚合 以及聚合模型的参数
            tmp_client_state_dict = {}
            cluster_model_vectors = {}
            for client_id in nets_this_round.keys():
                tmp_client_state_dict[client_id] = copy.deepcopy(global_w)
                # cluster_model_vectors[client_id] = torch.zeros_like(weight_flatten_all(global_w))
                params = []
                for param in nets_this_round[client_id].parameters():
                    params.append(param.reshape(-1))                    
                params = torch.cat(params)
                cluster_model_vectors[client_id] = torch.zeros_like(params)

                for key in tmp_client_state_dict[client_id]:
                    tmp_client_state_dict[client_id][key] = torch.zeros_like(tmp_client_state_dict[client_id][key])

            for client_id in nets_this_round.keys():
                tmp_client_state = tmp_client_state_dict[client_id]
                cluster_model_state = cluster_model_vectors[client_id]
                aggregation_weight_vector = graph_matrix[client_id]
                
                for neighbor_id in nets_this_round.keys():
                    net_para = nets_this_round[neighbor_id].state_dict()
                    for key in tmp_client_state:
                        if 'num_batches_tracked' in key:
                            tmp_client_state[key] += (net_para[key] * aggregation_weight_vector[neighbor_id]).long()
                        else:
                            tmp_client_state[key] += net_para[key] * aggregation_weight_vector[neighbor_id]

                for neighbor_id in nets_this_round.keys():
                    net_para = []
                    for param in nets_this_round[neighbor_id].parameters():
                        net_para.append(param.reshape(-1))                    
                    net_para = torch.cat(net_para)
                    # net_para = weight_flatten_all(nets_this_round[neighbor_id].state_dict())
                    cluster_model_state += net_para * (aggregation_weight_vector[neighbor_id] / torch.linalg.norm(net_para))      
            
            for client_id in nets_this_round.keys():
                nets_this_round[client_id].load_state_dict(tmp_client_state_dict[client_id])
        
        return cluster_model_vectors

    def execute(self, SAVE_PATH, train_loaders, test_loaders, val_loaders=None):
        best_epoch = 0
        best_val_acc = [0] * self.args.n_parties
        best_test_acc = [0] * self.args.n_parties        
        save_parties_acc_loss ={}   

        dw = []
        for i in range(self.args.n_parties):
            dw.append({key : torch.zeros_like(value) for key, value in self.client_model[i].named_parameters()})
        global_parameters = self.server_model.state_dict()
        cluster_model_vectors = {}

        total_data_points = sum([len(train_loaders[k]) for k in range(self.args.n_parties)])
        fed_avg_freqs = {k: len(train_loaders[k]) / total_data_points for k in range(self.args.n_parties)}

        for round in range(self.args.global_iters):
            # local training
            nets_this_round = {k: self.client_model[k] for k in range(len(self.client_model))}
            self.local_train_pfedgraph(round, nets_this_round, cluster_model_vectors, train_loaders)
            
            # update matrix and aggregation
            graph_matrix = self.update_graph_matrix_neighbor(self.graph_matrix, nets_this_round, global_parameters, 
                                                        dw, fed_avg_freqs, self.args.alpha, self.args.difference_measure)            
            
            cluster_model_vectors = self.aggregation_by_graph(graph_matrix, nets_this_round, global_parameters)                       
            
            # evaluation 
            if self.args.is_val:
                save_dict = self.eval_print_with_valid(self.args.n_parties, train_loaders, val_loaders, 
                                                    test_loaders, best_val_acc, best_test_acc, best_epoch, round)
                best_epoch = save_dict['best_epoch']
                best_val_acc = save_dict['best_val_acc']
                best_test_acc = save_dict['best_test_acc']
            else:
                save_dict = self.eval_print(self.args.n_parties, train_loaders, test_loaders,round==self.args.global_iters-1)         
            
            # save the results
            save_parties_acc_loss[round] = save_dict.copy() 
            save_acc_loss = SAVE_PATH + '_acc_loss'
            with open(save_acc_loss, 'wb') as f:
                pickle.dump(save_parties_acc_loss, f)

def weight_flatten(model):
    params = []
    for k in model:
        if 'fc' in k:
            params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

# attack
def manipulate_one_model(self, args, net, client_id, global_model=None, indivial_init_model=None):
    print(f'Manipulating Client {client_id}')
    if args.attack_type == 'inv_grad':       # inverse the gradient of client
        start_w = indivial_init_model[client_id].state_dict() if indivial_init_model is not None else global_model.state_dict()
        local_w = net.state_dict()
        local_w = inverse_gradient(start_w, local_w)
        net.load_state_dict(local_w)
    elif args.attack_type == 'shuffle':     # shuffle model parameters
        flat_params = get_flat_params_from(net)
        shuffled_flat_params = flat_params[torch.randperm(len(flat_params))]
        set_flat_params_to(net, shuffled_flat_params)
    elif args.attack_type == 'same_value':
        flat_params = get_flat_params_from(net)
        flat_params = torch.ones_like(flat_params)
        set_flat_params_to(net, flat_params)
    elif args.attack_type == 'sign_flip':
        flat_params = get_flat_params_from(net)
        flat_params = -flat_params
        set_flat_params_to(net, flat_params)
    elif args.attack_type == 'gauss':
        flat_params = get_flat_params_from(net)
        flat_params = torch.normal(0, 1, size=flat_params.shape)
        set_flat_params_to(net, flat_params)

def inverse_gradient(global_w, local_w):
    for key in local_w:
        local_w[key] = global_w[key] - (local_w[key] - global_w[key])
    return local_w

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

