import os
import pickle
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from alg.fedavg import fedavg
from alg.utils import train 


class feddng(fedavg):
    def __init__(self, args, server_model=None, client_model=None, client_weight=None):
        super(feddng, self).__init__(args, server_model, client_model, client_weight)
        self.graph = nx.from_numpy_array(np.zeros((args.n_parties, args.n_parties)), 
                                          create_using=nx.DiGraph)        
        self.graphs = [self.graph.copy() for _ in range(args.n_parties)]
        self.cycle = args.cycle 
        self.token = 0 
        self.client_model = client_model
        self.aggre_model = server_model
        self.optimizers = [torch.optim.SGD(params=self.client_model[idx].parameters(), lr=args.lr) 
                           for idx in range(args.n_parties)]
        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.args = args        

    def init_topology(self, dataloaders):
        '''
        initialize the topology of the network based on the size of the dataset of each client 
        args:
            dataloaders: list of dataloaders for each client        
        '''
        n_parties = self.args.n_parties
        data_sizes = [len(dataloader) for dataloader in dataloaders]        
        total_size = sum(data_sizes)

        for i in range(n_parties):
            base_neighbors = max(1, int(data_sizes[i] / total_size * n_parties))            
            random_factor = np.random.randint(0, abs(n_parties - base_neighbors))  
            num_neighbors = min(base_neighbors + random_factor, int(n_parties/2)) 
            num_neighbors = max(1, int(n_parties/2)) 
            
            neighbors = np.random.choice([j for j in range(n_parties) if j != i], size=num_neighbors, replace=False)
            weights = np.random.rand(num_neighbors) 
            weights /= np.sum(weights) 
            for idx, neighbor in enumerate(neighbors):
                # self.graph.add_edge(i, neighbor, weight=weights[idx])
                self.graph.add_edge(neighbor, i, weight=weights[idx])
        self.graphs = [self.graph.copy() for _ in range(n_parties)]

    def init_model(self, dataloaders):
        '''
        initialize the model for each client
        args:
            dataloaders: list of dataloaders for each client
        '''
        for c_idx in range(self.args.n_parties):
            for _ in range(self.args.local_iters):
                train(self.client_model[c_idx], dataloaders[c_idx], self.optimizers[c_idx], self.loss_fun, self.args.device)
              
    def updata_topology(self, cur_party, k=1, m=1, l=1):    
        if self.args.centrality_type == 'betweenness':
            centrality = nx.betweenness_centrality(self.graphs[cur_party])
        elif self.args.centrality_type == 'closeness':
            centrality = nx.closeness_centrality(self.graphs[cur_party])
        elif self.args.centrality_type == 'eigenvector':
            centrality = nx.eigenvector_centrality(self.graphs[cur_party],max_iter=2000)
        elif self.args.centrality_type == 'pagerank':
            centrality = nx.pagerank(self.graphs[cur_party])
        elif self.args.centrality_type == 'degree':
            centrality = nx.degree_centrality(self.graphs[cur_party])
        else:
            raise ValueError("Unsupported centrality type. Choose from 'betweenness', 'degree', 'closeness', 'eigenvector', or 'pagerank'.")

        sorted_indices = sorted(centrality, key=centrality.get, reverse=True)
        initial_candidates = sorted_indices[:k] 

        neighbors = set()
        for node in initial_candidates:
            neighbors.update(self.graphs[cur_party].neighbors(node))
        neighbors = list(neighbors)
        # neighbors_weights = {neighbor: self.graphs[cur_party].degree(neighbor) for neighbor in neighbors}
        # sorted_neighbors = sorted(neighbors_weights, key=neighbors_weights.get, reverse=True)
        # candidate_neighbors = sorted_neighbors[:m]
        candidate_idx = np.random.choice(np.arange(len(neighbors)), size=m, replace=False)
        candidate_neighbors = [neighbors[idx] for idx in candidate_idx]
        candidate_neighbors = list(set(initial_candidates + candidate_neighbors) - set([cur_party]))        

        current_model = self.client_model[cur_party]
        candidate_models = [self.client_model[neighbor] for neighbor in candidate_neighbors]        
        para_current = get_classifier_params(current_model)
        para_candidates = [get_classifier_params(candidate_model) for candidate_model in candidate_models]

        similarities = [F.cosine_similarity(para_current, para_candidate, dim=0).cpu().detach()
                        for para_candidate in para_candidates]
                           
        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)       
        top_l_indices = [s for s in sorted_indices if similarities[s]>0]
        # top_l_indices = top_l_indices[:l]
        final_neighbors = [candidate_neighbors[i] for i in top_l_indices]
        final_similarities = [similarities[i] for i in top_l_indices]
        
        final_similarities = np.array(final_similarities)
        denominator = sum(final_similarities)
        if denominator != 0:
            edges_to_remove = [(u, cur_party) for u in self.graphs[cur_party].predecessors(cur_party)]
            self.graphs[cur_party].remove_edges_from(edges_to_remove)
            weights = final_similarities / denominator
            for n, w in zip(final_neighbors, weights):
                self.graphs[cur_party].add_edge(n, cur_party, weight=w) 
                        
    def update_client(self, c_idx, data_loader, t=0):             
        neighbors = self.graphs[c_idx].in_edges(c_idx, data=True)        
        model = self.client_model[c_idx]
        aggre_model = self.aggre_model        
        with torch.no_grad():
            for key in aggre_model.state_dict().keys():
                    if 'num_batches_tracked' in key:
                        aggre_model.state_dict()[key].data.copy_(model.state_dict()[key])
                    else:
                        temp = torch.zeros_like(aggre_model.state_dict()[key])
                        for src, _ , data in neighbors:
                            weight = data['weight']
                            temp += weight * self.client_model[src].state_dict()[key]
                        aggre_model.state_dict()[key].data.copy_(temp)            
        
        if self.args.model_aggre == 1:
            for local_param, aggre_param in zip(model.parameters(), aggre_model.parameters()):
                local_param.data.copy_(aggre_param.data)                
        elif self.args.model_aggre == 2:
            for local_param, aggre_param in zip(model.features.parameters(), aggre_model.features.parameters()):
                local_param.data.copy_(aggre_param.data)
        elif self.args.model_aggre == 3:
            for local_param, aggre_param in zip(model.classifier.parameters(), aggre_model.classifier.parameters()):
                local_param.data.copy_(aggre_param.data)
        elif self.args.model_aggre == 4:
            if t<int(self.args.global_iters/2):
                for local_param, aggre_param in zip(model.features.parameters(), aggre_model.features.parameters()):
                    local_param.data.copy_(aggre_param.data)
            else:
                for local_param, aggre_param in zip(model.classifier.parameters(), aggre_model.classifier.parameters()):
                    local_param.data.copy_(aggre_param.data)
        elif self.args.model_aggre == 5:
            if t<int(self.args.global_iters/2):
                for local_param, aggre_param in zip(model.classifier.parameters(), aggre_model.classifier.parameters()):
                    local_param.data.copy_(aggre_param.data)
            else:
                for local_param, aggre_param in zip(model.features.parameters(), aggre_model.features.parameters()):
                    local_param.data.copy_(aggre_param.data)
        else:
            raise ValueError("Unsupported model aggregation type. Choose from 1, 2, 3, 4, or 5.")
        
        model.train()
        aggre_model.eval()
        optimizer = self.optimizers[c_idx]
        loss_list = []
        correct_list = []
        
        for _ in range(self.args.local_iters):
            loss_all = 0
            total = 0
            correct = 0            
            for data, target in data_loader:
                if data.size(0) == 1:
                    continue
                optimizer.zero_grad()
                data = data.to(self.args.device).float()
                target = target.to(self.args.device).long()
                output = model(data)

                loss = self.loss_fun(output, target)

                f1 = model.get_sel_fea(data, self.args.plan)
                f2 = aggre_model.get_sel_fea(data, self.args.plan).detach()
                loss += (self.args.lam2 * F.mse_loss(f1, f2))            
                
                # if self.args.is_cosine:
                #     theta_1 = get_classifier_params(model)
                #     theta_2 = get_classifier_params(aggre_model)
                #     loss += ((1-self.args.lam2) * (1 -  F.cosine_similarity(theta_1, theta_2, dim=0)))

                loss_all += loss.item()
                total += target.size(0)
                pred = output.data.max(1)[1]
                correct += pred.eq(target.view(-1)).sum().item()
                loss.backward()
                optimizer.step()
            loss_list.append(loss_all / len(data_loader))
            correct_list.append(correct/total) 
        return np.mean(np.array(loss_list)), np.mean(np.array(correct_list))

    def aggre_topology(self):
        self.graph = nx.from_numpy_array(np.zeros((self.args.n_parties, self.args.n_parties)), 
                                    create_using=nx.DiGraph)
        for i in range(self.args.n_parties):    
            add_edges = self.graphs[i].in_edges(i, data=True)                
            for src, _, data in add_edges: 
                self.graph.add_edge(src, i, weight=data['weight'])            
        for i in range(self.args.n_parties):
            if self.graph.out_degree(i) == 0:
                add_edges = self.graphs[i].in_edges(i, data=True)
                # max_weight = 0
                # for src, _, data in add_edges:
                #     if data['weight'] > max_weight:
                #         max_src = src
                #         max_weight = data['weight']   
                # max_src = np.random.choice([src for src, _, _ in add_edges])
                # max_weight = self.graph.get_edge_data(max_src, i)['weight'] 
                max_src = i
                while max_src == i:
                    max_src = np.random.randint(0, self.args.n_parties) 
                max_weight = 1/self.args.n_parties
                self.graph.add_edge(i, max_src, weight=max_weight)
        # Normalize the weights
        for i in range(self.args.n_parties):
            sum_weights = sum([data['weight'] for _, _, data in self.graph.in_edges(i, data=True)])
            for src, _, data in self.graph.in_edges(i, data=True):
                data['weight'] /= sum_weights
        self.graphs = [self.graph.copy() for _ in range(self.args.n_parties)]                 

    def execute(self, SAVE_PATH, train_loaders, test_loaders, val_loaders=None):
        best_epoch = 0
        best_val_acc = [0] * self.args.n_parties
        best_test_acc = [0] * self.args.n_parties        
        save_parties_acc_loss ={}
        save_graph = {}
        self.init_topology(train_loaders)
        self.init_model(train_loaders)
        
        # initialize the topology
        for i in range(self.args.n_parties):
            num_indegree = self.graphs[i].in_degree(i)
            k = m = max(1, num_indegree)
            self.updata_topology(i, k, m, k)        
        
        # model aggregation
        self.aggre_topology()
        save_graph[0] = self.graph.copy()
        
        # client training
        for t in range(1, self.args.global_iters):
            print(f"============ Train round {t} ============")               
            # train the client models         
            for c_idx in range(self.args.n_parties): 
                num_indegree = self.graphs[c_idx].in_degree(c_idx) 
                if self.args.sparsity == 0:           
                    k = m = 1
                elif self.args.sparsity == 1:
                    k = m = max(1, int(num_indegree/2))
                else:
                    k = m = self.args.n_parties 
                #  [0, 5],[1, 6],[2,7],[3, 8],[4,9]
                if c_idx % self.cycle == self.token:
                    self.updata_topology(c_idx, k, m, k)                    
                self.update_client(c_idx, train_loaders[c_idx], t)                                 
            self.token = (self.token + 1) % self.cycle                 

            # evaluation 
            if self.args.is_val:
                save_dict = self.eval_print_with_valid(self.args.n_parties, train_loaders, val_loaders, 
                                                    test_loaders, best_val_acc, best_test_acc, best_epoch, t)
                best_epoch = save_dict['best_epoch']
                best_val_acc = save_dict['best_val_acc']
                best_test_acc = save_dict['best_test_acc']
            else:
                save_dict = self.eval_print(self.args.n_parties, train_loaders, test_loaders,t==self.args.global_iters-1)
            
            # model aggregation
            self.aggre_topology()             
            
            # save the results
            save_graph[t] = self.graph.copy()
            save_parties_acc_loss[t] = save_dict.copy() 
            save_acc_loss = SAVE_PATH + '_acc_loss'
            with open(save_acc_loss, 'wb') as f:
                pickle.dump(save_parties_acc_loss, f)
            save_graph_path = SAVE_PATH + '_graph'
            with open(save_graph_path, 'wb') as f:
                pickle.dump(save_graph, f)

    def eval_print(self, n_parties, train_loaders, test_loaders,is_last=False): 
        # evaluation on training data
        save_acc_loss = { 
            'tra_acc_list': [], 
            'tra_loss_list': [],
            'test_acc_list': []                                  
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
            print(f' Test site-{client_idx:02d} | Train Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}') 
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


def get_classifier_params(model):
    params = []
    for name, param in model.classifier.named_parameters():
        if 'weight' in name:
            params.append(param.view(-1)) 
    return torch.cat(params)