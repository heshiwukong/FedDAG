import os
import sys
import logging
import numpy as np
import random
import argparse
import csv


def partition_data(targets, K, partition, n_parties, beta, main_prop=0.8, users = None):
    '''
    partition data among clients using non-iid strategy
    '''
    # np.random.seed(seed)
    
    # n_train = dataset.shape[0]
    # y_train = dataset[:,class_id] 
    n_train = len(targets)
    y_train = targets
    parties_dataidx_map = {}


    if partition == "iid-homo" or partition == 'noniid-feature-noisy':
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        parties_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    # each party is allocated a proportion of the samples of each label - - according to Dirichlet distribution.
    elif partition == "noniid-label-dir":
        """
        Distribution-based label heterogeneity
        """
        min_size = 0
        # 每个参与方至少分到10个数据
        min_require_size = 10

        N = len(targets)
        # min_require_size = int(0.1 * N / n_parties)
        # parties_dataidx_map = {}        
        while min_size < min_require_size:
            
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                temp_y = np.array(y_train).flatten().astype(int)
                idx_k = np.where(np.equal(temp_y, k))[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]                
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            parties_dataidx_map[j] = idx_batch[j]

    elif partition[:13] == "noniid-label#":
        '''               
        Quantity-based label Heterogeneity: each party owns data samples of a fixed number of labels
            Randomly assign num different label IDs to each party.
            Randomly and equally divide the samples of each label into the parties which own the label.
        '''
        num = eval(partition[13:])        
        
        if num <= K and num * n_parties >= K:
            pass
        else:
            raise NotImplementedError       

        times=[ 1 for _ in range(K)]
        contain=[]
        # assign label for each party
        for i in range(n_parties):
            current=[i%K]
            if int(i/K) > 0:
                times[i%K] += 1            
            j=1
            while (len(current)<num):
                ind=random.randint(0,K-1)
                if (ind not in current and times[ind] < num * int(n_parties / K)):                    
                    current.append(ind)
                    times[ind]+=1
                j=j+1
                if j>100*num:
                    i = 0
                    while i<K:
                        if times[i] < num * int(n_parties / K):
                            current.append(i)
                            times[i]+=1
                            break
                        i+=1
                    if i==K:
                        break                    
            contain.append(current)
        parties_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}

        for i in range(K):
            temp_y = np.array(y_train).flatten().astype(int)
            idx_k = np.where(np.equal(temp_y, i))[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, times[i])
            ids=0
            for j in range(n_parties):
                if i in contain[j]:
                    parties_dataidx_map[j]=np.append(parties_dataidx_map[j], split[ids])
                    ids+=1
        for i in range(n_parties):
            parties_dataidx_map[i] = parties_dataidx_map[i].tolist()

    elif partition[:22] == "noniid-label-mix-dir-#":
        N = y_train.shape[0]
        num = eval(partition[22:])
        if num <= K and num * n_parties >= K:
            pass
        else:
            raise NotImplementedError(
                f"The classes value k={K} is not enough for the partition fix label num={num}"
            )
                
        times=[ 1 for _ in range(K)]
        contain=[]
        # assign label for each party
        for i in range(n_parties):
            current=[i%K]
            if int(i/K) > 0:
                times[i%K] += 1            
            j=1
            while (len(current)<num):
                ind=random.randint(0,K-1)
                if (ind not in current and times[ind] < num * int(n_parties / K)):                    
                    current.append(ind)
                    times[ind]+=1
                j=j+1
                if j>100*num:
                    i = 0
                    while i<K:
                        if times[i] < num * int(n_parties / K):
                            current.append(i)
                            times[i]+=1
                            break
                        i+=1
                    if i==K:
                        break                    
            contain.append(current)
        parties_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
                
        for i in range(K):
            temp_y = np.array(y_train).flatten().astype(int)
            idx_k = np.where(np.equal(temp_y, i))[0]
            np.random.shuffle(idx_k)

            proportions_k = np.random.dirichlet(np.repeat(beta, times[i]))
            proportions_k = proportions_k + 0.01
            proportions_k = proportions_k/proportions_k.sum()
            proportions_k = (np.cumsum(proportions_k)*len(idx_k)).astype(int)[:-1]

            split = np.split(idx_k, proportions_k)
            
            ids=0
            for j in range(n_parties):
                if i in contain[j]:
                    parties_dataidx_map[j] = np.append(parties_dataidx_map[j], split[ids])
                    ids += 1  
        
        for i in range(n_parties):
            parties_dataidx_map[i] = parties_dataidx_map[i].tolist()             

    elif partition == "noniid-quantity-whole":
        '''
        Quantity-based data heterogeneity
        '''
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:            
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions + 0.01
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(idxs))
            # min_size = min([p*len(idxs) for p in proportions])
        proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs,proportions)
        parties_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

        for i in range(n_parties):
            parties_dataidx_map[i] = parties_dataidx_map[i].tolist()
    
    elif partition == "noniid-quantity-long-tail":
        '''
        Long-tail label heterogeneity
        '''
        if n_parties <= K:
            pass
        else:
            raise NotImplementedError(
                f"The classes value k={K} is not enough"
            )
        
        tail_prop = (1 - main_prop) / (n_parties - 1)
        parties_dataidx_map = {}
        idx_batch = [[] for _ in range(n_parties)]        
        for k in range(K):
            temp_y = np.array(y_train).flatten().astype(int)
            idx_k = np.where(np.equal(temp_y, k))[0]
            np.random.shuffle(idx_k)
            proportions = np.array([tail_prop if j != k%K else main_prop for j in range(n_parties)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]                
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]    
        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            parties_dataidx_map[j] = idx_batch[j]

    elif partition == "noniid-feature-users":
        '''
        User-based feature heterogeneity
        '''
        # u_train = np.zeros(3358,dtype=np.int32)
        # u_train = dataset[:,feature_id]
        u_train = users
        num_user = len(u_train)
        user = np.zeros(num_user+1, dtype=np.int32)
        for i in range(1, num_user+1):
            user[i] = user[i-1] + u_train[i-1]
        no = np.random.permutation(num_user)
        batch_idxs = np.array_split(no, n_parties)
        parties_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}        
        for i in range(n_parties):
            for j in batch_idxs[i]:
                parties_dataidx_map[i]=np.append(parties_dataidx_map[i], np.arange(user[j], user[j+1]))
        
        for i in range(n_parties):
            parties_dataidx_map[i] = parties_dataidx_map[i].tolist()  

    # elif partition == "feature-iid-domain": 
    #     pass  
    else:
        raise NotImplementedError(
                f"The partition scheme={partition} is not implemented yet"
            )    
    return parties_dataidx_map
            
def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts_dict = {}
    net_cls_counts_npy = np.array([])
    # num_classes = int(y_train.max()) + 1
    # get the number of classes
    num_classes = len(np.unique(y_train))
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}        
        net_cls_counts_dict[net_i] = tmp

        tmp_npy = np.zeros(num_classes)
        for i in range(len(unq)):
            tmp_npy[unq[i]] = unq_cnt[i]
        net_cls_counts_npy = np.concatenate((net_cls_counts_npy, tmp_npy), axis=0)
    net_cls_counts_npy = np.reshape(net_cls_counts_npy, (-1, num_classes))
    return net_cls_counts_npy
