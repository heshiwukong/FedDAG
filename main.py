import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


from utils.data_pre import *
from utils.config import *
from backbone import utils as backbone_utils
from alg.fedavg import fedavg
from alg.metafed import metafed 
from alg.fedprox import fedprox
from alg.fedbn import fedbn
from alg.base import base
from alg.feddng import feddng
from alg.fpl import FPL as fpl
from alg.pfedgraph import pfedgraph
from alg.ditto import ditto
from alg.cfl import cfl
from alg.brainTorrent import brainTorrent


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3, help="Random seed")
    
    # data related
    parser.add_argument('--root_dir', type=str,required=False, default='/home/ttmou/code/FedDNG/data/', 
                        help='The root directory of the data')    
    parser.add_argument('--dataset', type=str, required=False, default="femnist", 
                        help="The dataset to use")
    parser.add_argument('--is_val', action='store_true', 
                        help="Whether to split validation set")
    parser.add_argument('--partition', type=str, default='noniid-label-dir', 
                        help='The partition strategy: iid-homo|noniid-label-dir|noniid-label#|noniid-label-mix-dir-#|noniid-quantity-whole|noniid-quantity-long-tail|noniid-feature-users')
    parser.add_argument('--beta', type=float, default=0.1, 
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--main_prop', type=float, default=0.8, 
                        help='The parameter for the long tail distribution for data partitioning')
    parser.add_argument('--is_noise', action='store_true',
                        help='Whether to add noise to the data')
    parser.add_argument('--noise_level', type=float, default=1, 
                        help='The noise level for the data')
    parser.add_argument('--is_whole_label', type=bool, default=False, 
                        help='Whether to split the whole label data to each party') 
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='The batch size for the Dataloader') 

    # training process related
    parser.add_argument('--n_parties', type=int, default=20,  
                        help='number of workers in federated learning')
    parser.add_argument('--global_iters', type=int, default=50, 
                        help='iterations for communication')
    parser.add_argument('--local_iters', type=int, default=10, 
                        help='optimization iters in local worker between communication')    
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='[cuda | cpu]')
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='learning rate')
    
    # algorithm-specific parameters
    parser.add_argument('--alg', type=str, default='brainTorrent', 
                        help='Algorithm to choose: [base | fedavg | fedbn | fedprox | fedap | metafed | feddng | fpl | pfedgraph | ditto | cfl]')
    parser.add_argument('--mu', type=float, default=1e-3, 
                        help='The hyper parameter for fedprox')
    parser.add_argument('--model_momentum', type=float, default=0.5, 
                        help='hyperparameter for fedap')
    # metafed
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='threshold to use copy or distillation, hyperparmeter for metafed')
    parser.add_argument('--lam', type=float, default=1, 
                        help='init lam, hyperparmeter for metafed')
    parser.add_argument('--plan', type=int, default=0, 
                        help='metafed and fedce: choose the feature type')    
    # feddng
    parser.add_argument('--centrality_type', type=str, default='eigenvector', 
                        help='fedce: centrality type for choosing the influencial node |eigenvector')
    parser.add_argument('--lam2', type=float, default=1.0, 
                        help='fedce: lam2 for the loss regularization')
    parser.add_argument('--cycle', type=float, default=5, 
                        help='fedce: adjust update frequency of the graph')
    parser.add_argument('--model_aggre', type=int, default=3, 
                        help='fedce: method to aggregate the model,1:whole, 2:encoder, 3:classifier')

    # pfedgraph
    parser.add_argument('--reg', type=float, default=1e-5, 
                        help="L2 regularization strength")
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--lam_pfedgraph', type=float, default=1.0, 
                        help='pfedgraph: lam for the loss regularization')
    parser.add_argument('--alpha', type=float, default=0.8, 
                        help='Hyper-parameter to avoid concentration')
    parser.add_argument('--difference_measure', type=str, default='all', 
                        help='How to measure the model difference: all|fc|')
    parser.add_argument('--sparsity', type=int, default=0,
                    help='fedce: sparsity of the graph')
    # ditto
    parser.add_argument('--lamda', type=float, default=1.0, 
                        help="lambda in the objective")
        
    # model related
    parser.add_argument('--nosharebn', action='store_true',
                        help='not share bn')

    # save related
    parser.add_argument('--save_path', type=str, default='/home/ttmou/code/FedDNG/log/Hetero/', 
                        help='path to save the results')    

    args = parser.parse_args() 
    return args

if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)     
    
    exp_folder = f'{args.dataset}/{args.partition}_{args.is_noise}/{args.global_iters}_{args.local_iters}_{args.is_whole_label}'
    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    file_name = f'{args.alg}_{args.seed}_test'
    SAVE_PATH = os.path.join(args.save_path, file_name) 
    if os.path.exists(SAVE_PATH):
        print(f'{SAVE_PATH} already exists!')
        exit(0)

    # # prepare the data
    if args.dataset in ['vlcs', 'pacs', 'off_home']:
        args = img_param_init(args)
    elif args.dataset in ['covid19']:
        args.num_classes = 4
    if args.is_val:
        train_loaders, val_loaders, test_loaders = get_data(args.dataset)(args)
    else:
        train_loaders, test_loaders = get_data(args.dataset)(args) 
        val_loaders = None
    args.n_parties = len(train_loaders)
    
    # model selection
    server_model, clients_models, client_weights = backbone_utils.model_select(args, args.device)    

    # algorithm selection
    if args.alg == 'fedavg':
        alg_class = fedavg(args, server_model, clients_models, client_weights)
        alg_class.execute(SAVE_PATH, train_loaders, test_loaders, val_loaders)
    elif args.alg == 'fedbn':
        alg_class = fedbn(args, server_model, clients_models, client_weights)
        alg_class.execute(SAVE_PATH, train_loaders, test_loaders, val_loaders)
    elif args.alg == 'fedprox':
        alg_class = fedprox(args, server_model, clients_models, client_weights)
        alg_class.execute(SAVE_PATH, train_loaders, test_loaders, val_loaders)
    elif args.alg == 'metafed':
        alg_class = metafed(args, server_model, clients_models, client_weights)
        alg_class.execute(SAVE_PATH, train_loaders, test_loaders, val_loaders)
    elif args.alg == 'feddng':
        alg_class = feddng(args, server_model, clients_models, client_weights)
        alg_class.execute(SAVE_PATH, train_loaders, test_loaders, val_loaders)
    elif args.alg == 'pfedgraph':
        alg_class = pfedgraph(args, server_model, clients_models, client_weights)
        alg_class.execute(SAVE_PATH, train_loaders, test_loaders, val_loaders)
    elif args.alg == 'ditto':
        alg_class = ditto(args, server_model, clients_models, client_weights)
        alg_class.execute(SAVE_PATH, train_loaders, test_loaders, val_loaders)
    elif args.alg == 'brainTorrent':
        alg_class = brainTorrent(args, server_model, clients_models, client_weights)
        alg_class.execute(SAVE_PATH, train_loaders, test_loaders, val_loaders)
    