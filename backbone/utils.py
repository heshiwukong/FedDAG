import copy

from backbone.alexnet import AlexNet
from backbone.pamapmodel import PamapModel
from backbone.lenet import lenet5v
from backbone.cnn import SimpleCNN, SimpleCNNMNIST


BACKBONE = [
    'AlexNet',
    'PamapModel',
    'lenet5v',
    'ResNet',
    'SimpleCNN',
    'SimpleCNNMNIST',
    'EfficientNet',
    'GoogLeNet',
    'MobileNetV2',
    'ShuffleNet'
]

def model_select(args, device):
    '''
    model selection: select the model based on the dataset
    '''
    if args.dataset in ['vlcs', 'pacs', 'off_home', 'off-cal', 'covid19']:
        server_model = AlexNet(num_classes=args.num_classes).to(device)
    elif args.dataset in ['medmnist','medmnistA', 'medmnistC']:
        server_model = lenet5v().to(device)
    elif args.dataset in ['cifar10', 'cifar100']:
        server_model = SimpleCNN().to(device)
    elif args.dataset in ['femnist']:
        server_model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).to(device)
    elif 'pamap' in args.dataset:
        server_model = PamapModel().to(device)    
    else:
        raise NotImplementedError("Backbone not found: {}".format(args.dataset))
        # server_model = get_model(args.model).to(device)       

    client_weights = [1/args.n_parties for _ in range(args.n_parties)]
    models = [copy.deepcopy(server_model).to(device)
              for _ in range(args.n_parties)]
    return server_model, models, client_weights