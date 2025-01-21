from alg.fedavg import fedavg
from alg.utils import train, train_prox


class fedprox(fedavg):
    def __init__(self, args, server_model=None, client_model=None, client_weight=None):
        super(fedprox, self).__init__(args, server_model, client_model, client_weight)

    def client_train(self, c_idx, dataloader, **kwargs):        
        if 'round' in kwargs.keys(): 
            if kwargs['round'] > 0:
                train_loss, train_acc = train_prox(
                    self.args, self.client_model[c_idx], self.server_model, dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
            else:
                train_loss, train_acc = train(
                    self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)        
            return train_loss, train_acc
        else:
            raise ValueError('round is not in kwargs')
    
