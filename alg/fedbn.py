from alg.fedavg import fedavg

class fedbn(fedavg):
    def __init__(self, args, server_model=None, client_model=None, client_weight=None):
        super(fedbn, self).__init__(args, server_model, client_model, client_weight)