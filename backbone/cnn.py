import torch
import torch.nn as nn
from collections import OrderedDict

class SimpleCNN(nn.Module):
    def __init__(self, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 6, kernel_size=5)),   # (32 - 5 + 1) = 28 -> (6, 28, 28)
                ('relu1', nn.ReLU()),
                ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),  # (6, 14, 14)

                ('conv2', nn.Conv2d(6, 16, kernel_size=5)),  #  (14 - 5 + 1) = 10 -> (16, 10, 10)
                ('relu2', nn.ReLU()),
                ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),  # (16, 5, 5)
            ])
        )

        self.flatten_dim = 16 * 5 * 5  

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(self.flatten_dim, 120)),  
                ('relu3', nn.ReLU()),
                ('fc2', nn.Linear(120, 84)),
                ('relu4', nn.ReLU()),
                ('fc3', nn.Linear(84, output_dim)), 
            ])
        )

    def forward(self, x):
        x = self.features(x)  
        x = x.view(-1, self.flatten_dim)  
        x = self.classifier(x) 
        return x
    
    def get_sel_fea(self, x, plan=0):
        if plan == 0:
            x = self.features(x)
            x = torch.flatten(x, 1)
            y = x
        elif plan == 1:
            x = self.features(x)
            x = torch.flatten(x, 1)
            for i in range(len(self.classifier)):
                if i != 4:
                    x = self.classifier[i](x)
            y = x
        else:
            y = []
            x = self.features(x)
            x = torch.flatten(x, 1)
            y.append(x)
            x = self.classifier[0](x)
            x = self.classifier[1](x)
            y.append(x)
            x = self.classifier[2](x)
            x = self.classifier[3](x)
            y.append(x)
            y = torch.cat(y, dim=1)
        return y

class SimpleCNNMNIST(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(1, 6, 5)),  
                ('relu1', nn.ReLU()),
                ('pool1', nn.MaxPool2d(2, 2)),  

                ('conv2', nn.Conv2d(6, 16, 5)),  
                ('relu2', nn.ReLU()),
                ('pool2', nn.MaxPool2d(2, 2)),  
            ])
        )
        
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(input_dim, hidden_dims[0])), 
                ('relu3', nn.ReLU()),
                ('fc2', nn.Linear(hidden_dims[0], hidden_dims[1])),  
                ('relu4', nn.ReLU()),
                ('fc3', nn.Linear(hidden_dims[1], output_dim)), 
            ])
        )
    
    def forward(self, x):
        x = self.features(x)  
        x = x.view(-1, 16 * 4 * 4)  
        x = self.classifier(x) 
        return x
    
    def get_sel_fea(self, x, plan=0):
        if plan == 0:
            x = self.features(x)
            x = torch.flatten(x, 1)
            y = x
        elif plan == 1:
            x = self.features(x)
            x = torch.flatten(x, 1)
            for i in range(len(self.classifier)):
                if i != 4:
                    x = self.classifier[i](x)
            y = x
        else:
            y = []
            x = self.features(x)
            x = torch.flatten(x, 1)
            y.append(x)
            x = self.classifier[0](x)
            x = self.classifier[1](x)
            y.append(x)
            x = self.classifier[2](x)
            x = self.classifier[3](x)
            y.append(x)
            y = torch.cat(y, dim=1)
        return y        