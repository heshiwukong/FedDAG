import torch
import torch.nn as nn
from collections import OrderedDict

class lenet5v(nn.Module):
    def __init__(self):
        super(lenet5v, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(1, 6, kernel_size=5)),
                ('bn1', nn.BatchNorm2d(6)),
                ('relu1', nn.ReLU()),
                ('maxpool1', nn.MaxPool2d(kernel_size=2)),

                ('conv2', nn.Conv2d(6, 16, kernel_size=5)),
                ('bn2', nn.BatchNorm2d(16)),
                ('relu2', nn.ReLU()),
                ('maxpool2', nn.MaxPool2d(kernel_size=2)),
            ])
        )
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256, 120)),  # Assuming input images are 32x32 
                ('relu3', nn.ReLU()),
                ('fc2', nn.Linear(120, 84)),
                ('relu4', nn.ReLU()),
                ('fc3', nn.Linear(84, 11)),
            ])
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def getallfea(self, x):
        fealist = []
        for i in range(len(self.features)):
            if i in [1, 5]:  # Indices of layers after which we want to record features
                fealist.append(x.clone().detach())
            x = self.features[i](x)
        return fealist

    def getfinalfea(self, x):
        for i in range(len(self.features)):
            y = self.features[i](x)
        y = torch.flatten(y, 1)
        return y

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
