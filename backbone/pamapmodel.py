import torch
import torch.nn as nn
from collections import OrderedDict

class PamapModel(nn.Module):
    def __init__(self, n_feature=64, out_dim=10):
        super(PamapModel, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(in_channels=27, out_channels=16, kernel_size=(1, 9))),
                ('bn1', nn.BatchNorm2d(16)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=(1, 2), stride=2)),

                ('conv2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 9))),
                ('bn2', nn.BatchNorm2d(32)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=(1, 2), stride=2)),
            ])
        )

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(in_features=32 * 44, out_features=n_feature)),
                ('fc1_relu', nn.ReLU(inplace=True)),
                ('fc2', nn.Linear(in_features=n_feature, out_features=out_dim)),
            ])
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(-1, 32 * 44)
        x = self.classifier(x)
        return x

    def get_sel_fea(self, x, plan=0):
        if plan == 0:
            x = self.features(x)
            x = x.reshape(-1, 32 * 44)
            y = x
        elif plan == 1:
            x = self.features(x)
            x = x.reshape(-1, 32 * 44)
            x = self.classifier[1](self.classifier[0](x))
            y = x
        else:
            y = []
            x = self.features[3](self.features[2](self.features[1](self.features[0](x))))
            y.append(x.view(x.shape[0], -1))
            x = self.features[7](self.features[6](self.features[5](self.features[4](x))))
            y.append(x.view(x.shape[0], -1))
            x = x.reshape(-1, 32 * 44)
            x = self.classifier[1](self.classifier[0](x))
            y.append(x)
            y = torch.cat(y, dim=1)
        return y
