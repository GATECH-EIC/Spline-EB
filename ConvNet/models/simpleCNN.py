import torch
import torch.nn as nn
from torch.autograd import Variable
from .data_utils import *

defaultcfg = [16, 16]

class Net(nn.Module):
    def __init__(self, datasetName, cfg=None):
        super(Net, self).__init__()

        self.in_channel, self.in_size, self.num_classes = get_dataset_setting(datasetName)
        self.final_size = self.in_size - 20 - 4
        if cfg is None:
            cfg = defaultcfg

        self.feature = self.make_layers(cfg)
        self.classifier = nn.Linear(cfg[-1] * self.final_size * self.final_size, self.num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def make_layers(self, cfg):
        layers = []
        in_channels = self.in_channel
        for i,v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                if i == 0:
                    conv = nn.Conv2d(in_channels, v, kernel_size=21, padding=0, stride=1, bias=False)
                else:
                    conv = nn.Conv2d(in_channels, v, kernel_size=5, bias=False)
                conv_bat = nn.BatchNorm2d(v)
                layers += [conv, conv_bat, nn.ReLU()]
                in_channels = v

        return nn.Sequential(*layers)

if __name__ == '__main__':
    net = Net("Mnist")
    x = Variable(torch.FloatTensor(16, 1, 28, 28))
    y = net(x)
    print(y.data.shape)