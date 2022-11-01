import torch.nn as nn
from torch.autograd import Variable
import torch
from .data_utils import *
import math
# code modified from https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class AlexNet(nn.Module):

    def __init__(self, datasetName, cfg=None ):
        super(AlexNet, self).__init__()

        self.in_channel, self.in_size, self.num_classes = get_dataset_setting(datasetName)
        self.final_size = math.ceil((math.ceil((self.in_size - 10) / 4)-2)/2)

        if cfg is None:
            cfg = []

        self.feature = nn.Sequential(
            nn.Conv2d(self.in_channel, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            # LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # LRN(local_size=5, alpha=0.0001, beta=0.75),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # TODO: change in_features of Linear
        # currently  256 * 6 * 6 because input pic size=227*227*3(Imagenet)
        self.classifier = nn.Sequential(
            nn.Linear(256 * self.final_size * self.final_size, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 256 * self.final_size * self.final_size)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    net = AlexNet("Cifar10")
    x = Variable(torch.FloatTensor(16, 3, 32, 32))

    hooks = []
    acti_conv = []
    for m in net.feature.children():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(lambda model, input, output : acti_conv.append(output.detach())))
    with torch.no_grad():
        outputs = net(x)

    for i in acti_conv:
        print(i.shape)