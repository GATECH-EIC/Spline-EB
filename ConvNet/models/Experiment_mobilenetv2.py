"""mobilenetv2 in pytorch
[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, cfg=None, t=6, class_num=100):
        super().__init__()

        settings = [in_channels*t, in_channels*t, out_channels]

        if cfg != None:
            settings = cfg
        # print(settings)
        
        conv_list = []
        in_c = in_channels
        for i,out_c in enumerate(settings):
            cv = nn.Conv2d(in_c, out_c, 1)
            if i == 1:
                cv = nn.Conv2d(in_c, out_c,3,stride=stride,padding=1,groups=in_c)
            bn = nn.BatchNorm2d(out_c)
            conv_list.append(cv)
            conv_list.append(bn)
            if i != 2:
                conv_list.append(nn.ReLU6(inplace=True))
            in_c = out_c
        
        self.residual = nn.Sequential(*conv_list)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class MobileNetV2(nn.Module):

    def __init__(self, class_num=100, cfg=None):
        super().__init__()

        # stage_settings = [
        #     # repeat, out_channel, stride, t
        #     [1, 16, 1, 1, None],
        #     [2, 24, 2, 6, None],
        #     [3, 32, 2, 6, None],
        #     [4, 64, 2, 6, None],
        #     [3, 96, 1, 6, None],
        #     [3, 160, 1, 6, None],
        #     [1, 320, 1, 6, None]
        # ]
        # TODO:
        stage_settings = [
            [1, 16, 1, 1, None],
            [2, 64, 2, 2, None],]

        if cfg != None:
            # print(cfg)
            assert len(cfg) == len(stage_settings)
            for i, cf in enumerate(cfg):
                stage_settings[i][-1] = cf

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        stages_list = []
        in_channel = 32
        for i,(repeat, out_channel, stride, t, cfg) in enumerate(stage_settings):
            stages_list.append(self._make_stage(repeat, in_channel, out_channel, stride,t=t, cfg=cfg))
            # in_channel = out_channel 
            if cfg != None:
                in_channel = cfg[i*3+2]  
            else:   
                in_channel = out_channel

        self.stages = nn.Sequential(*stages_list)
 
        if cfg != None:
            self.conv1 = nn.Sequential(
                nn.Conv2d(cfg[-1], 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU6(inplace=True)) 
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(stage_settings[-1][1], 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU6(inplace=True)
            )

        self.conv2 = nn.Conv2d(128, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stages(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t, cfg=None):

        cfg_list = \
        [in_channels * t, in_channels* t, out_channels] + [out_channels * t, out_channels* t, out_channels] * (repeat - 1)

        if cfg != None:
            cfg_list = cfg 

        layers = []
        start_ind = 0
        layers.append(LinearBottleNeck(in_channels=in_channels, out_channels=out_channels, stride=stride, t=t, cfg=cfg_list[start_ind:start_ind+3]))

        while repeat - 1:
            if cfg != None:
                in_channels = cfg_list[start_ind+2]  
            else:   
                in_channels = out_channels
            start_ind += 3
            layers.append(LinearBottleNeck(in_channels=in_channels, out_channels=out_channels, stride=1, t=t,cfg=cfg_list[start_ind:start_ind+3]))
            repeat -= 1
        

        return nn.Sequential(*layers)

def mobilenetv2():
    return MobileNetV2()

# cfg = [29, 29, 15, 87, 87, 58, 346, 346, 58]
# model = MobileNetV2_test(cfg=cfg)
# print(model)