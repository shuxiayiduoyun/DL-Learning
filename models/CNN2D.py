#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：CNN2D.py
@IDE     ：PyCharm 
@Author  ：wei liyu
@Date    ：2025/9/19 20:41 
'''
import torch
import torch.nn as nn


class CNN2D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                                    nn.BatchNorm2d(num_features=16), nn.GELU())
        self.dwconv1 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(num_features=16), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                                    nn.BatchNorm2d(num_features=32), nn.GELU())
        self.dwconv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(num_features=32), nn.GELU())
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(dim=1).permute(0, 1, 3, 2)
        x = self.conv1(x)
        x = self.dwconv1(x)
        x = self.conv2(x)
        x = self.dwconv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    inputs = torch.randn(1, 30, 10)
    model = CNN2D(in_channels=10, num_classes=52)
    outputs = model(inputs)
    print(f"Output shape: {outputs.shape}")
    from thop import profile
    flops, params = profile(model, inputs=(torch.randn(1, 30, 10),))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
