#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：mlpnet.py
@IDE     ：PyCharm 
@Author  ：wei liyu
@Date    ：2026/1/8 21:45 
'''
import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, h=14, w=22, num_outputs=3, hidden=(512, 256, 128), dropout=0.2):
        super().__init__()
        in_dim = 1 * h * w

        layers = [nn.Flatten()]
        last = in_dim
        for hd in hidden:
            layers += [nn.Linear(last, hd), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            last = hd
        layers += [nn.Linear(last, num_outputs)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)