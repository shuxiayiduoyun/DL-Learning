#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/21 15:31
 @Author  : wly
 @File    : CNNv1.py
 @Description: 
"""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.fc = nn.Linear(256, num_classes)
        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.avg(x).squeeze(2)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(32, 20, 10)
    model = CNN1D(in_channels=10, num_classes=52)
    outputs = model(x)
    print(outputs.shape)
