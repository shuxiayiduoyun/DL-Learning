#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：EnergyRegModel.py
@IDE     ：PyCharm 
@Author  ：wei liyu
@Date    ：2026/1/8 21:35 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyRegModel(nn.Module):
    def __init__(self):
        super(EnergyRegModel, self).__init__()

        # CNN 分支：处理 1x14x22 的图像
        self.cnn = nn.Sequential(
            # Conv1: 1 -> 16
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 16 x 7 x 11

            # Conv2: 16 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 32 x 3 x 5

            # Conv3: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            # Output: 64 x 3 x 5
        )

        # 计算 CNN 展平后的维度: 64 * 3 * 5 = 960
        self.feature_dim = 64 * 3 * 5

        # 融合后的全连接层 (Feature + Theta)
        # Theta 维度为 1，所以输入维度是 feature_dim + 1
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出能量值
        )

    def forward(self, x, theta):
        # 1. 提取图像特征
        feat = self.cnn(x)
        feat = feat.view(feat.size(0), -1)  # Flatten

        # 2. 特征融合
        # theta 需要调整形状为 [Batch, 1] 以便拼接
        if theta.dim() == 1:
            theta = theta.unsqueeze(1)

        combined = torch.cat((feat, theta), dim=1)

        # 3. 回归预测
        out = self.fc(combined)
        return out.squeeze()  # 移除多余维度，返回 [Batch]