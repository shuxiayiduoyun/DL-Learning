#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：PhysicsGuidedNet.py
@IDE     ：PyCharm 
@Author  ：wei liyu
@Date    ：2026/1/9 11:01 
'''
import torch
import torch.nn as nn


class PhysicsGuidedNet(nn.Module):
    def __init__(self):
        super(PhysicsGuidedNet, self).__init__()

        # --- 分支1：视觉分支 (CNN) ---
        # 目的：提取粒子的入射模式（是直射、斜射还是擦边？）
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 14x22 -> 7x11

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 7x11 -> 3x5

            nn.Flatten()
        )

        cnn_out_dim = 64 * 3 * 5

        # --- 分支2：物理特征融合 ---
        # 输入维度 = CNN特征 + 2个物理特征 (Theta, Log_Sum_Energy)
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, img, dense_feat):
        # img: [batch, 1, 14, 22]
        # dense_feat: [batch, 2] (包含 theta 和 log_sum_energy)

        x_cnn = self.cnn(img)

        # 拼接视觉特征和物理特征
        combined = torch.cat([x_cnn, dense_feat], dim=1)

        # 预测
        out = self.fc(combined)
        return out.squeeze()