#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：EnergyDataset.py
@IDE     ：PyCharm 
@Author  ：wei liyu
@Date    ：2026/1/8 16:38 
'''
import torch
from torch.utils.data import Dataset
import numpy as np


class EnergyDataset2(Dataset):
    def __init__(self, x, y, theta, transform=None):
        super().__init__()
        # 引用自提供的 EnergyDataset.py
        self.x = x  # 保持 numpy 格式，节省 clone 开销
        self.y = y
        self.theta = theta
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # 1. 图像数据转换
        img = self.x[idx]  # shape (1, 14, 22)
        img_tensor = torch.from_numpy(img).float()

        if self.transform:
            img_tensor = self.transform(img_tensor)

        # 2. 标签和Theta数据转换
        # 确保是 float32，并且如果是标量，保持 tensor 维度一致性
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        theta = torch.tensor(self.theta[idx], dtype=torch.float32)

        return img_tensor, label, theta


class EnergyDataset(Dataset):
    def __init__(self, x, y, theta):
        super().__init__()
        self.x = x  # numpy array
        self.y = y
        self.theta = theta
        # self.phase = phase

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # 1. 获取原始数据
        img_raw = self.x[idx]  # shape: (1, 14, 22)
        label = self.y[idx]  # scalar
        theta = self.theta[idx]  # scalar

        # 2. 关键特征提取：总沉积能量 (Total Deposited Energy)
        # 这是预测初始能量的最重要基准，必须显式告诉模型
        sum_dep = np.sum(img_raw)

        # 3. 图像预处理：对数变换
        # 不使用均值方差归一化！因为像素绝对值代表能量大小，不能丢。
        # 使用 log(x+1) 压缩动态范围，防止大数值导致梯度爆炸
        img_tensor = torch.from_numpy(img_raw).float()
        img_log = torch.log1p(img_tensor)

        # 4. 构建结构化特征向量
        # 将 theta 和 sum_dep 组合，后续直接输入全连接层
        # 对 sum_dep 也做个 log 处理方便网络消化
        dense_feat = torch.tensor([theta, np.log1p(sum_dep)], dtype=torch.float32)

        return img_log, torch.tensor(label, dtype=torch.float32), dense_feat
