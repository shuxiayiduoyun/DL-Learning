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


class EnergyDataset(Dataset):
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
