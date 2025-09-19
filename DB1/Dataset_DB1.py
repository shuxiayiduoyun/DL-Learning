#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/21 15:16
 @Author  : wly
 @File    : Dataset_DB1.py
 @Description: 
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class DB1Dataset(Dataset):
    def __init__(self, wins_arr, labels_arr, ss):
        self.wins_data = wins_arr
        self.labels = labels_arr
        self.ss = ss

    def __len__(self):
        return self.wins_data.shape[0]

    def __getitem__(self, idx):
        win_arr = self.wins_data[idx]
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.int64)
        win_arr = self.ss.transform(win_arr)
        win_arr = torch.tensor(win_arr, dtype=torch.float32)
        return win_arr, label
