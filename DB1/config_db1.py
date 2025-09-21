#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/21 14:15
 @Author  : wly
 @File    : config_db1.py
 @Description: 
"""
import torch


class Config:
    def __init__(self):
        self.db1_folder = r'D:\datasets\DB1'
        self.global_db1_fs = 100
        self.rep_total = 10
        self.channel_num = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_reps = [1, 3, 4, 6, 8, 9, 10]
        self.test_reps = [2, 5, 7]
        # self.step_size = 1
        # self.win_size = 30
        self.init_lr = 0.0001
        self.epoch_num = 150
        self.batch_size = 256

config = Config()
