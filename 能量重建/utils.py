#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：utils.py
@IDE     ：PyCharm 
@Author  ：wei liyu
@Date    ：2026/1/7 21:25 
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

def filter_outlines(x):
    """
    :param x: shape [N, 1, w, h]
    :return: 返回滤除的那些样本，以及对应哪个像素值高
    """
    x = x.reshape(x.shape[0], -1)  # [N, 1*w*h]
    idx_max = []
    # softmax归一化 fail ! 造成分母无限大
    for x_i in x:
        """
        # 是否存在负值
        is_negative = np.any(x_i < 0)
        if is_negative:
            print("存在负值")
        """
        # x_i = np.exp(x_i) / np.sum(np.exp(x_i))
        x_i = x_i / np.sum(x_i)
        max_idx = np.argmax(x_i)
        idx_max.append([max_idx, x_i[max_idx]])
    idx_max = np.array(idx_max)
    return idx_max


def plot_probability_density(data, bins=30):
    # 创建图形
    plt.figure(figsize=(10, 6))
    # 使用seaborn绘制概率密度图
    sns.histplot(data, kde=True, stat='density', bins=bins)
    # 设置标题和标签
    plt.title("Probability Density Plot")
    plt.xlabel('Value')
    plt.ylabel('Density')
    # 显示图形
    plt.show()


def load_data(paths=[]):
    X_path = paths[0]
    theta_path = paths[2]
    y_path = paths[1]

    theta = np.load(theta_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)
    X = np.load(X_path).astype(np.float32).reshape(-1, 1, 14, 22)
    return X, y, theta


def calculate_mean_std(dataset):
    """
    计算数据集的均值和标准差
    """
    mean = 0.
    std = 0.
    total_images = dataset.shape[0]

    for images in dataset:
        # images: [channels, height, width]
        images = images.reshape(images.shape[0], -1)
        mean += images.mean(1)  # 按像素维度求均值
        std += images.std(1)  # 按像素维度求标准差

    mean /= total_images
    std /= total_images

    return mean.tolist(), std.tolist()

