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
