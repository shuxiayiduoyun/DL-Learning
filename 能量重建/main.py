#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：wei liyu
@Date    ：2026/1/7 21:25 
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import filter_outlines, plot_probability_density


if __name__ == '__main__':
    X_path = r"E:\datasets\dampe\bgoEdep.npy"
    theta_path = r"E:\datasets\dampe\\priTheta.npy"
    y_path = r"E:\datasets\dampe\\priE.npy"

    X = np.load(X_path).astype(np.float32).reshape(-1, 1, 14, 22)
    print(f'X shape: {X.shape}')
    # 筛选异常样本
    idx_max = filter_outlines(X)
    print(f'idx_max shape: {idx_max.shape}')
    max_rate = idx_max[:, 1]
    max_rate_df = pd.DataFrame(idx_max, columns=['idx', 'max_rate'])
    plot_probability_density(data=max_rate, bins=50)
    print(max_rate_df.head())
    filtered_df = max_rate_df.query('0.1 <= max_rate <= 0.2')
    print(filtered_df.head())
    ids, idx_, max_ = filtered_df.index.tolist(), filtered_df['idx'].tolist(), filtered_df['max_rate'].tolist()
    print(f'len ids: {len(ids)}')

    random_select_ids = np.random.choice(ids, size=16, replace=False)
    random_x = X[random_select_ids]

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("random X Subplots", fontsize=16)

    for i in range(16):
        row = i // 4
        col = i % 4
        axes[row, col].imshow(random_x[i].squeeze(), cmap='viridis')
        axes[row, col].set_title(f'Image {i + 1}')
        axes[row, col].axis('off')  # 关闭坐标轴

    # 调整子图间距
    plt.tight_layout()
    plt.show()

    X = np.log1p(X)
    y = np.load(y_path).astype(np.float32)
    theta = np.load(theta_path).astype(np.float32)
    theta = np.deg2rad(theta)
