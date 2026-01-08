#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：data_process.py
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

    theta = np.load(theta_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)
    X = np.load(X_path).astype(np.float32).reshape(-1, 1, 14, 22)
    print(f'X shape: {X.shape}')
    # 筛选异常样本
    idx_max = filter_outlines(X)
    print(f'idx_max shape: {idx_max.shape}')
    max_rate = idx_max[:, 1]
    max_rate_df = pd.DataFrame(idx_max, columns=['idx', 'max_rate'])
    # plot_probability_density(data=max_rate, bins=50)
    # plot_probability_density(data=y, bins=50)
    # plot_probability_density(data=theta, bins=50)
    print(max_rate_df.head())
    filtered_df = max_rate_df.query('max_rate >= 0.6')
    print(filtered_df.head())
    ids, idx_, max_ = filtered_df.index.tolist(), filtered_df['idx'].tolist(), filtered_df['max_rate'].tolist()
    print(f'len ids: {len(ids)}')

    random_select_ids = np.random.choice(ids, size=25, replace=False)
    random_x = X[random_select_ids]
    random_x_theta = theta[random_select_ids]
    random_x_y = y[random_select_ids]

    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    fig.suptitle("random X Subplots", fontsize=16)
    for i in range(25):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(random_x[i].squeeze(), cmap='viridis')
        # 在子图中添加theta, y数值标注
        axes[row, col].text(0.02, 0.98, f'θ={theta[random_select_ids[i]]:.2f}\nE={y[random_select_ids[i]]:.2f}',
                            transform=axes[row, col].transAxes,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[row, col].set_title(f'Image {i + 1}')
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()

    X = np.log1p(X)
    y = np.load(y_path).astype(np.float32)
    theta = np.load(theta_path).astype(np.float32)
    theta = np.deg2rad(theta)
