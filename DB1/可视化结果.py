#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/21 16:22
 @Author  : wly
 @File    : 可视化结果.py
 @Description: 
"""
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    result_path = './db1_epochs_result_1.csv'
    result_df = pd.read_csv(result_path)
    train_accs = result_df['train_acc'].tolist()
    train_losses = result_df['train_loss'].tolist()
    test_accs = result_df['test_acc'].tolist()
    test_losses = result_df['test_loss'].tolist()
    epochs = list(range(1, len(train_accs) + 1))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, label='train_acc')
    plt.plot(epochs, test_accs, label='test_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('acc')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label='train_loss')
    plt.plot(epochs, test_losses, label='test_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss')
    plt.show()
    plt.savefig('./db1_epochs_result_1.png', dpi=300)
