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
    result_path = './data_db1/db1_epochs_result_1.csv'
    result_df = pd.read_csv(result_path)
    train_accs = result_df['train_acc'].tolist()
    train_losses = result_df['train_loss'].tolist()
    test_accs = result_df['test_acc'].tolist()
    test_losses = result_df['test_loss'].tolist()
    epochs = list(range(1, len(train_accs) + 1))

    fig, ax_acc = plt.subplots(figsize=(8, 5))
    ax_loss = ax_acc.twinx()  # 克隆一个共享 x 轴的右轴

    l1 = ax_acc.plot(epochs, train_accs, 'b-', label='Train Acc')
    l2 = ax_acc.plot(epochs, test_accs, 'b--', label='Test Acc')
    l3 = ax_loss.plot(epochs, train_losses, 'r-', label='Train Loss')
    l4 = ax_loss.plot(epochs, test_losses, 'r--', label='Test Loss')

    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)', color='b')
    ax_loss.set_ylabel('Loss', color='r')
    ax_acc.set_title('Training & Test Accuracy / Loss vs Epoch')

    lines = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in lines]
    ax_acc.legend(lines, labs, loc='center right')

    plt.tight_layout()
    # plt.savefig('acc_loss_single.png', dpi=300)
    plt.show()

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs, train_accs, label='train_acc')
    # plt.plot(epochs, test_accs, label='test_acc')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('acc')
    # plt.title('acc')
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, train_losses, label='train_loss')
    # plt.plot(epochs, test_losses, label='test_loss')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.title('loss')
    # plt.show()
    # plt.savefig('./data_db1/db1_epochs_result_1.png', dpi=300)
