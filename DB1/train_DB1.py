#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/21 15:15
 @Author  : wly
 @File    : train_DB1.py
 @Description: 
"""
import os
import argparse
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from DB1.Dataset_DB1 import DB1Dataset
from CNNv1 import CNN1D
from models.CNN2D import CNN2D
from utils_db1 import get_db1_data, get_train_scaler, get_sub_wins, get_wins, seed_torch
from config_db1 import config


def evaluate(model, test_loader, loss_fn):
    model.eval()
    test_correct, test_loss = 0., 0.
    size = len(test_loader.dataset)
    batch_total = len(test_loader)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            test_loss += loss_fn(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
    accuracy = 100 * test_correct / size
    loss = test_loss / batch_total
    return accuracy, loss


def train(sub_num):
    all_e_data, all_repetitions = [], []
    for e_num in range(1, 4):
        print(f'sub_num: {sub_num}, e_num: {e_num}')
        sub_dict = get_db1_data(sub_num=sub_num, e_num=e_num)
        all_e_data.append(sub_dict)
        repetitions = get_sub_wins(sub_dict=sub_dict, win_size=args.win_size, step_size=args.step_size)
        all_repetitions.append(repetitions)

    if not os.path.exists('./data_db1/db1_scaler_' + str(sub_num) + '.pkl'):
        print('calculate sub_num standard scaler...')
        scaler = get_train_scaler(all_data=all_e_data, train_reps=config.train_reps)
        joblib.dump(scaler, f'./data_db1/db1_scaler_{sub_num}.pkl')
    else:
        print('load sub_num standard scaler...')
        scaler = joblib.load('./data_db1/db1_scaler_' + str(sub_num) + '.pkl')

    test_emg_wins, test_labels = get_wins(all_repetitions=all_repetitions, repetitions_num=config.test_reps)
    train_emg_wins, train_labels = get_wins(all_repetitions=all_repetitions, repetitions_num=config.train_reps)
    print(f'train_emg_wins: {train_emg_wins.shape} train_labels: {train_labels.shape}, test_emg_wins: {test_emg_wins.shape} test_labels: {test_labels.shape}')
    train_dataset = DB1Dataset(wins_arr=train_emg_wins, labels_arr=train_labels, ss=scaler)
    test_dataset = DB1Dataset(wins_arr=test_emg_wins, labels_arr=test_labels, ss=scaler)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    # model = CNN1D(in_channels=config.channel_num, num_classes=52)
    model = CNN2D(in_channels=10, num_classes=52)
    model.to(config.device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.init_lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=config.init_lr * 0.1, last_epoch=-1)

    # train
    best_test_acc = 0.
    train_losses, test_losses = [], []
    train_acces, test_acces = [], []
    for epoch in range(config.epoch_num):
        running_loss = 0.
        for (step, data) in enumerate(train_loader):
            model.train()
            inputs, labels = data
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 10 == 9:
                test_accuracy, test_loss = evaluate(model=model, test_loader=test_loader, loss_fn=loss_function)
                if test_accuracy > best_test_acc:
                    best_test_acc = test_accuracy
                    torch.save(model, f'./runs/best_model_{args.sub_num}.pt')
                    curr_lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(
                    f'[{epoch + 1}, {step + 1:5d}] train_loss: {running_loss / 10:.3f}, test_accuracy: {test_accuracy:.3f}, '
                    f'test_loss: {test_loss:.3f}, best_test_acc: {best_test_acc:.3f}, lr={curr_lr}.')
                running_loss = 0.
        #  保存每一轮结束后的训练集和测试集的acc和loss
        epoch_train_acc, epoch_train_loss = evaluate(model=model, test_loader=train_loader, loss_fn=loss_function)
        epoch_test_acc, epoch_test_loss = evaluate(model=model, test_loader=test_loader, loss_fn=loss_function)
        train_acces.append(epoch_train_acc)
        train_losses.append(epoch_train_loss)
        test_acces.append(epoch_test_acc)
        test_losses.append(epoch_test_loss)
        scheduler.step()
        print(f'best_test_acc: {best_test_acc:.3f}')

    epochs_result_df = pd.DataFrame({'train_acc': train_acces, 'train_loss': train_losses, 'test_acc': test_acces, 'test_loss': test_losses})
    epochs_result_df.to_csv(f'./data_db1/db1_epochs_result_{sub_num}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--sub_num', type=int, default=1)
    parser.add_argument('--win_size', type=int, default=30)
    parser.add_argument('--step_size', type=int, default=1)
    args = parser.parse_args()

    seed_torch(seed=1029)
    train(args.sub_num)
