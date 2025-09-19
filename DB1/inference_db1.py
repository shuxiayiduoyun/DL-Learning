#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/22 16:15
 @Author  : wly
 @File    : inference_db1.py
 @Description: 
"""
import os
import numpy as np
import tqdm
import joblib
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

from project_sEMG.Ninapro.DB1.Dataset_DB1 import DB1Dataset
from project_sEMG.Ninapro.DB1.config_db1 import config
from project_sEMG.Ninapro.DB1.utils_db1 import get_db1_data, get_train_scaler, get_sub_wins, get_wins

if __name__ == '__main__':
    model_path = '../save_models_db/best_model.pt'
    model = torch.load(model_path)
    model.to(config.device)
    print(model)

    sub_num = 1
    all_e_data, all_repetitions = [], []
    for e_num in range(1, 4):
        print(f'sub_num: {sub_num}, e_num: {e_num}')
        sub_dict = get_db1_data(sub_num=sub_num, e_num=e_num)
        all_e_data.append(sub_dict)
        repetitions = get_sub_wins(sub_dict=sub_dict, win_size=config.win_size, step_size=config.step_size)
        all_repetitions.append(repetitions)

    if not os.path.exists('./db1_scaler_' + str(sub_num) + '.pkl'):
        print('calculate sub_num standard scaler...')
        scaler = get_train_scaler(all_data=all_e_data, train_reps=config.train_reps)
        joblib.dump(scaler, f'./db1_scaler_{sub_num}.pkl')
    else:
        print('load sub_num standard scaler...')
        scaler = joblib.load('./db1_scaler_' + str(sub_num) + '.pkl')

    test_emg_wins, test_labels = get_wins(all_repetitions=all_repetitions, repetitions_num=config.test_reps)
    print(f'test_emg_wins: {test_emg_wins.shape} test_labels: {test_labels.shape}')
    test_dataset = DB1Dataset(wins_arr=test_emg_wins, labels_arr=test_labels, ss=scaler)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    trues, preds = [], []
    model.eval()
    with torch.no_grad():
        print('start inference...')
        for i, (x, y) in enumerate(tqdm.tqdm(test_loader)):
            x = x.to(config.device)
            y = y.to(config.device)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            trues.extend(y.cpu().numpy())
            preds.extend(pred.cpu().numpy())

    print(f'sub_num: {sub_num}, trues: {len(trues)}, preds: {len(preds)}')
    # 计算评价指标
    accuracy = accuracy_score(trues, preds)
    precision = precision_score(trues, preds, average='weighted')
    recall = recall_score(trues, preds, average='weighted')
    f1 = f1_score(trues, preds, average='weighted')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # 绘制混淆矩阵
    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = np.unique(trues)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
