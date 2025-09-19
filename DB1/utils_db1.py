#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/21 14:31
 @Author  : wly
 @File    : utils_db1.py
 @Description: 
"""
import scipy
import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, iirnotch
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from config_db1 import config


# db1 tools
def get_db1_data(sub_num, e_num):
    sub_root = os.path.join(config.db1_folder, f's{sub_num}')
    file_path = os.path.join(sub_root, f'S{sub_num}_A1_E{e_num}.mat')
    db1_data = scipy.io.loadmat(file_name=file_path)
    emg_data = db1_data['emg'].astype(float)
    emg_data = np.abs(emg_data)  # 全波整流
    repetition_data = db1_data['rerepetition'].astype(int)
    stimulus_data = db1_data['restimulus'].astype(int)
    if e_num == 1:
        stimulus_data = stimulus_data - 1
    elif e_num == 2:
        stimulus_data[stimulus_data == 0] -= 1
        stimulus_data[stimulus_data != -1] += 11
    elif e_num == 3:
        stimulus_data[stimulus_data == 0] -= 1
        stimulus_data[stimulus_data != -1] += 28
    sub_e_dict = {'sub_num': sub_num,
                'emg_data': emg_data,
                'repetition_data': repetition_data,
                'stimulus_data': stimulus_data}
    return sub_e_dict


def get_sub_wins(sub_dict, win_size, step_size):
    emg_data = sub_dict['emg_data']
    repetition_data = sub_dict['repetition_data'].squeeze()
    stimulus_data = sub_dict['stimulus_data'].squeeze()
    # 初始化一个字典
    repetitions_dict = {i: [] for i in range(1, config.rep_total+1)}
    # 滑动窗口循环
    for start in range(0, emg_data.shape[0] - win_size + 1, step_size):
        win_stimulus = stimulus_data[start:start + win_size]
        win_repetition = repetition_data[start:start + win_size]
        count_stimulus = len(np.unique(win_stimulus))
        if count_stimulus > 1:
            continue
        label_num = win_stimulus[win_size // 2]
        if label_num == -1:
            # 不包括rest手势
            continue
        repetition_num = win_repetition[win_size // 2]
        if repetition_num == 0:
            continue
        win_arr = emg_data[start:start + win_size, :]
        win_dict = {'label_num': label_num,
                    'repetition_num': repetition_num,
                    'win_arr': win_arr}
        repetitions_dict[repetition_num].append(win_dict)
    return repetitions_dict


def get_wins(all_repetitions, repetitions_num):
    wins_emg_list, wins_label_list = [], []
    for repetitions_dict in all_repetitions:
        for repetition_num in repetitions_num:
            wins_list = repetitions_dict[repetition_num]
            for win_dict in wins_list:
                wins_label_list.append(int(win_dict['label_num']))
                wins_emg_list.append(win_dict['win_arr'])
    return np.array(wins_emg_list), np.array(wins_label_list)


def get_train_scaler(all_data, train_reps=[1, 3, 4, 6]):
    all_train_data = []
    for e_data in all_data:
        emg_data = e_data['emg_data']
        rep_data = e_data['repetition_data']
        for rep_num in train_reps:
            train_idx = np.where(rep_data == rep_num)[0]
            train_data = emg_data[train_idx, :]
            all_train_data.append(train_data)
    all_train_data = np.concatenate(all_train_data, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_train_data)
    return scaler


if __name__ == '__main__':
    e_dict = get_db1_data(sub_num=1, e_num=1)
    emg_data = e_dict['emg_data']
    repetition_data = e_dict['repetition_data']
    stimulus_data = e_dict['stimulus_data']
    if 1 == 0:
        fig, axs = plt.subplots(3, 1, figsize=(20, 10))
        for i in range(emg_data.shape[1]):
            axs[0].plot(emg_data[:, i], label=f'Channel {i + 1}')
        axs[0].set_title('10 Channel EMG Data')
        axs[0].set_xlabel('Sample Number')
        axs[0].set_ylabel('Amplitude')
        axs[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))

        # 绘制 repetition_data
        axs[1].plot(repetition_data, label='Repetition Data')
        axs[1].set_title('Repetition Data')
        axs[1].set_xlabel('Sample Number')
        axs[1].set_ylabel('Repetition')
        axs[1].legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))

        # 绘制 stimulus_data
        axs[2].plot(stimulus_data, label='Stimulus Data')
        axs[2].set_title('Stimulus Data')
        axs[2].set_xlabel('Sample Number')
        axs[2].set_ylabel('Stimulus')
        axs[2].legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))

        # 调整子图之间的间距
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle('Ninapro Dataset Visualization')
        plt.savefig('DB1_visualization.png', dpi=300)
        plt.show()

    one_e_dict = get_sub_wins(sub_dict=e_dict, win_size=20, step_size=5)
    for key, value in one_e_dict.items():
        print(key, len(value))
