#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：wei liyu
@Date    ：2026/1/8 15:35 
'''
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch.optim as optim
from EnergyDataset import EnergyDataset
from utils import load_data, filter_outlines, calculate_mean_std
from EnergyRegModel import EnergyRegModel
from mlpnet import MLPRegressor


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, labels, thetas in tqdm(loader, desc="Training"):
        imgs, labels, thetas = imgs.to(device), labels.to(device), thetas.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs.squeeze(-1), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


def validate(model, loader, criterion, device, scaler_y=None):
    model.eval()
    running_loss = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for imgs, labels, thetas in tqdm(loader, desc="Validation"):
            imgs, labels, thetas = imgs.to(device), labels.to(device), thetas.to(device)
            outputs = model(imgs)
            loss = criterion(outputs.squeeze(-1), labels)
            running_loss += loss.item()

            # 如果有缩放器，计算真实的 MAE (Mean Absolute Error) 能量误差
            if scaler_y:
                # 反归一化
                preds_real = scaler_y.inverse_transform(outputs.cpu().numpy().reshape(-1, 1))
                labels_real = scaler_y.inverse_transform(labels.cpu().numpy().reshape(-1, 1))
                total_mae += np.mean(np.abs(preds_real - labels_real)) * imgs.size(0)

    avg_loss = running_loss / len(loader)
    avg_mae = total_mae / len(loader.dataset) if scaler_y else 0
    return avg_loss, avg_mae


if __name__ == '__main__':
    # 配置参数
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 500
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = "./checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
    X_path = r"E:\datasets\dampe\bgoEdep.npy"
    theta_path = r"E:\datasets\dampe\\priTheta.npy"
    y_path = r"E:\datasets\dampe\\priE.npy"

    x_data, y_data, theta_data = load_data(paths=[X_path, y_path, theta_path])
    print(f'x_data shape: {x_data.shape}, y_data shape: {y_data.shape}, theta_data shape: {theta_data.shape}')

    # 筛选异常样本
    idx_max = filter_outlines(x_data)
    max_rate_df = pd.DataFrame(idx_max, columns=['idx', 'max_rate'])
    filtered_df = max_rate_df.query('max_rate >= 0.2')
    ids, idx_, max_ = filtered_df.index.tolist(), filtered_df['idx'].tolist(), filtered_df['max_rate'].tolist()
    selected_x_data, selected_theta_data, selected_y_data = x_data[ids], theta_data[ids], y_data[ids]

    X_tr, X_te, s_tr, s_te, y_tr, y_te = train_test_split(selected_x_data, selected_theta_data, selected_y_data, test_size=0.3, random_state=42)
    print(f'X_tr shape: {X_tr.shape}, X_te shape: {X_te.shape}, s_tr shape: {s_tr.shape}, s_te shape: {s_te.shape}, y_tr shape: {y_tr.shape}, y_te shape: {y_te.shape}')
    mean, std = calculate_mean_std(X_tr)
    print(f'mean: {mean}, std: {std}')
    scaler_y = StandardScaler()
    scaler_y.fit(y_tr.reshape(-1, 1))
    y_tr = scaler_y.transform(y_tr.reshape(-1, 1)).reshape(-1)
    y_te = scaler_y.transform(y_te.reshape(-1, 1)).reshape(-1)
    train_transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    train_dataset = EnergyDataset(X_tr, y_tr, s_tr, transform=train_transform)
    test_dataset = EnergyDataset(X_te, y_te, s_te, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6. 模型、损失函数与优化器
    # model = EnergyRegModel().to(DEVICE)
    model = MLPRegressor(h=14, w=22, num_outputs=1, hidden=[512, 256, 128]).to(DEVICE)
    criterion = nn.MSELoss()  # 回归任务常用均方误差
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)  # 学习率衰减

    # 7. 训练循环
    best_loss = float('inf')

    print(f"Start training on {DEVICE}...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_mae = validate(model, test_loader, criterion, DEVICE, scaler_y)

        scheduler.step()

        print(f"Epoch [{epoch + 1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE (Real Energy): {val_mae:.4f}")

        # 保存最优模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"--> Saved best model with Val Loss: {best_loss:.4f}")

    print("Training finished.")
