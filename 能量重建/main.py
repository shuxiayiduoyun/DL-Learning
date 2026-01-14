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
import time
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
from PhysicsGuidedNet import PhysicsGuidedNet
from mlpnet import MLPRegressor


# --- 定义 MSLE Loss (关注相对误差) ---
class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        # 加上1避免log(0)，保证正值
        return self.mse(torch.log1p(pred), torch.log1p(actual))


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    train_loss = 0
    for img, label, dense in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        img, label, dense = img.to(DEVICE), label.to(DEVICE), dense.to(DEVICE)

        optimizer.zero_grad()
        pred = model(img, dense)

        # 确保预测值为正（能量不能为负），加个 ReLU 或者 abs
        pred = torch.relu(pred)

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(loader)


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    mape_accum = 0  # Mean Absolute Percentage Error
    mae_accum = 0
    with torch.no_grad():
        for img, label, dense in val_loader:
            img, label, dense = img.to(DEVICE), label.to(DEVICE), dense.to(DEVICE)
            pred = torch.relu(model(img, dense))
            loss = criterion(pred, label)
            val_loss += loss.item()

            # 计算直观的物理误差指标 MAPE: |Pred - True| / True
            # 避免分母为0
            mape = torch.abs(pred - label) / (label + 1e-5)
            mape_accum += torch.mean(mape).item()
            mae_accum += torch.mean(torch.abs(pred - label)).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_mape = mape_accum / len(val_loader)
    avg_mae = mae_accum / len(val_loader)
    return avg_val_loss, avg_mape, avg_mae


if __name__ == '__main__':
    # 配置参数
    BATCH_SIZE = 64
    LR = 0.001
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
    filtered_df = max_rate_df.query('max_rate <= 0.2')
    ids, idx_, max_ = filtered_df.index.tolist(), filtered_df['idx'].tolist(), filtered_df['max_rate'].tolist()
    selected_x_data, selected_theta_data, selected_y_data = x_data[ids], theta_data[ids], y_data[ids]

    # X_tr, X_te, s_tr, s_te, y_tr, y_te = train_test_split(selected_x_data, selected_theta_data, selected_y_data, test_size=0.3, random_state=42)
    # print(f'X_tr shape: {X_tr.shape}, X_te shape: {X_te.shape}, s_tr shape: {s_tr.shape}, s_te shape: {s_te.shape}, y_tr shape: {y_tr.shape}, y_te shape: {y_te.shape}')
    # mean, std = calculate_mean_std(X_tr)
    # print(f'mean: {mean}, std: {std}')
    # scaler_y = StandardScaler()
    # scaler_y.fit(y_tr.reshape(-1, 1))
    # y_tr = scaler_y.transform(y_tr.reshape(-1, 1)).reshape(-1)
    # y_te = scaler_y.transform(y_te.reshape(-1, 1)).reshape(-1)
    # train_transform = transforms.Compose([
    #     transforms.Normalize(mean, std)
    # ])
    # train_dataset = EnergyDataset(X_tr, y_tr, s_tr, transform=train_transform)
    # test_dataset = EnergyDataset(X_te, y_te, s_te, transform=train_transform)

    # 3. 划分数据集
    X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
        selected_x_data, selected_y_data, selected_theta_data, test_size=0.3, random_state=42
    )
    # 保存到本地
    np.savez(r"E:\datasets\dampe\selected_data.npz", X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te, t_tr=t_tr, t_te=t_te)
    train_ds = EnergyDataset(X_tr, y_tr, t_tr)
    val_ds = EnergyDataset(X_te, y_te, t_te)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 6. 模型、损失函数与优化器
    # model = EnergyRegModel().to(DEVICE)
    model = PhysicsGuidedNet().to(DEVICE)
    # model = MLPRegressor(h=14, w=22, num_outputs=1, hidden=[512, 256, 128]).to(DEVICE)
    # criterion = nn.MSELoss()  # 回归任务常用均方误差
    # 使用 MSLE 损失函数，或者 L1Loss
    criterion = MSLELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 7. 训练循环
    best_loss = float('inf')

    print(f"Start training on {DEVICE}...")
    for epoch in range(EPOCHS):
        time.sleep(0.5)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, avg_mape, avg_mae = validate(model, test_loader, criterion)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAPE={avg_mape*100:.2f}% | "
              f"Val MAE={avg_mae:.2f}")

        # 保存最优模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"--> Saved best model with Val Loss: {best_loss:.4f}")

    print("Training finished.")
