#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：mnist_test.py
@IDE     ：PyCharm 
@Author  ：wei liyu
@Date    ：2025/9/28 17:08 
'''
import optuna
import argparse
import torch
import torch.nn as nn
from torch.nn import Sequential
from matplotlib import pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as transforms


# 构建卷积神经网络
class CNN(nn.Module):  # 从父类 nn.Module 继承
    def __init__(self, trial):
        super(CNN, self).__init__()
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5, step=0.1)
        dim_cnn1 = trial.suggest_int("dim_cnn1", 16, 96, step=16)
        dim_cnn2 = trial.suggest_int("dim_cnn2", 32, 128, step=32)
        self.conv1 = Sequential(
            nn.Conv2d(in_channels=1, out_channels=dim_cnn1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_cnn1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = Sequential(
            nn.Conv2d(in_channels=dim_cnn1, out_channels=dim_cnn2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_cnn2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.dense = Sequential(
            nn.Linear(7 * 7 * dim_cnn2, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        # x = x2.view(-1, 7 * 7 * 128)
        x = x2.view(x2.size(0), -1)
        x = self.dense(x)
        return x


def objective(trial):
    model = CNN(trial).to(args.device)
    optimizer_name = trial.suggest_categorical("optimizer", ["RMSprop", "SGD", "Adam", "AdamW"])
    momentum = trial.suggest_float("momentum", 0.0, 0.99)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    if optimizer_name in ["SGD", "RMSprop"]:
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    batch_size = trial.suggest_int("batch_size", 64, 256, step=64)
    criterion = nn.CrossEntropyLoss()
    train_data, test_data = get_data(batch_size)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for images, labels in train_data:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证集评估
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_data:
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = model(images)
                preds = outputs.argmax(dim=1, keepdim=True)
                correct += preds.eq(labels.view_as(preds)).sum().item()

        acc = correct / len(test_data.dataset)
        print(f"[Epoch {epoch + 1}/{args.epochs}] Loss: {total_loss:.4f}, Val Acc: {acc:.4f}")

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")

        # 向 optuna 报告中间结果
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return best_acc


def get_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)])

    data_train = datasets.MNIST(root=args.data_root, transform=transform, train=True, download=True)
    data_test = datasets.MNIST(root=args.data_root, transform=transform, train=False, download=True)
    dataloader_train = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False)
    return dataloader_train, dataloader_test


def main():
    # Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial), n_trials=20)
    print("最佳超参数：", study.best_params)
    print("最佳验证集准确率：", study.best_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mnist test')
    parser.add_argument('--data_root', type=str, default='D:\datasets')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main()
