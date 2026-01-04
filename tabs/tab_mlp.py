#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：tab_mlp.py
@IDE     ：PyCharm 
@Author  ：wei liyu
@Date    ：2025/11/20 20:36 
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


def map_class1_to_class6(class1_value):
    for class6_group, class1_list in classes6.items():
        if class1_value in class1_list:
            return CLASS6[class6_group]
    return -1


def map_class1_to_class4(class1_value):
    for class4_group, class1_list in classes4.items():
        if class1_value in class1_list:
            return CLASS4[class4_group]
    return -1


# ================ 1. 数据预处理函数 ================
def preprocess_data(df, target_column):
    """
    预处理表格数据（数值特征和类别特征）

    参数:
    df (DataFrame): 包含特征和目标列的DataFrame
    target_column (str): 目标列名称

    返回:
    X_train, X_test, y_train, y_test: 处理后的训练/测试数据
    """
    # 分离特征和目标
    X = df.drop(columns=['name', target_column])
    y = df[target_column]

    # 识别数值特征和类别特征
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
        ])

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 应用预处理
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # 如果是分类任务，转换为整数标签
    if y.dtype in [np.int64, np.int32]:
        y_train_tensor = y_train_tensor.long()
        y_test_tensor = y_test_tensor.long()

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, preprocessor


# ================ 2. 简化的MLP模型 ================
class SimpleMLP(nn.Module):
    """
    简化的多层感知机模型（适合新手）

    参数:
    input_dim (int): 输入特征维度
    hidden_dims (list): 隐藏层神经元数量列表
    dropout_rate (float): Dropout率
    output_dim (int): 输出层维度
    """

    def __init__(self, input_dim, hidden_dims, dropout_rate, output_dim):
        super(SimpleMLP, self).__init__()

        # 创建隐藏层
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(in_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ================ 3. 训练和评估函数 ================
def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32, lr=0.001):
    """
    训练MLP模型并返回训练历史

    参数:
    model: MLP模型
    X_train, y_train: 训练数据
    X_test, y_test: 测试数据
    epochs: 训练轮数
    batch_size: 批次大小
    lr: 学习率

    返回:
    train_losses, test_losses: 训练和测试损失历史
    """
    # 选择损失函数（分类/回归自动判断）
    if y_train.dtype == torch.long:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # 训练
        model.train()
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / (len(X_train) // batch_size)

        # 评估
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    return train_losses, test_losses


if __name__ == '__main__':
    file_data = r"E:\temp file\15009099\4FGL-DR4_v34_6classes_GMM_no_coord_features_weighted_prob_cat_v2.csv"
    all_data_df = pd.read_csv(file_data)
    feature_columns = ['sin(GLAT)', 'cos(GLON)', 'sin(GLON)', 'log10(Energy_Flux100)', 'log10(Unc_Energy_Flux100)', 'log10(Signif_Avg)', 'LP_index1000MeV', 'LP_beta', 'LP_SigCurv', 'log10(Variability_Index)']
    label = 'CLASS1'
    temp_df = all_data_df[feature_columns]
    classes6 = {'spp+': ['nov', 'spp'], 'fsrq+': ['fsrq', 'nlsy1'], 'psr+': ['snr', 'hmb', 'pwn', 'psr', 'gc'],
                'msp+': ['msp', 'lmc', 'glc', 'gal', 'sfr', 'bin'], 'bcu+': ['sey', 'bcu', 'sbg', 'agn', 'rdg'],
                'bll+': ['bll', 'ssrq', 'css']}
    classes4 = {'fsrq++': ['fsrq', 'nlsy1', 'css'], 'bll+': ['bll', 'sey', 'sbg', 'agn', 'ssrq', 'rdg'],
                'psr+': ['snr', 'hmb', 'nov', 'pwn', 'psr', 'gc'], 'msp+': ['msp', 'lmb', 'glc', 'gal', 'sfr', 'bin']}
    CLASS6 = {'spp+': 0, 'fsrq+': 1, 'psr+': 2, 'msp+': 3, 'bcu+': 4, 'bll+': 5}
    CLASS4 = {'fsrq++': 0, 'bll+': 1, 'psr+': 2, 'msp+': 3}
    all_data_df['CLASS6'] = all_data_df['CLASS1'].apply(map_class1_to_class6)
    all_data_df['CLASS4'] = all_data_df['CLASS1'].apply(map_class1_to_class4)
    all_data4 = all_data_df[all_data_df['CLASS4'] != -1][['name'] + feature_columns + ['CLASS4']]
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(all_data4, 'CLASS4')
    print(f"输入特征维度: {X_train.shape[1]}")
    print(f"训练样本数量: {X_train.shape[0]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    print("\n创建MLP模型...")
    model = SimpleMLP(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32],
        dropout_rate=0.2,
        output_dim=1 if y_train.dtype == torch.float32 else 4
    )

    model = model.to(device)

    print("\n开始训练模型...")
    train_losses, test_losses = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=50,
        batch_size=16,
        lr=0.0005
    )

    print("\n评估模型...")
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)

        # 如果是分类任务，转换为预测类别
        if y_train.dtype == torch.long:
            _, predicted = torch.max(predictions, 1)
            accuracy = (predicted == y_test).sum().item() / len(y_test)
            print(f"模型准确率: {accuracy:.2%}")
        # else:
        #     # 回归任务
        #     mse = ((predictions - y_test).pow(2)).mean().item()
        #     print(f"均方误差 (MSE): {mse:.4f}")

