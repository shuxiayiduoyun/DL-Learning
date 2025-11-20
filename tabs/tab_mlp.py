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
    X = df.drop(columns=[target_column])
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
    X_train_tensor = torch.tensor(X_train_processed.toarray(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_processed.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # 如果是分类任务，转换为整数标签
    if y.dtype in [np.int64, np.int32]:
        y_train_tensor = y_train_tensor.long()
        y_test_tensor = y_test_tensor.long()

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, preprocessor


if __name__ == '__main__':
    pass
