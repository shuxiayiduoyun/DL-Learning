#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：sarc_test.py
@IDE     ：PyCharm 
@Author  ：wei liyu
@Date    ：2026/1/16 10:30 
'''
import numpy as np
import torch
import sys
import os

# 导入你上传的模块
# 确保 FreqRes.py 和 extract_period.py 在同一目录下
try:
    from extract_period import extract_period
    from FreqRes import FreqBiESN
except ImportError as e:
    print("错误: 无法导入模块。请确保 'FreqRes.py' 和 'extract_period.py' 在当前目录下。")
    print(f"详细错误: {e}")
    sys.exit(1)


def generate_synthetic_data(batch_size=8, length=200, input_dim=1, period=25):
    """
    生成带有明显周期性和趋势的合成数据。

    参数:
        period: 设置一个主周期 (例如 25)，用于测试 extract_period 是否能捕捉到。
    """
    print(f"\n[Step 1] 生成合成数据 (Batch={batch_size}, Length={length}, Dim={input_dim}, Period={period})...")

    # 时间轴
    t = np.arange(length)

    # 1. 周期项 (Sine wave)
    cycle = np.sin(2 * np.pi * t / period)

    # 2. 趋势项 (Linear trend) - 测试 HP 滤波器的去趋势能力 [cite: 141]
    trend = 0.05 * t

    # 3. 噪声
    noise = np.random.normal(0, 0.1, size=(batch_size, length, input_dim))

    # 组合数据 (广播机制)
    # 基础信号 shape: (length,) -> (1, length, 1)
    base_signal = (cycle + trend).reshape(1, length, 1)

    # 生成 Batch 数据
    data = base_signal + noise

    # 转换为 float32 (PyTorch 默认格式)
    return data.astype(np.float32)


def test_sarc_pipeline():
    # --- 配置参数 ---
    BATCH_SIZE = 4
    LENGTH = 150
    INPUT_DIM = 1
    TRUE_PERIOD = 20  # 我们手动注入的周期

    # 1. 获取数据
    data_np = generate_synthetic_data(BATCH_SIZE, LENGTH, INPUT_DIM, TRUE_PERIOD)

    # 2. 测试周期提取 (Spectral Analysis)
    print("\n[Step 2] 测试周期提取模块 (extract_period)...")

    # extract_period 需要单个样本，形状为 [length, dim]
    sample_series = data_np[0]

    # 调用 extract_period
    # 对应论文中：去趋势 -> 小波分解 -> FFT 提取显著频率 [cite: 132, 140]
    detected_periods = extract_period(sample_series, k=1, mode="extended-100")

    print(f" -> 注入的真实周期: {TRUE_PERIOD}")
    print(f" -> 算法检测到的周期集合: {detected_periods}")

    if not detected_periods:
        print("警告: 未检测到周期，使用默认列表 [1...10]")
        detected_periods = list(range(1, 10))

    # 3. 测试模型构建与前向传播 (FreqRes)
    print("\n[Step 3] 测试 FreqRes 模型 (FreqBiESN)...")

    # 初始化模型
    # hidden_dim=10 是论文实验中的默认设置 [cite: 343]
    model = FreqBiESN(
        input_dim=INPUT_DIM,
        periods=detected_periods,
        hidden_dim=10,
        spectral_radius=(0.8, 0.8),
        regular=1.0
    )

    # 打印模型结构简述
    print(f" -> 模型初始化成功。包含 {len(detected_periods)} 个 FreqRes 模块。")
    print(f" -> 使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # 准备输入 Tensor
    input_tensor = torch.from_numpy(data_np)
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    # 前向传播
    # 对应论文公式 (6) 状态更新 和 (8) 岭回归特征提取 [cite: 209, 220]
    try:
        features = model(input_tensor)
        print(f" -> 前向传播成功。")
        print(f" -> 输入形状: {input_tensor.shape} (Batch, Length, Dim)")
        print(f" -> 输出特征形状: {features.shape} (Batch, Feature_Dim)")
        print("\n[测试通过] SARC 流程跑通！")

    except Exception as e:
        print(f"\n[测试失败] 模型前向传播出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_sarc_pipeline()
