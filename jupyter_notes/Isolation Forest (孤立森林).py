#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DL-Learning 
@File    ：Isolation Forest (孤立森林).py
@IDE     ：PyCharm 
@Author  ：wei liyu
@Date    ：2026/1/6 20:52

构建一个数据集，维度为 (N, T)，其中 N 是样本数，T 是每条光变曲线的时间步数（例如 50 个观测点）。
实验设计数据生成：
正常样本：模拟无信号背景，由高斯噪声组成。
异常样本：模拟瞬变事件（如微引力透镜），在噪声基础上叠加一个高斯峰（Gaussian Peak）。
三种模型：Isolation Forest (iForest)
One-Class SVM (OCSVM)
Local Outlier Factor (LOF)
两种检测模式：
离群点检测 (Outlier Detection)：训练集混杂了异常，模型需要“容忍”并找出它们。
奇异值检测 (Novelty Detection)：训练集纯净，模型学习“正常”的边界，检测新来的数据。
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score


# ==========================================
# 1. 数据生成：模拟光变曲线
# ==========================================
def generate_light_curves(n_samples=200, length=50, random_state=42, has_signal=False):
    """
    生成模拟光变曲线数据集。
    - 正常样本: 纯噪声
    - 异常样本: 噪声 + 高斯峰 (模拟瞬变事件)
    """
    rng = np.random.RandomState(random_state)
    X = np.zeros((n_samples, length))

    # 基础噪声 (Normal/Constant)
    for i in range(n_samples):
        # 随机基准亮度 + 高斯白噪声
        baseline = rng.uniform(0, 1)
        noise = rng.normal(0, 0.1, length)
        X[i, :] = baseline + noise

        # 如果是异常样本，叠加一个信号 (Signal)
        if has_signal:
            # 随机峰值位置和宽度
            peak_pos = rng.randint(10, length - 10)
            peak_width = rng.uniform(2, 5)
            peak_amp = rng.uniform(0.5, 1.5)  # 信号强度

            x_axis = np.arange(length)
            signal = peak_amp * np.exp(-0.5 * ((x_axis - peak_pos) / peak_width) ** 2)
            X[i, :] += signal

    return X


# ==========================================
# 2. 辅助绘图函数
# ==========================================
def plot_results(models, X_train, X_test, y_test, mode_name):
    """
    绘制检测结果：展示被判定为最异常的样本和正常的样本
    """
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 4 * len(models)))
    plt.subplots_adjust(hspace=0.4)
    if len(models) == 1: axes = [axes]  # Handle single model case

    for idx, (name, model) in enumerate(models.items()):
        # 获取预测分数 (Score)
        # 注意：sklearn中，score_samples通常越大越正常，越小越异常
        if name == "LOF" and mode_name == "Outlier Detection":
            # LOF 在 outlier detection 模式下没有 score_samples，只有 negative_outlier_factor_
            scores = model.negative_outlier_factor_
            # 对于展示用的测试集，LOF无法直接预测新数据(Outlier模式)，这里仅为演示训练集上的拟合
            # 为了统演示，我们跳过LOF Outlier模式下的Test集评估，直接看它对Train的判断
            test_scores = scores
            X_eval = X_train
            y_eval = np.zeros(len(X_train))  # 伪标签
            eval_title = "Training Set Evaluation (LOF Outlier Mode)"
        else:
            # 正常预测流程
            if mode_name == "Outlier Detection":
                # 在离群点检测中，我们通常看训练数据本身的异常分
                scores = model.decision_function(X_train)
                X_eval = X_train
                # 这里假设混入的后5%是异常
                y_eval = np.hstack([np.ones(int(len(X_train) * 0.95)), -1 * np.ones(int(len(X_train) * 0.05))])
                eval_title = "Training Set (Outlier Detection)"
            else:
                # 奇异值检测：看测试集
                scores = model.score_samples(X_test)
                X_eval = X_test
                y_eval = y_test
                eval_title = "Test Set (Novelty Detection)"

        # 找出得分最低（最异常）和最高（最正常）的样本索引
        top_anomaly_indices = np.argsort(scores)[:3]  # 最异常的3个
        top_normal_indices = np.argsort(scores)[-3:]  # 最正常的3个

        # 绘图 - 左侧：最异常的样本
        ax_row = axes[idx]
        for i in top_anomaly_indices:
            ax_row[0].plot(X_eval[i], alpha=0.8, linewidth=2, label=f'Score: {scores[i]:.2f}')
        ax_row[0].set_title(f"{name} - Detected Anomalies\n(Lowest Scores)", fontsize=10, fontweight='bold',
                            color='darkred')
        ax_row[0].legend(loc='upper right', fontsize=8)
        ax_row[0].grid(True, alpha=0.3)

        # 绘图 - 右侧：最正常的样本
        for i in top_normal_indices:
            ax_row[1].plot(X_eval[i], alpha=0.6, linewidth=1.5, label=f'Score: {scores[i]:.2f}')
        ax_row[1].set_title(f"{name} - Detected Normals\n(Highest Scores)", fontsize=10, fontweight='bold',
                            color='darkgreen')
        ax_row[1].legend(loc='upper right', fontsize=8)
        ax_row[1].grid(True, alpha=0.3)

    plt.suptitle(f"--- {mode_name} Results ---", fontsize=16, y=1.02)
    plt.show()


# ==========================================
# 主程序
# ==========================================

# 全局参数
N_SAMPLES = 300
LEN_SERIES = 50
CONTAMINATION = 0.05  # 异常比例

# ---------------------------------------------------------
# 场景 1: 离群点检测 (Outlier Detection)
# 训练数据: 混杂了异常 (脏数据)
# ---------------------------------------------------------
print("Generating 'Dirty' Data for Outlier Detection...")
# 95% 正常
X_normal = generate_light_curves(n_samples=int(N_SAMPLES * (1 - CONTAMINATION)), length=LEN_SERIES, has_signal=False)
# 5% 异常 (有峰值)
X_outlier = generate_light_curves(n_samples=int(N_SAMPLES * CONTAMINATION), length=LEN_SERIES, has_signal=True)
X_train_dirty = np.vstack([X_normal, X_outlier])

# 打乱数据
rng = np.random.RandomState(42)
rng.shuffle(X_train_dirty)

print(f"Dataset Shape: {X_train_dirty.shape} (Each row is a light curve)")

# 定义模型
models_outlier = {
    "Isolation Forest": IsolationForest(contamination=CONTAMINATION, random_state=42),
    "One-Class SVM": OneClassSVM(nu=CONTAMINATION, kernel="rbf", gamma='scale'),
    "LOF": LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION)
}

# 训练
print("\n--- Training Outlier Detection Models ---")
for name, model in models_outlier.items():
    if name == "LOF":
        # LOF 在 Outlier 模式下直接 fit_predict
        y_pred = model.fit_predict(X_train_dirty)
    else:
        model.fit(X_train_dirty)

# 可视化展示 (展示模型认为最像异常的曲线)
plot_results(models_outlier, X_train_dirty, None, None, "Outlier Detection")

# ---------------------------------------------------------
# 场景 2: 奇异值检测 (Novelty Detection)
# 训练数据: 纯净 (Clean)
# 测试数据: 包含新异常
# ---------------------------------------------------------
print("\nGenerating 'Clean' Data for Novelty Detection...")
# 训练集：全是正常
X_train_clean = generate_light_curves(n_samples=300, length=LEN_SERIES, has_signal=False)

# 测试集：一半正常，一半异常
X_test_norm = generate_light_curves(n_samples=50, length=LEN_SERIES, has_signal=False)
X_test_anom = generate_light_curves(n_samples=50, length=LEN_SERIES, has_signal=True)
X_test = np.vstack([X_test_norm, X_test_anom])
# 标签: 1 为正常, -1 为异常
y_test = np.hstack([np.ones(50), -1 * np.ones(50)])

# 定义模型 (注意 LOF 需要开启 novelty=True)
models_novelty = {
    "Isolation Forest": IsolationForest(contamination='auto', random_state=42),  # 假设不知道异常比例
    "One-Class SVM": OneClassSVM(nu=0.01, kernel="rbf", gamma='scale'),  # nu 设小点，因为训练集很纯
    "LOF (Novelty Mode)": LocalOutlierFactor(n_neighbors=20, novelty=True, contamination='auto')
}

print("\n--- Training Novelty Detection Models ---")
for name, model in models_novelty.items():
    model.fit(X_train_clean)  # 只看纯净数据

    # 评估
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, model.score_samples(X_test))
    print(f"{name} ROC AUC on Test Set: {auc:.4f}")

# 可视化展示 (展示测试集中被判为异常的曲线)
plot_results(models_novelty, X_train_clean, X_test, y_test, "Novelty Detection")
