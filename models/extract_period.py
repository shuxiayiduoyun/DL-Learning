import numpy as np
import pywt
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import periodogram


def wavelet_decompose(time_series, wavelet="db4", lamb=1600):
    # 对应论文公式 (3): 使用 HP 滤波器将序列分离为周期项 (cycle) 和 趋势项 (trend)
    # lamb=1600 是论文中提到的平滑参数 lambda 的典型值 [cite: 174]
    cycle, trend = hpfilter(time_series, lamb=lamb)
    # 对应论文 Section 4.1.1: 对残差 (cycle) 进行小波分解
    # 将残差分离为不同尺度的分量，这有助于解耦混合的周期影响 [cite: 176]
    coeffs = pywt.wavedec(cycle, wavelet)
    imfs = [trend]
    # 重构小波分量 (Components)，每个分量代表不同尺度的潜在周期模式
    for i in range(len(coeffs)):
        coeffs_temp = [np.zeros_like(c) for c in coeffs]
        coeffs_temp[i] = coeffs[i]
        imf = pywt.waverec(coeffs_temp, wavelet)
        imfs.append(imf[:len(time_series)])  # 确保长度一致
    imfs = np.array(imfs)
    return imfs


def from_periodogram(imfs, k=1, fs=1.0):
    """
    对分解后的每个分量进行快速傅里叶变换（FFT），提取振幅最大的频率 。
    :param imfs:
    :param k:
    :param fs:
    :return:
    """
    all_periods = set()
    length = imfs.shape[1]
    # 对每一个分解后的分量 (imf) 进行谱分析
    for imf in imfs:
        # 计算周期图 (Periodogram) 以获得功率谱密度
        f, Pxx = periodogram(imf, fs)
        # 对应公式 (4): f = argmax(A)
        # 选取功率最大的前 k 个频率索引
        top_k_indices = np.argsort(Pxx)[-k:][::-1]
        top_k_freqs = f[top_k_indices]
        # 将频率转换为周期 p = round(1/f) [cite: 158]
        # 过滤掉过高频或无效的周期
        top_k_periods_imf = [round(1 / freq) for freq in top_k_freqs if freq >= 1/(length - 1)]
        all_periods.update(top_k_periods_imf)
    return list(all_periods)


def extend(original_periods, length, upper=100):
    """
    为了捕捉多尺度的依赖关系，论文提出了将基础频率扩展到其次谐波（sub-harmonics）
    :param original_periods:
    :param length:
    :param upper:
    :return:
    """
    # 对应论文公式 (5): F = U {f_j / k}
    # 这里通过倍增周期 (range(period, ..., period)) 来实现次谐波的逻辑
    # 即：如果基础周期较短，则考虑其倍数周期，以覆盖更长的依赖范围
    extended_periods = set(original_periods)
    for period in original_periods:
        if period <= length/5:  # 只有当周期较短时才扩展，避免超出序列长度
            extended_periods.update(list(range(period, min(length, upper), period)))
        else:
            extended_periods.update(list(range(period, length, period)))
    return list(extended_periods)


def extract_period(sample, k=1, wave="db4", mode="extended-100"):
    """
    sample: ndarray [length, dim]
    results: list
    """
    length, dim = sample.shape
    if mode.split("-")[0] == "all":
        periods = list(range(1, min(int(mode.split("-")[1]) + 1, length)))
        return periods

    single_dim_series = sample[:, 0]
    imfs = wavelet_decompose(single_dim_series, wavelet=wave)
    periods = from_periodogram(imfs, k)

    if mode.split("-")[0] == "extended":
        periods = extend(periods, length, upper=int(mode.split("-")[1]))

    if not periods:
        periods = list(range(1, min(10, length)))
    else:
        periods = sorted(periods)

    return periods







