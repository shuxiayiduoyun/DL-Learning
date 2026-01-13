import numpy as np
import pywt
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import periodogram


def wavelet_decompose(time_series, wavelet="db4", lamb=1600):
    cycle, trend = hpfilter(time_series, lamb=lamb)
    coeffs = pywt.wavedec(cycle, wavelet)
    imfs = [trend]
    for i in range(len(coeffs)):
        coeffs_temp = [np.zeros_like(c) for c in coeffs]
        coeffs_temp[i] = coeffs[i]
        imf = pywt.waverec(coeffs_temp, wavelet)
        imfs.append(imf[:len(time_series)])
    imfs = np.array(imfs)
    return imfs


def from_periodogram(imfs, k=1, fs=1.0):
    all_periods = set()
    length = imfs.shape[1]
    for imf in imfs:
        f, Pxx = periodogram(imf, fs)
        top_k_indices = np.argsort(Pxx)[-k:][::-1]
        top_k_freqs = f[top_k_indices]
        top_k_periods_imf = [round(1 / freq) for freq in top_k_freqs if freq >= 1/(length - 1)]
        all_periods.update(top_k_periods_imf)
    return list(all_periods)


def extend(original_periods, length, upper=100):
    extended_periods = set(original_periods)
    for period in original_periods:
        if period <= length/5:
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







