import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import welch
from signal_processing import process_emg, segment
from emg_generator import generate_emg, GESTURES


def rms(window):
    """Root Mean Square — signal power / contraction strength."""
    return np.sqrt(np.mean(window ** 2))


def mav(window):
    """Mean Absolute Value — mean activation level."""
    return np.mean(np.abs(window))


def wl(window):
    """Waveform Length — waveform complexity."""
    return np.sum(np.abs(np.diff(window)))


def zc(window, threshold=0.01):
    """Zero Crossings — computed as slope direction changes on rectified signal."""
    diff = np.diff(window)
    count = 0
    for i in range(1, len(diff)):
        if diff[i] * diff[i-1] < 0 and abs(diff[i] - diff[i-1]) > threshold:
            count += 1
    return count


def ssc(window, threshold=0.01):
    """Slope Sign Changes — frequency content indicator."""
    count = 0
    for i in range(1, len(window) - 1):
        diff1 = window[i] - window[i - 1]
        diff2 = window[i + 1] - window[i]
        if diff1 * diff2 < 0 and (abs(diff1) > threshold or abs(diff2) > threshold):
            count += 1
    return count


def var(window):
    """Variance — signal variability."""
    return np.var(window)


def frequency_features(window, fs=1000):
    """Mean frequency, median frequency, total power via Welch method."""
    freqs, psd = welch(window, fs=fs, nperseg=len(window))

    total_power = np.sum(psd)

    if total_power == 0:
        return 0.0, 0.0, 0.0

    mean_freq   = np.sum(freqs * psd) / total_power

    cumulative  = np.cumsum(psd)
    median_freq = freqs[np.searchsorted(cumulative, cumulative[-1] / 2)]

    return mean_freq, median_freq, total_power


def extract_features(smoothed_window, rectified_window, fs=1000):
    """
    Extract all 9 features from a single window.
    ZC and SSC computed on rectified signal (pre-smooth).
    Returns a dict.
    """
    mf, medf, tp = frequency_features(smoothed_window, fs)

    return {
        'rms'         : rms(smoothed_window),
        'mav'         : mav(smoothed_window),
        'wl'          : wl(smoothed_window),
        'zc'          : zc(rectified_window),
        'ssc'         : ssc(rectified_window),
        'var'         : var(smoothed_window),
        'mean_freq'   : mf,
        'median_freq' : medf,
        'total_power' : tp
    }


def plot_feature_distributions():
    """Boxplot of each feature across all gestures."""
    records = []

    for gesture_id, name in GESTURES.items():
        for _ in range(20):
            t, raw                    = generate_emg(gesture_id)
            smoothed, rectified       = process_emg(raw)
            smoothed_windows          = segment(smoothed)
            rectified_windows         = segment(rectified)

            for sw, rw in zip(smoothed_windows, rectified_windows):
                feats = extract_features(sw, rw)
                feats['gesture'] = name
                records.append(feats)

    df = pd.DataFrame(records)
    features = ['rms', 'mav', 'wl', 'zc', 'ssc', 'var', 'mean_freq', 'median_freq', 'total_power']

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle("Feature Distributions per Gesture", fontsize=14)

    for ax, feat in zip(axes.flatten(), features):
        sns.boxplot(data=df, x='gesture', y=feat, ax=ax, palette='Set2')
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    plt.savefig("results/plots/feature_distributions.png", dpi=150)
    plt.show()
    print("Plot saved to results/plots/feature_distributions.png")


if __name__ == "__main__":
    t, raw                = generate_emg(1)
    smoothed, rectified   = process_emg(raw)
    smoothed_windows      = segment(smoothed)
    rectified_windows     = segment(rectified)
    feats                 = extract_features(smoothed_windows[0], rectified_windows[0])

    print("Sample features for Fist gesture:")
    for k, v in feats.items():
        print(f"  {k:15s}: {v:.4f}")

    print()
    plot_feature_distributions()