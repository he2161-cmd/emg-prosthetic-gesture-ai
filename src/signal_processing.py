import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from emg_generator import generate_emg, GESTURES


def bandpass_filter(signal, lowcut=20, highcut=450, fs=1000, order=4):
    """Keep only frequencies where real EMG lives: 20–450 Hz."""
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def notch_filter(signal, freq=60, fs=1000, Q=30):
    """Remove powerline interference at exactly 60 Hz."""
    b, a = iirnotch(freq / (fs / 2), Q)
    return filtfilt(b, a, signal)


def rectify(signal):
    """Full-wave rectification — take absolute value."""
    return np.abs(signal)


def smooth(signal, cutoff=5, fs=1000):
    """Low-pass filter to get the muscle activation envelope."""
    b, a = butter(2, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, signal)


def process_emg(signal, fs=1000):
    """Run the full processing pipeline on a raw EMG signal.
    Returns both the smoothed envelope and the rectified (pre-smooth) signal."""
    signal = bandpass_filter(signal, fs=fs)
    signal = notch_filter(signal, fs=fs)
    rectified = rectify(signal)
    smoothed = smooth(rectified, fs=fs)
    return smoothed, rectified


def segment(signal, fs=1000, window_ms=200, overlap_ms=100):
    """
    Divide signal into overlapping windows.
    Returns list of windows (numpy arrays).
    """
    window_size  = int(window_ms  * fs / 1000)
    overlap_size = int(overlap_ms * fs / 1000)
    step = window_size - overlap_size

    windows = []
    start = 0
    while start + window_size <= len(signal):
        windows.append(signal[start:start + window_size])
        start += step

    return windows


def plot_raw_vs_processed():
    """Plot raw vs processed signal side by side for all gestures."""
    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
    fig.suptitle("Raw vs Processed EMG Signal", fontsize=14)

    for gesture_id, name in GESTURES.items():
        t, raw = generate_emg(gesture_id)
        processed = process_emg(raw)

        axes[gesture_id][0].plot(t, raw, linewidth=0.8, color='steelblue')
        axes[gesture_id][0].set_ylabel(name, fontsize=9)
        axes[gesture_id][0].set_ylim(-1.5, 1.5)
        axes[gesture_id][0].grid(True, alpha=0.3)

        axes[gesture_id][1].plot(t, processed, linewidth=0.8, color='darkorange')
        axes[gesture_id][1].set_ylim(0, 1.0)
        axes[gesture_id][1].grid(True, alpha=0.3)

    axes[0][0].set_title("Raw", fontsize=10)
    axes[0][1].set_title("Processed", fontsize=10)
    axes[-1][0].set_xlabel("Time (s)")
    axes[-1][1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("results/plots/raw_vs_processed.png", dpi=150)
    plt.show()
    print("Plot saved to results/plots/raw_vs_processed.png")


if __name__ == "__main__":
    plot_raw_vs_processed()