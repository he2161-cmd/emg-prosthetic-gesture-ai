import numpy as np
import matplotlib.pyplot as plt

# Gesture definitions
GESTURES = {
    0: "Rest",
    1: "Fist",
    2: "Open",
    3: "Pinch",
    4: "Point"
}

# Amplitude per gesture (how strongly each muscle fires)
AMPLITUDES = {
    0: 0.05,   # Rest    — baseline noise only
    1: 1.0,    # Fist    — all flexors, strong
    2: 0.6,    # Open    — extensors, moderate
    3: 0.4,    # Pinch   — targeted, narrow
    4: 0.35    # Point   — specific muscle
}

def generate_emg(gesture_id, duration=1.0, fs=1000):
    """
    Simulate a raw EMG signal for a given gesture.
    
    gesture_id : 0–4
    duration   : signal length in seconds
    fs         : sampling frequency in Hz
    """
    t = np.linspace(0, duration, int(fs * duration))
    A = AMPLITUDES[gesture_id]

    # Burst envelope — smooth on/off contraction shape
    envelope = np.zeros_like(t)
    start = int(0.1 * len(t))
    end   = int(0.9 * len(t))
    envelope[start:end] = np.hanning(end - start)

    # True muscle signal — amplitude modulated Gaussian noise
    emg_raw = A * np.random.randn(len(t)) * envelope

    # Powerline interference — 60 Hz
    noise_powerline = 0.1 * np.sin(2 * np.pi * 60 * t)

    # Motion artifact — low frequency drift + white noise
    noise_motion = 0.05 * np.sin(2 * np.pi * 0.5 * t) + 0.02 * np.random.randn(len(t))

    # Observed signal
    emg_observed = emg_raw + noise_powerline + noise_motion

    return t, emg_observed


def plot_all_gestures():
    """Plot raw EMG signal for all 5 gestures."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Raw EMG Signals per Gesture", fontsize=14)

    for gesture_id, name in GESTURES.items():
        t, signal = generate_emg(gesture_id)
        axes[gesture_id].plot(t, signal, linewidth=0.8)
        axes[gesture_id].set_ylabel(name, fontsize=10)
        axes[gesture_id].set_ylim(-1.5, 1.5)
        axes[gesture_id].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("results/plots/raw_emg_signals.png", dpi=150)
    plt.show()
    print("Plot saved to results/plots/raw_emg_signals.png")


if __name__ == "__main__":
    plot_all_gestures()