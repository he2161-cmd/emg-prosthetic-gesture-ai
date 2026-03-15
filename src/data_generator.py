import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from emg_generator import generate_emg, GESTURES
from signal_processing import process_emg, segment
from feature_extraction import extract_features


def generate_dataset(samples_per_gesture=200, fs=1000):
    """
    Generate a labeled dataset of EMG features.
    samples_per_gesture : number of signal windows per gesture
    Returns a pandas DataFrame.
    """
    records = []

    for gesture_id, name in GESTURES.items():
        print(f"Generating samples for: {name}...")
        count = 0

        while count < samples_per_gesture:
            # Simulate a fresh signal each iteration
            t, raw                = generate_emg(gesture_id, fs=fs)
            smoothed, rectified   = process_emg(raw, fs=fs)
            smoothed_windows      = segment(smoothed, fs=fs)
            rectified_windows     = segment(rectified, fs=fs)

            for sw, rw in zip(smoothed_windows, rectified_windows):
                if count >= samples_per_gesture:
                    break
                feats = extract_features(sw, rw, fs=fs)
                feats['gesture'] = gesture_id
                records.append(feats)
                count += 1

    df = pd.DataFrame(records)
    return df


def save_dataset(df, path="data/emg_features.csv"):
    df.to_csv(path, index=False)
    print(f"\nDataset saved to {path}")
    print(f"Shape      : {df.shape}")
    print(f"Gestures   : {df['gesture'].value_counts().to_dict()}")
    print(f"\nSample rows:")
    print(df.head(10).to_string())


if __name__ == "__main__":
    df = generate_dataset(samples_per_gesture=200)
    save_dataset(df)