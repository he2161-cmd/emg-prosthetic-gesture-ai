import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sys
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR    = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from emg_generator import generate_emg, GESTURES
from signal_processing import process_emg, segment
from feature_extraction import extract_features

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EMG Gesture Classifier",
    page_icon="🦾",
    layout="wide"
)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf     = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    svm    = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
    mlp    = joblib.load(os.path.join(MODELS_DIR, "mlp_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return rf, svm, mlp, scaler

rf, svm, mlp, scaler = load_models()

GESTURE_NAMES = {0: "Rest", 1: "Fist", 2: "Open", 3: "Pinch", 4: "Point"}
GESTURE_EMOJI = {0: "🤚", 1: "✊", 2: "🖐", 3: "🤏", 4: "👆"}
FEATURE_NAMES = ["rms", "mav", "wl", "zc", "ssc", "var",
                 "mean_freq", "median_freq", "total_power"]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🦾 EMG-Based Prosthetic Hand Gesture Classifier")
st.markdown(
    "Simulates a complete EMG signal pipeline — "
    "from raw muscle signal to AI gesture prediction."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")
gesture_id = st.sidebar.selectbox(
    "Select gesture to simulate",
    options=list(GESTURE_NAMES.keys()),
    format_func=lambda x: f"{GESTURE_EMOJI[x]}  {GESTURE_NAMES[x]}"
)
run_button = st.sidebar.button("▶  Run Classification", use_container_width=True)

st.sidebar.divider()
st.sidebar.markdown("**Model accuracies:**")
st.sidebar.markdown("- 🌲 Random Forest: **90.5%**")
st.sidebar.markdown("- 🔵 SVM: **78.0%**")
st.sidebar.markdown("- 🧠 MLP: **84.0%**")

st.sidebar.divider()
st.sidebar.markdown("**Pipeline:**")
st.sidebar.markdown(
    "1. Simulate EMG signal\n"
    "2. Add powerline + motion noise\n"
    "3. Bandpass filter (20–450 Hz)\n"
    "4. Notch filter (60 Hz)\n"
    "5. Rectify + smooth\n"
    "6. Extract 9 features\n"
    "7. Classify with 3 AI models"
)

# ── Main ──────────────────────────────────────────────────────────────────────
if run_button:

    # ── Generate + process signal ─────────────────────────────────────────────
    t, raw              = generate_emg(gesture_id)
    smoothed, rectified = process_emg(raw)
    smoothed_windows    = segment(smoothed)
    rectified_windows   = segment(rectified)

    mid_idx             = len(smoothed_windows) // 2
    sw                  = smoothed_windows[mid_idx]
    rw                  = rectified_windows[mid_idx]
    feats               = extract_features(sw, rw)
    feat_vector         = np.array(list(feats.values())).reshape(1, -1)
    feat_scaled         = scaler.transform(feat_vector)

    # ── Classify ──────────────────────────────────────────────────────────────
    rf_pred   = rf.predict(feat_scaled)[0]
    svm_pred  = svm.predict(feat_scaled)[0]
    mlp_pred  = mlp.predict(feat_scaled)[0]
    rf_proba  = rf.predict_proba(feat_scaled)[0]
    svm_proba = svm.predict_proba(feat_scaled)[0]
    mlp_proba = mlp.predict_proba(feat_scaled)[0]

    # ── Section 1: Signals ────────────────────────────────────────────────────
    st.subheader("① Raw vs Processed Signal")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.plot(t, raw, linewidth=0.8, color='steelblue')
        ax.set_title(f"Raw EMG — {GESTURE_NAMES[gesture_id]}", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.plot(t, smoothed, linewidth=1.2, color='darkorange')
        ax.set_title("Processed signal (envelope)", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ── Section 2: Features ───────────────────────────────────────────────────
    st.subheader("② Extracted Feature Vector")
    feat_vals = list(feats.values())

    fig, ax = plt.subplots(figsize=(10, 3))
    colors  = ['mediumpurple'] * len(FEATURE_NAMES)
    bars    = ax.bar(FEATURE_NAMES, feat_vals, color=colors)
    ax.set_title("9 features extracted from middle window", fontsize=10)
    ax.tick_params(axis='x', labelsize=9)
    for bar, val in zip(bars, feat_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.3f}", ha='center', fontsize=7
        )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Feature table
    feat_df = pd.DataFrame({
        "Feature"    : FEATURE_NAMES,
        "Value"      : [f"{v:.5f}" for v in feat_vals],
        "Description": [
            "Signal power",
            "Mean activation",
            "Waveform complexity",
            "Direction changes",
            "Slope sign changes",
            "Signal variability",
            "Center frequency",
            "Median frequency",
            "Total power"
        ]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Section 3: Predictions ────────────────────────────────────────────────
    st.subheader("③ Model Predictions")
    col1, col2, col3 = st.columns(3)

    def prediction_card(col, model_name, pred, proba, accuracy, color):
        with col:
            correct = pred == gesture_id
            st.markdown(f"**{model_name}** — accuracy: {accuracy}")
            st.metric(
                label="Predicted gesture",
                value=f"{GESTURE_EMOJI[pred]}  {GESTURE_NAMES[pred]}",
                delta="✅ Correct" if correct else "❌ Wrong"
            )
            fig, ax = plt.subplots(figsize=(4, 2.5))
            bar_colors = [color if i == pred else '#dfe6e9' for i in range(5)]
            ax.barh(list(GESTURE_NAMES.values()), proba, color=bar_colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Confidence", fontsize=8)
            ax.tick_params(axis='y', labelsize=9)
            for i, v in enumerate(proba):
                ax.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    prediction_card(col1, "🌲 Random Forest", rf_pred,  rf_proba,  "90.5%", "#2ecc71")
    prediction_card(col2, "🔵 SVM",           svm_pred, svm_proba, "78.0%", "#3498db")
    prediction_card(col3, "🧠 MLP",           mlp_pred, mlp_proba, "84.0%", "#e67e22")

    st.divider()

    # ── Section 4: Majority vote ──────────────────────────────────────────────
    st.subheader("④ Model Agreement")

    preds  = [rf_pred, svm_pred, mlp_pred]
    labels = ["🌲 Random Forest", "🔵 SVM", "🧠 MLP"]

    agree_col, vote_col = st.columns([1, 1])

    with agree_col:
        for label, pred in zip(labels, preds):
            icon = "✅" if pred == gesture_id else "❌"
            st.markdown(
                f"{icon} **{label}** → "
                f"{GESTURE_EMOJI[pred]} {GESTURE_NAMES[pred]}"
            )

    with vote_col:
        votes  = pd.Series(preds).value_counts()
        winner = votes.index[0]
        count  = votes.iloc[0]
        st.metric(
            label="Majority vote",
            value=f"{GESTURE_EMOJI[winner]} {GESTURE_NAMES[winner]}",
            delta=f"{'✅ Correct' if winner == gesture_id else '❌ Wrong'} — {count}/3 models agree"
        )

    st.divider()

    # ── Section 5: Ground truth ───────────────────────────────────────────────
    st.subheader("⑤ Ground Truth")
    st.info(
        f"You simulated: **{GESTURE_EMOJI[gesture_id]} {GESTURE_NAMES[gesture_id]}**  |  "
        f"Majority prediction: **{GESTURE_EMOJI[winner]} {GESTURE_NAMES[winner]}**  |  "
        f"Result: {'✅ Correct' if winner == gesture_id else '❌ Incorrect'}"
    )

else:
    # ── Idle state ────────────────────────────────────────────────────────────
    st.info("👈 Select a gesture from the sidebar and click **Run Classification** to start.")

    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Signal simulation**")
        st.markdown(
            "Each gesture produces a unique EMG amplitude pattern. "
            "Powerline and motion noise are added to simulate real electrode recordings."
        )
    with col2:
        st.markdown("**Signal processing**")
        st.markdown(
            "Bandpass filter keeps 20–450 Hz. "
            "Notch filter removes 60 Hz hum. "
            "Rectification and smoothing extract the muscle activation envelope."
        )
    with col3:
        st.markdown("**AI classification**")
        st.markdown(
            "9 features are extracted per window. "
            "Three models vote on the gesture. "
            "Random Forest leads at 90.5% accuracy."
        )

st.divider()

# ── Upload Section ────────────────────────────────────────────────────────────
st.header("📂 Upload Your Own EMG Signal")
st.markdown(
    "Upload a CSV or TXT file with a **single column** of raw EMG signal values. "
    "The app will run the full pipeline and classify the gesture."
)

uploaded_file = st.file_uploader(
    "Choose a CSV or TXT file",
    type=["csv", "txt"]
)

if uploaded_file is not None:

    # ── Load uploaded signal ──────────────────────────────────────────────────
    try:
        content = uploaded_file.read().decode('latin-1')
        lines   = content.strip().split('\n')

        data = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            try:
                data.append(float(parts[0]))
            except:
                continue

        raw_upload = np.array(data[:5000], dtype=float)

        if len(raw_upload) == 0:
            st.error("No numeric data found in file.")
            st.stop()

    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.success(f"Loaded {len(raw_upload)} samples from `{uploaded_file.name}`")

    # ── Normalize if needed ───────────────────────────────────────────────────
    max_val = np.max(np.abs(raw_upload))
    if max_val > 10:
        raw_upload = raw_upload / max_val
        st.info("Signal normalized to [-1, 1] range.")

    # ── Build time axis ───────────────────────────────────────────────────────
    fs       = 1000
    t_upload = np.linspace(0, len(raw_upload) / fs, len(raw_upload))

    # ── Process uploaded signal ───────────────────────────────────────────────
    smoothed_up, rectified_up = process_emg(raw_upload, fs=fs)
    smoothed_windows_up       = segment(smoothed_up, fs=fs)
    rectified_windows_up      = segment(rectified_up, fs=fs)

    if len(smoothed_windows_up) == 0:
        st.error("Signal too short to segment. Please upload at least 1 second of data (1000 samples at 1000 Hz).")
        st.stop()

    mid_idx_up = len(smoothed_windows_up) // 2
    sw_up      = smoothed_windows_up[mid_idx_up]
    rw_up      = rectified_windows_up[mid_idx_up]
    feats_up   = extract_features(sw_up, rw_up, fs=fs)

    feat_vector_up = np.array(list(feats_up.values())).reshape(1, -1)
    feat_scaled_up = scaler.transform(feat_vector_up)

    # ── Classify ──────────────────────────────────────────────────────────────
    rf_pred_up    = rf.predict(feat_scaled_up)[0]
    svm_pred_up   = svm.predict(feat_scaled_up)[0]
    mlp_pred_up   = mlp.predict(feat_scaled_up)[0]
    rf_proba_up   = rf.predict_proba(feat_scaled_up)[0]
    svm_proba_up  = svm.predict_proba(feat_scaled_up)[0]
    mlp_proba_up  = mlp.predict_proba(feat_scaled_up)[0]

    preds_up      = [rf_pred_up, svm_pred_up, mlp_pred_up]
    votes_up      = pd.Series(preds_up).value_counts()
    winner_up     = votes_up.index[0]

    # ── Section A: Raw signal plot ────────────────────────────────────────────
    st.subheader("① Uploaded Signal")
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t_upload, raw_upload, linewidth=0.8, color='steelblue')
    ax.set_title(f"Uploaded raw EMG signal — {len(raw_upload)} samples", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── Section B: Raw vs Processed ───────────────────────────────────────────
    st.subheader("② Raw vs Processed")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.plot(t_upload, raw_upload, linewidth=0.8, color='steelblue')
        ax.set_title("Raw uploaded signal", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.plot(t_upload, smoothed_up, linewidth=1.2, color='darkorange')
        ax.set_title("Processed envelope", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ── Section C: Extracted features ─────────────────────────────────────────
    st.subheader("③ Extracted Features")
    feat_vals_up = list(feats_up.values())

    fig, ax = plt.subplots(figsize=(10, 3))
    bars = ax.bar(FEATURE_NAMES, feat_vals_up, color='mediumpurple')
    ax.set_title("9 features from uploaded signal", fontsize=10)
    ax.tick_params(axis='x', labelsize=9)
    for bar, val in zip(bars, feat_vals_up):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.3f}", ha='center', fontsize=7
        )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    feat_df_up = pd.DataFrame({
        "Feature"    : FEATURE_NAMES,
        "Value"      : [f"{v:.5f}" for v in feat_vals_up],
        "Description": [
            "Signal power", "Mean activation", "Waveform complexity",
            "Direction changes", "Slope sign changes", "Signal variability",
            "Center frequency", "Median frequency", "Total power"
        ]
    })
    st.dataframe(feat_df_up, use_container_width=True, hide_index=True)

    st.divider()

    # ── Section D: Model predictions ──────────────────────────────────────────
    st.subheader("④ Model Predictions")
    col1, col2, col3 = st.columns(3)

    def upload_prediction_card(col, model_name, pred, proba, accuracy, color):
        with col:
            st.markdown(f"**{model_name}** — {accuracy}")
            st.metric(
                label="Predicted gesture",
                value=f"{GESTURE_EMOJI[pred]}  {GESTURE_NAMES[pred]}"
            )
            fig, ax = plt.subplots(figsize=(4, 2.5))
            bar_colors = [color if i == pred else '#dfe6e9' for i in range(5)]
            ax.barh(list(GESTURE_NAMES.values()), proba, color=bar_colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Confidence", fontsize=8)
            ax.tick_params(axis='y', labelsize=9)
            for i, v in enumerate(proba):
                ax.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    upload_prediction_card(col1, "🌲 Random Forest", rf_pred_up,  rf_proba_up,  "90.5%", "#2ecc71")
    upload_prediction_card(col2, "🔵 SVM",           svm_pred_up, svm_proba_up, "78.0%", "#3498db")
    upload_prediction_card(col3, "🧠 MLP",           mlp_pred_up, mlp_proba_up, "84.0%", "#e67e22")

    st.divider()

    # ── Section E: Compare with simulated ─────────────────────────────────────
    st.subheader("⑤ Comparison with Simulated Signal")
    st.markdown(
        f"Majority vote predicted: "
        f"**{GESTURE_EMOJI[winner_up]} {GESTURE_NAMES[winner_up]}** — "
        f"comparing your upload against a fresh simulation of that gesture."
    )

    t_sim, raw_sim          = generate_emg(winner_up)
    smoothed_sim, _         = process_emg(raw_sim)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.plot(t_upload, smoothed_up, linewidth=1.2, color='steelblue')
        ax.set_title("Your uploaded signal (processed)", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.plot(t_sim, smoothed_sim, linewidth=1.2, color='darkorange')
        ax.set_title(
            f"Simulated {GESTURE_NAMES[winner_up]} signal (processed)", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ── Section F: Majority vote ───────────────────────────────────────────────
    st.subheader("⑥ Final Verdict")
    labels_up = ["🌲 Random Forest", "🔵 SVM", "🧠 MLP"]

    for label, pred in zip(labels_up, preds_up):
        st.markdown(f"→ **{label}** predicted: {GESTURE_EMOJI[pred]} {GESTURE_NAMES[pred]}")

    st.success(
        f"**Majority vote → {GESTURE_EMOJI[winner_up]} {GESTURE_NAMES[winner_up]}** "
        f"— {votes_up.iloc[0]}/3 models agree"
    )