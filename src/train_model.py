import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ── Absolute paths ────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(BASE_DIR, "models")
DATA_DIR     = os.path.join(BASE_DIR, "data")
RESULTS_DIR  = os.path.join(BASE_DIR, "results", "plots")

GESTURE_NAMES = {0: "Rest", 1: "Fist", 2: "Open", 3: "Pinch", 4: "Point"}


def load_data():
    path = os.path.join(DATA_DIR, "emg_features.csv")
    df   = pd.read_csv(path)
    X    = df.drop(columns=["gesture"]).values
    y    = df["gesture"].values
    return X, y


def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM"          : SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
        "MLP"          : MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
    }
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
    return models


def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                       target_names=list(GESTURE_NAMES.values()))
        results[name] = {"accuracy": acc, "confusion_matrix": cm, "report": report}
        print(f"\n{'='*40}")
        print(f"{name} — Accuracy: {acc*100:.2f}%")
        print(f"{'='*40}")
        print(report)
    return results


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrices", fontsize=14)

    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(res["confusion_matrix"],
                    annot=True, fmt="d", cmap="Blues",
                    xticklabels=GESTURE_NAMES.values(),
                    yticklabels=GESTURE_NAMES.values(),
                    ax=ax)
        ax.set_title(f"{name}\n{res['accuracy']*100:.1f}%", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"), dpi=150)
    plt.show()
    print("Saved confusion_matrices.png")


def plot_accuracy_comparison(results):
    names  = list(results.keys())
    accs   = [res["accuracy"] * 100 for res in results.values()]
    colors = ["#2ecc71", "#3498db", "#e67e22"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accs, color=colors, width=0.5)
    plt.ylim(0, 105)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"{acc:.1f}%", ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison.png"), dpi=150)
    plt.show()
    print("Saved accuracy_comparison.png")


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)),
            importances[indices],
            color="#2ecc71")
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices],
               rotation=45, ha="right")
    plt.title("Random Forest — Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
    plt.show()
    print("Saved feature_importance.png")


def save_models(models, scaler):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(models["Random Forest"], os.path.join(MODELS_DIR, "rf_model.pkl"))
    joblib.dump(models["SVM"],           os.path.join(MODELS_DIR, "svm_model.pkl"))
    joblib.dump(models["MLP"],           os.path.join(MODELS_DIR, "mlp_model.pkl"))
    joblib.dump(scaler,                  os.path.join(MODELS_DIR, "scaler.pkl"))
    print("\nAll models saved to", MODELS_DIR)


if __name__ == "__main__":
    X, y = load_data()
    feature_names = ["rms", "mav", "wl", "zc", "ssc",
                     "var", "mean_freq", "median_freq", "total_power"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    models  = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    plot_confusion_matrices(results)
    plot_accuracy_comparison(results)
    plot_feature_importance(models["Random Forest"], feature_names)

    save_models(models, scaler)