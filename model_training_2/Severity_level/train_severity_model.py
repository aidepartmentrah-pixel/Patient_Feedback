"""
train_severity_model.py

Training script for SEVERITY LEVEL (ordinal 1–4)
using embedding_text123 only.

Outputs in this folder:
- Severity_OrdinalModel.pkl
- severity_metrics.txt
- severity_confusion_matrix.png

Usage:
    python train_severity_model.py
"""

import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import traceback
import mord
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ----------------
HERE = Path(__file__).resolve().parent
DB_PATH = HERE.parent / "patient_feedback_ml.db"

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

EMBED_COL = "embedding_text123"
TARGET_COL = "severity_level"

MODEL_PATH = HERE / "Severity_OrdinalModel.pkl"
REPORT_PATH = HERE / "severity_metrics.txt"
CM_PATH = HERE / "severity_confusion_matrix.png"

RANDOM_STATE = 42

# -------------- Helpers -----------------
def load_table(db_path: Path, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()
    return df

def parse_embedding_series(series: pd.Series) -> np.ndarray:
    """
    Convert a pandas Series containing embeddings into a 2D numpy array.
    Supported formats:
      - list / tuple
      - JSON string "[0.1, 0.2, ...]"
      - raw float32 BLOB (np.frombuffer)
    """
    out = []
    for i, v in enumerate(series):
        try:
            # CASE 1 — already an array
            if isinstance(v, np.ndarray):
                arr = v.astype(float)

            # CASE 2 — python list
            elif isinstance(v, (list, tuple)):
                arr = np.asarray(v, dtype=float)

            # CASE 3 — raw float32 bytes (BLOB)
            elif isinstance(v, (bytes, bytearray)):
                arr = np.frombuffer(v, dtype=np.float32).astype(float)

            # CASE 4 — JSON string
            elif isinstance(v, str):
                arr = np.asarray(json.loads(v), dtype=float)

            else:
                raise ValueError(f"Unknown embedding format type={type(v)}")

            out.append(arr)

        except Exception as e:
            raise ValueError(f"Unable to parse embedding at row {i}: {e}")

    # Validate equal lengths
    lengths = [len(a) for a in out]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent embedding lengths found: {set(lengths)}")

    return np.vstack(out)

def save_confusion_matrix(cm: np.ndarray, labels, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    thresh = cm.max() / 2.0 if cm.max() != 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# -------------- Main flow -----------------
def main():
    try:
        print(f"\nUsing database: {DB_PATH}")
        print("Loading train/test tables...")

        df_train = load_table(DB_PATH, TRAIN_TABLE)
        df_test = load_table(DB_PATH, TEST_TABLE)

        # Validate columns
        for name, df in (("train", df_train), ("test", df_test)):
            if EMBED_COL not in df.columns:
                raise KeyError(f"Embedding column '{EMBED_COL}' missing in {name} table")
            if TARGET_COL not in df.columns:
                raise KeyError(f"Target column '{TARGET_COL}' missing in {name} table")

        # ---------------- FILTER BEFORE EMBEDDINGS ----------------
        df_train = df_train[df_train[TARGET_COL].isin([1, 2, 3, 4])]
        df_test = df_test[df_test[TARGET_COL].isin([1, 2, 3, 4])]

        if df_train.empty or df_test.empty:
            raise ValueError("After filtering, train or test is empty!")

        # ---------------- Extract now ----------------
        X_train = parse_embedding_series(df_train[EMBED_COL])
        X_test = parse_embedding_series(df_test[EMBED_COL])

        # Now labels match perfectly
        y_train = df_train[TARGET_COL].astype(int).to_numpy() - 1
        y_test = df_test[TARGET_COL].astype(int).to_numpy() - 1

        print(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"Labels -> y_train: {y_train.shape}, y_test: {y_test.shape}")

        print(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"Labels -> y_train: {y_train.shape}, y_test: {y_test.shape}")

        # ----------------- Ordinal Logistic Regression -----------------
        print("Training Ordinal Logistic Regression...")
        model = mord.LogisticIT()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ----------------- Metrics -----------------
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        report_txt = classification_report(y_test, y_pred, zero_division=0)

        print(f"Ordinal LogReg -> acc: {acc:.4f} | f1_macro: {f1_macro:.4f}")

        # ----------------- Save model -----------------
        joblib.dump(model, MODEL_PATH)
        print(f"✅ Saved model: {MODEL_PATH.name}")
        print(f"   Size: {os.path.getsize(MODEL_PATH)} bytes")

        # ----------------- Save metrics -----------------
        label_names = sorted(np.unique(np.concatenate((y_test, y_train))).tolist())
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("Severity Level (ordinal) results\n\n")
            f.write(f"Accuracy: {acc:.6f}\n")
            f.write(f"F1 macro: {f1_macro:.6f}\n")
            f.write("Classification report:\n")
            f.write(report_txt + "\n\n")
        print(f"✅ Saved report: {REPORT_PATH.name}")

        # ----------------- Confusion Matrix -----------------
        cm = confusion_matrix(y_test, y_pred)
        save_confusion_matrix(
            cm,
            labels=label_names,
            out_path=CM_PATH,
            title="Confusion Matrix - Severity Ordinal"
        )
        print(f"✅ Saved confusion matrix: {CM_PATH.name}")

        print("\n🎉 Severity training complete.\n")

    except Exception:
        print("❌ An error occurred during training:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
