"""
train_severity_model.py

Training script for SEVERITY LEVEL (ordinal 1–4)
using embedding_text123 only.

Outputs:
- Severity_OrdinalModel.pkl
- severity_metrics.txt
- severity_confusion_matrix.png
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


SCRIPT_DIR = Path(__file__).resolve().parent

# TWO LEVELS UP
DB_PATH = SCRIPT_DIR.parent.parent / "patient_feedback_ml.db"

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

EMBED_COL = "embedding_text123"
TARGET_COL = "severity_level"

MODEL_PATH = SCRIPT_DIR / "Severity_OrdinalModel.pkl"
REPORT_PATH = SCRIPT_DIR / "severity_metrics.txt"
CM_PATH = SCRIPT_DIR / "severity_confusion_matrix.png"


# -------------- Helpers -----------------
def load_table(db_path: Path, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()
    return df


def parse_embedding_series(series: pd.Series) -> np.ndarray:
    out = []
    for i, v in enumerate(series):
        try:
            if isinstance(v, np.ndarray):
                arr = v.astype(float)
            elif isinstance(v, (list, tuple)):
                arr = np.asarray(v, dtype=float)
            elif isinstance(v, (bytes, bytearray)):
                arr = np.frombuffer(v, dtype=np.float32).astype(float)
            elif isinstance(v, str):
                arr = np.asarray(json.loads(v), dtype=float)
            else:
                raise ValueError(f"Unknown format: {type(v)}")
            out.append(arr)
        except Exception as e:
            raise ValueError(f"Error parsing embedding at row {i}: {e}")

    lengths = {len(a) for a in out}
    if len(lengths) != 1:
        raise ValueError(f"Embedding lengths not equal: {lengths}")

    return np.vstack(out)


def save_confusion_matrix(cm, labels, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2 if cm.max() != 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------
# 🚀 FUNCTION VERSION — THIS IS WHAT YOU NEED
# ---------------------------------------------------------
def train_severity_model():
    """
    Trains the severity model and returns:
        model, metrics_dict
    """

    try:
        df_train = load_table(DB_PATH, TRAIN_TABLE)
        df_test = load_table(DB_PATH, TEST_TABLE)

        # Filter 1–4
        df_train = df_train[df_train[TARGET_COL].isin([1, 2, 3, 4])]
        df_test = df_test[df_test[TARGET_COL].isin([1, 2, 3, 4])]

        X_train = parse_embedding_series(df_train[EMBED_COL])
        X_test = parse_embedding_series(df_test[EMBED_COL])

        y_train = df_train[TARGET_COL].astype(int).to_numpy() - 1
        y_test = df_test[TARGET_COL].astype(int).to_numpy() - 1

        # Train
        model = mord.LogisticIT()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        report = classification_report(y_test, y_pred, zero_division=0)

        metrics = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "report": report,
        }

        # Save model
        joblib.dump(model, MODEL_PATH)

        # Save report
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("Severity Level – Ordinal Model Metrics\n\n")
            f.write(f"Accuracy: {acc}\n")
            f.write(f"F1 Macro: {f1_macro}\n\n")
            f.write(report)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        label_list = sorted(list(set(y_train) | set(y_test)))
        save_confusion_matrix(cm, label_list, CM_PATH, "Severity Confusion Matrix")

        return model, metrics

    except Exception:
        traceback.print_exc()
        raise


# ---------------------------------------------------------
# Legacy CLI entry point
# ---------------------------------------------------------
def main():
    model, metrics = train_severity_model()
    print("\nTraining Completed.")
    print(metrics)


if __name__ == "__main__":
    main()
