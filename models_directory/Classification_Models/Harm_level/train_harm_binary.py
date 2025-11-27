#!/usr/bin/env python3
"""
Harm_BinaryModel.py
Binary Harm Level Model
Uses embedding_text123.
"""

import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import traceback
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = (SCRIPT_DIR / ".." / ".." / "patient_feedback_ml.db").resolve()
DB_PATH = os.path.abspath(DB_PATH)

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

EMBED_COL = "embedding_text123"
TARGET_COL = "harm_level"

MODEL_PATH = SCRIPT_DIR / "Harm_BinaryModel.pkl"
REPORT_PATH = SCRIPT_DIR / "harm_binary_metrics.txt"
CM_PATH = SCRIPT_DIR / "harm_binary_confusion.png"


# ------------ Helpers ------------
def load_table(db_path: Path, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()


def parse_embedding_series(series: pd.Series) -> np.ndarray:
    out = []
    for v in series:
        if isinstance(v, np.ndarray):
            arr = v.astype(float)
        elif isinstance(v, (list, tuple)):
            arr = np.asarray(v, dtype=float)
        elif isinstance(v, (bytes, bytearray)):
            arr = np.frombuffer(v, dtype=np.float32).astype(float)
        else:
            arr = np.asarray(json.loads(v), dtype=float)
        out.append(arr)
    return np.vstack(out)


def save_confusion_matrix(cm, labels, file):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Binary Harm Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(file)
    plt.close()


# ------------ TRAIN FUNCTION ------------
def train_harm_binary():
    """
    Train binary harm level Logistic Regression model.
    Returns: trained model and metrics dict
    """
    try:
        df_train = load_table(DB_PATH, TRAIN_TABLE)
        df_test = load_table(DB_PATH, TEST_TABLE)

        # Convert 1–6 harm into binary
        df_train["harm_bin"] = df_train[TARGET_COL].apply(lambda x: 1 if x >= 4 else 0)
        df_test["harm_bin"] = df_test[TARGET_COL].apply(lambda x: 1 if x >= 4 else 0)

        X_train = parse_embedding_series(df_train[EMBED_COL])
        X_test = parse_embedding_series(df_test[EMBED_COL])

        y_train = df_train["harm_bin"].to_numpy()
        y_test = df_test["harm_bin"].to_numpy()

        # Train model
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary")
        metrics = {"accuracy": acc, "f1": f1}

        # Save report
        with open(REPORT_PATH, "w") as f:
            f.write(f"Accuracy: {acc}\n")
            f.write(f"F1: {f1}\n")
            f.write(classification_report(y_test, y_pred))

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        save_confusion_matrix(cm, ["low", "high"], CM_PATH)

        # Save model
        joblib.dump(model, MODEL_PATH)
        print("✔ Binary harm model trained.")

        return model, metrics

    except Exception:
        traceback.print_exc()
        return None, None


# ------------ STANDALONE RUN ------------
if __name__ == "__main__":
    model, metrics = train_harm_binary()
    if metrics:
        print("\nMetrics:")
        print(json.dumps(metrics, indent=4))
