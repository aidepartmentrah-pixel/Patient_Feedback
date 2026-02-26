#!/usr/bin/env python3
"""
Harm Binary Classification Model
"""

import json
import os
import sys
from pathlib import Path
import traceback

# Add workspace root to Python path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt

from project_paths import get_db_path
from models_directory.Classification_Models.Hierarchical_Classification_Model.Helper_Functions import (
    load_table,
    parse_embedding_series,
    compute_standardized_metrics,
)

# ============================
# CONSTANTS
# ============================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR
MODEL_PATH = MODEL_DIR / "harm_binary_model.pkl"
REPORT_PATH = MODEL_DIR / "harm_binary_metrics.txt"
CM_PATH = MODEL_DIR / "harm_binary_confusion.png"

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

EMBED_COL = "embedding_text123"
TARGET_COL = "harm_level"


# ============================
# VISUALIZATION
# ============================

def save_confusion_matrix(cm, labels, path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Harm Binary Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ============================
# TRAIN FUNCTION
# ============================

def train_harm_binary(base_path: str | None = None):
    """
    Train binary harm-level classifier.

    Returns:
        model, standardized_metrics
    """

    try:
        db_path = base_path or get_db_path()

        df_train = load_table(db_path, TRAIN_TABLE)
        df_test = load_table(db_path, TEST_TABLE)

        if df_train.empty or df_test.empty:
            raise ValueError("Training or testing dataset is empty")

        # Binary mapping
        df_train["harm_bin"] = df_train[TARGET_COL].apply(lambda x: 1 if x >= 4 else 0)
        df_test["harm_bin"] = df_test[TARGET_COL].apply(lambda x: 1 if x >= 4 else 0)

        X_train = parse_embedding_series(df_train[EMBED_COL])
        X_test = parse_embedding_series(df_test[EMBED_COL])

        y_train = df_train["harm_bin"].values
        y_test = df_test["harm_bin"].values

        # Train model
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Compute standardized metrics
        unique_labels = sorted(np.unique(y_train).tolist())
        standardized_metrics = compute_standardized_metrics(
            model_name="Harm_Binary",
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            label_names=unique_labels,
        )

        # Save report
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("=== Harm Binary Classification ===\n")
            f.write(f"Accuracy: {standardized_metrics['accuracy']:.4f}\n")
            f.write(f"F1 Score: {standardized_metrics['f1']:.4f}\n\n")
            f.write(classification_report(y_test, y_pred))

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        save_confusion_matrix(cm, ["Low", "High"], CM_PATH)

        # Save model
        joblib.dump(model, MODEL_PATH)

        print("[OK] Harm binary model trained successfully")

        return model, standardized_metrics

    except Exception as e:
        print(f"[ERROR] train_harm_binary failed: {str(e)}")
        traceback.print_exc()
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        return None, None


# ============================
# STANDALONE EXECUTION
# ============================

if __name__ == "__main__":
    model, metrics = train_harm_binary()
    if metrics:
        print(json.dumps(metrics, indent=4))
