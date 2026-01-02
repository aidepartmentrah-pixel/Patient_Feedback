"""
train_improvement_model.py

Ordinal model for Improvement Level:
1 = Ordinary
2 = Red Flag
3 = Never Event
"""

import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import traceback
import mord
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import sys

# --------------------------------------------------
# Project imports
# --------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from models_directory.Classification_Models.Hierarchical_Classification_Model.Helper_Functions import (
    compute_standardized_metrics,
)

# --------------------------------------------------
# Paths & Config
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR.parent.parent / "patient_feedback_ml.db"

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

EMBED_COL = "embedding_text123"
TARGET_COL = "improvement_opportunity_type"

MODEL_PATH = SCRIPT_DIR / "Improvement_OrdinalModel.pkl"
REPORT_PATH = SCRIPT_DIR / "improvement_metrics.txt"
CM_PATH = SCRIPT_DIR / "improvement_confusion_matrix.png"


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_table(db_path: Path, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()


def parse_embedding_series(series: pd.Series) -> np.ndarray:
    vectors = []
    for i, v in enumerate(series):
        try:
            if isinstance(v, np.ndarray):
                arr = v.astype(float)
            elif isinstance(v, (list, tuple)):
                arr = np.asarray(v, dtype=float)
            elif isinstance(v, (bytes, bytearray)):
                arr = np.frombuffer(v, dtype=np.float32)
            elif isinstance(v, str):
                arr = np.asarray(json.loads(v), dtype=float)
            else:
                raise ValueError(f"Unsupported embedding type: {type(v)}")
            vectors.append(arr)
        except Exception as e:
            raise ValueError(f"Embedding error at row {i}: {e}")

    if len({len(v) for v in vectors}) != 1:
        raise ValueError("Inconsistent embedding dimensions")

    return np.vstack(vectors)


# --------------------------------------------------
# Oversampling (safe, in-memory)
# --------------------------------------------------
def oversample_class(df, target_col, target_value, desired_ratio=0.35, random_state=42):
    """
    Oversample minority class without touching DB.
    """
    majority = df[df[target_col] != target_value]
    minority = df[df[target_col] == target_value]

    if len(minority) == 0:
        print(f"⚠️ No samples for class {target_value}. Skipping oversampling.")
        return df

    target_size = int(len(majority) * desired_ratio)

    if len(minority) >= target_size:
        return df

    minority_upsampled = minority.sample(
        n=target_size,
        replace=True,
        random_state=random_state
    )

    return pd.concat([majority, minority_upsampled]).sample(
        frac=1, random_state=random_state
    )


def save_confusion_matrix(cm, labels, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------
# TRAINING FUNCTION
# --------------------------------------------------
def train_improvement_model():
    """
    Trains ordinal improvement model using standardized metrics.
    """

    try:
        df_train = load_table(DB_PATH, TRAIN_TABLE)
        df_test = load_table(DB_PATH, TEST_TABLE)

        # Valid labels only
        df_train = df_train[df_train[TARGET_COL].isin([1, 2, 3])]
        df_test = df_test[df_test[TARGET_COL].isin([1, 2, 3])]

        # 🔥 Oversample Red Flag (class 2)
        df_train = oversample_class(
            df_train,
            target_col=TARGET_COL,
            target_value=2,
            desired_ratio=0.35
        )

        # Embeddings
        X_train = parse_embedding_series(df_train[EMBED_COL])
        X_test = parse_embedding_series(df_test[EMBED_COL])

        # Ordinal encoding
        y_train = df_train[TARGET_COL].astype(int).to_numpy() - 1
        y_test = df_test[TARGET_COL].astype(int).to_numpy() - 1

        # Train ordinal model
        model = mord.LogisticIT()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Restore original labels
        y_test_orig = y_test + 1
        y_pred_orig = y_pred + 1

        # Standardized metrics (same style as Severity)
        standardized_metrics = compute_standardized_metrics(
            model_name="Improvement_Ordinal_Model",
            y_train=df_train[TARGET_COL].to_numpy(),
            y_test=y_test_orig,
            y_pred=y_pred_orig,
            label_names=[1, 2, 3],
        )

        # Save model
        joblib.dump(model, MODEL_PATH)

        # Save report
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("Improvement Ordinal Model Metrics\n\n")
            f.write(f"Accuracy: {standardized_metrics['accuracy']}\n")
            f.write(f"F1: {standardized_metrics['f1']}\n\n")
            f.write(classification_report(y_test, y_pred, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        save_confusion_matrix(
            cm,
            labels=["Ordinary", "Red Flag", "Never Event"],
            out_path=CM_PATH,
            title="Improvement Ordinal Confusion Matrix"
        )

        return model, standardized_metrics

    except Exception:
        traceback.print_exc()
        raise


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    model, metrics = train_improvement_model()
    print("\nTraining completed.")
    print(metrics)
