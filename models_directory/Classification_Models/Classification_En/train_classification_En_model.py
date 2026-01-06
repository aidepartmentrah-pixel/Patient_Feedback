import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import traceback
from sklearn.linear_model import LogisticRegression
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
# CONFIG
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR.parent.parent / "patient_feedback_ml.db"

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

EMBED_COL = "embedding_text123"
TARGET_COL = "classification_en"

MODEL_PATH = SCRIPT_DIR / "ClassificationEN_Model.pkl"
REPORT_PATH = SCRIPT_DIR / "classification_en_metrics.txt"
CM_PATH = SCRIPT_DIR / "classification_en_confusion_matrix.png"

# 🔴 IMPORTANT: BASE ID
CLASS_BASE = 78   # 78 → 0, 79 → 1, ...

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
                arr = v
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


def save_confusion_matrix(cm, labels, out_path, title):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# --------------------------------------------------
# TRAINING
# --------------------------------------------------
def train_classification_en_model():
    try:
        df_train = load_table(DB_PATH, TRAIN_TABLE)
        df_test = load_table(DB_PATH, TEST_TABLE)

        # Drop NaNs
        df_train = df_train[~df_train[TARGET_COL].isna()]
        df_test = df_test[~df_test[TARGET_COL].isna()]

        df_train[TARGET_COL] = df_train[TARGET_COL].astype(int)
        df_test[TARGET_COL] = df_test[TARGET_COL].astype(int)

        X_train = parse_embedding_series(df_train[EMBED_COL])
        X_test = parse_embedding_series(df_test[EMBED_COL])

        y_train_real = df_train[TARGET_COL].to_numpy()
        y_test_real = df_test[TARGET_COL].to_numpy()

        # ------------------------------
        # Convert REAL → LOCAL
        # ------------------------------
        y_train = y_train_real - CLASS_BASE
        y_test = y_test_real - CLASS_BASE

        # Safety check
        if np.min(y_train) < 0:
            raise RuntimeError("Found classification_en < CLASS_BASE (78). Fix DB or BASE.")

        unique_classes = np.unique(y_train)
        print("Local classes:", unique_classes)

        if len(unique_classes) < 2:
            raise RuntimeError("❌ Only one class present. Cannot train classifier.")

        # ------------------------------
        # Model
        # ------------------------------
        model = LogisticRegression(
            max_iter=5000,
            n_jobs=-1,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        y_pred_local = model.predict(X_test)
        y_pred_real = y_pred_local + CLASS_BASE

        # ------------------------------
        # Metrics (REAL IDs)
        # ------------------------------
        metrics = compute_standardized_metrics(
            model_name="ClassificationEN_Model",
            y_train=y_train_real,
            y_test=y_test_real,
            y_pred=y_pred_real,
            label_names=sorted(set(y_train_real) | set(y_test_real)),
        )

        # Save model
        joblib.dump(model, MODEL_PATH)

        # Report
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("Classification EN Model Metrics\n\n")
            f.write(f"Accuracy: {metrics['accuracy']}\n")
            f.write(f"F1: {metrics['f1']}\n\n")
            f.write(classification_report(y_test_real, y_pred_real, zero_division=0))

        # Confusion matrix
        labels = sorted(set(y_train_real) | set(y_test_real))
        cm = confusion_matrix(y_test_real, y_pred_real, labels=labels)

        save_confusion_matrix(
            cm,
            labels=[str(x) for x in labels],
            out_path=CM_PATH,
            title="Classification_EN Confusion Matrix (REAL IDs)"
        )

        print("✅ Training finished successfully.")
        return model, metrics

    except Exception:
        traceback.print_exc()
        raise


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    model, metrics = train_classification_en_model()
    print("\nTraining completed.")
    print(metrics)
