import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import traceback
import mord
from sklearn.dummy import DummyClassifier
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
# Paths
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR.parent.parent / "patient_feedback_ml.db"

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

EMBED_COL = "embedding_text123"
TARGET_COL = "feedback_type"

MODEL_PATH = SCRIPT_DIR / "FeedbackType_OrdinalModel.pkl"
REPORT_PATH = SCRIPT_DIR / "feedback_type_metrics.txt"
CM_PATH = SCRIPT_DIR / "feedback_type_confusion_matrix.png"


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
# TRAINING
# --------------------------------------------------
def train_feedback_type_model():
    try:
        df_train = load_table(DB_PATH, TRAIN_TABLE)
        df_test = load_table(DB_PATH, TEST_TABLE)

        df_train = df_train[df_train[TARGET_COL].isin([1, 2, 3, 4])]
        df_test = df_test[df_test[TARGET_COL].isin([1, 2, 3, 4])]

        X_train = parse_embedding_series(df_train[EMBED_COL])
        X_test = parse_embedding_series(df_test[EMBED_COL])

        y_train = df_train[TARGET_COL].astype(int).to_numpy()
        y_test = df_test[TARGET_COL].astype(int).to_numpy()

        unique_classes = np.unique(y_train)

        # ----------------------------------------
        # Handle degenerate case (only 1 class)
        # ----------------------------------------
        if len(unique_classes) == 1:
            print("⚠️ Only one class found. Using DummyClassifier.")
            model = DummyClassifier(strategy="most_frequent")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            # Ordinal classification
            model = mord.LogisticIT()
            model.fit(X_train, y_train - 1)
            y_pred = model.predict(X_test) + 1

        metrics = compute_standardized_metrics(
            model_name="FeedbackType_Ordinal_Model",
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            label_names=[1, 2, 3, 4],
        )

        joblib.dump(model, MODEL_PATH)

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("Feedback Type Model Metrics\n\n")
            f.write(f"Accuracy: {metrics['accuracy']}\n")
            f.write(f"F1: {metrics['f1']}\n\n")
            f.write(classification_report(y_test, y_pred, zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        save_confusion_matrix(
            cm,
            labels=["Improvement", "Notice", "Critique", "Other"],
            out_path=CM_PATH,
            title="Feedback Type Confusion Matrix"
        )

        return model, metrics

    except Exception:
        traceback.print_exc()
        raise


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    model, metrics = train_feedback_type_model()
    print("\nTraining completed.")
    print(metrics)
