"""
train_stage_model.py

Training script for "stage" using embedding_text1.

Outputs:
- stage_logreg.pkl
- stage_rf.pkl
- stage_metrics.txt
- stage_confusion_matrix_logreg.png
- stage_confusion_matrix_rf.png

Usage:
    python train_stage_model.py
"""

import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import traceback
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# ---------------- CONFIG ----------------
HERE = Path(__file__).resolve().parent
DB_PATH = HERE.parent / "patient_feedback_ml.db"
TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"
EMBED_COL = "embedding_text1"
TARGET_COL = "stage"

# Output paths
MODEL_LOGREG = HERE / "stage_logreg.pkl"
MODEL_RF = HERE / "stage_rf.pkl"
REPORT_PATH = HERE / "stage_metrics.txt"
CM_LOGREG = HERE / "stage_confusion_matrix_logreg.png"
CM_RF = HERE / "stage_confusion_matrix_rf.png"

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
    """Convert Series of JSON arrays or lists into 2D numpy array."""
    out = []
    for i, v in enumerate(series):
        if pd.isna(v):
            raise ValueError(f"Missing embedding at index {i}")
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v, dtype=float)
        else:
            try:
                arr = np.asarray(json.loads(v), dtype=float)
            except Exception as e:
                raise ValueError(f"Unable to parse embedding at row {i}: {e}")
        out.append(arr)

    lengths = [a.size for a in out]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent embedding lengths: {set(lengths)}")
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
        print("Loading train/test tables...")
        df_train = load_table(DB_PATH, TRAIN_TABLE)
        df_test = load_table(DB_PATH, TEST_TABLE)

        for name, df in (("train", df_train), ("test", df_test)):
            if EMBED_COL not in df.columns:
                raise KeyError(f"Embedding column '{EMBED_COL}' not found in {name}")
            if TARGET_COL not in df.columns:
                raise KeyError(f"Target column '{TARGET_COL}' not found in {name}")

        # Drop NaN targets early to prevent conversion errors
        df_train = df_train.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        df_test = df_test.dropna(subset=[TARGET_COL]).reset_index(drop=True)

        X_train = parse_embedding_series(df_train[EMBED_COL])
        X_test = parse_embedding_series(df_test[EMBED_COL])
        y_train = df_train[TARGET_COL].astype(int).to_numpy()
        y_test = df_test[TARGET_COL].astype(int).to_numpy()

        print(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

        # --- Train models ---
        logreg = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)

        print("Training Logistic Regression...")
        logreg.fit(X_train, y_train)
        y_pred_log = logreg.predict(X_test)

        print("Training Random Forest...")
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        # --- Evaluate ---
        results = {}
        for name, y_pred in (("LogisticRegression", y_pred_log), ("RandomForest", y_pred_rf)):
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
            report = classification_report(y_test, y_pred, zero_division=0)
            results[name] = {"accuracy": acc, "f1_macro": f1_macro, "report": report, "y_pred": y_pred}
            print(f"{name}: Accuracy={acc:.4f}, F1={f1_macro:.4f}")

        # --- Save models ---
        joblib.dump(logreg, MODEL_LOGREG)
        joblib.dump(rf, MODEL_RF)
        print("Models saved.")

        # --- Save metrics ---
        labels = sorted(np.unique(np.concatenate((y_train, y_test))).tolist())
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("Stage classification results\n\n")
            for name, res in results.items():
                f.write(f"=== Model: {name} ===\n")
                f.write(f"Accuracy: {res['accuracy']:.6f}\n")
                f.write(f"F1 macro: {res['f1_macro']:.6f}\n")
                f.write(res["report"] + "\n\n")
        print("Report saved.")

        # --- Confusion Matrices ---
        cm_log = confusion_matrix(y_test, results["LogisticRegression"]["y_pred"])
        cm_rf = confusion_matrix(y_test, results["RandomForest"]["y_pred"])

        save_confusion_matrix(cm_log, labels, CM_LOGREG, "Confusion Matrix - Logistic Regression")
        save_confusion_matrix(cm_rf, labels, CM_RF, "Confusion Matrix - Random Forest")
        print("Confusion matrices saved.")
        print("Training complete ✅")

    except Exception:
        print("Error during training:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
