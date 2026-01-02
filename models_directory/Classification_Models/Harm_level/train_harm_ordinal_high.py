# train_harm_ordinal_high.py
"""
Ordinal Harm Model (HIGH PART)
Predicts harm levels: 4, 5, 6 → mapped to ordinal 0,1,2

Produces:
- Harm_OrdinalHighModel.pkl
- harm_high_report.txt
"""

import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import traceback
import mord
from sklearn.metrics import accuracy_score, f1_score, classification_report

from project_paths import get_db_path

# ---------------- PATH CONFIG ----------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Database path: 4 levels up (matches your other scripts)

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE  = "table_feedback_test"

# Use same embedding column as other scripts
EMBED_COL  = "embedding_text1"
TARGET_COL = "harm_level"

MODEL_PATH  = SCRIPT_DIR / "Harm_OrdinalHighModel.pkl"
REPORT_PATH = SCRIPT_DIR / "harm_high_report.txt"


# ------------ Load Table --------------
def load_table(db_path, table):
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


# ------------ Parse Embeddings --------------
def parse_embedding_series(series):
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


# ------------ TRAIN FUNCTION ----------------
def train_harm_ordinal_high(base_path: str | None = None):
    """
    Train ordinal logistic regression for harm levels 4–6.
    Returns:
        model: trained mord.LogisticIT model or None on failure
        metrics: dict with keys 'accuracy', 'f1_macro', 'report' or None on failure
    """
    try:
        DB_PATH = base_path or get_db_path()

        # Load data
        df_train = load_table(DB_PATH, TRAIN_TABLE)
        df_test  = load_table(DB_PATH, TEST_TABLE)

        # Keep only harm levels 4–6
        df_train = df_train[df_train[TARGET_COL].isin([4, 5, 6])]
        df_test  = df_test[df_test[TARGET_COL].isin([4, 5, 6])]

        # Parse embeddings
        X_train = parse_embedding_series(df_train[EMBED_COL])
        X_test  = parse_embedding_series(df_test[EMBED_COL])

        # Re-map labels: 4→0, 5→1, 6→2
        y_train = df_train[TARGET_COL].astype(int).to_numpy() - 4
        y_test  = df_test[TARGET_COL].astype(int).to_numpy() - 4

        # Quick diagnostics (prints helpful info; remove if noisy)
        print("HARM HIGH: X_train.shape", X_train.shape, "X_test.shape", X_test.shape)
        print("HARM HIGH: y_train dist:\n", pd.Series(y_train).value_counts().sort_index())
        print("HARM HIGH: y_test dist:\n", pd.Series(y_test).value_counts().sort_index())

        # Basic checks
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError("Embedding dimension mismatch between train and test.")
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            raise ValueError("NaN found in embeddings.")

        # Train ordinal model
        model = mord.LogisticIT()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred, zero_division=0)

        metrics = {
            "accuracy": acc,
            "f1_macro": f1,
            "report": report
        }

        # Save model and report
        joblib.dump(model, MODEL_PATH)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("HIGH Harm Ordinal Model\n")
            f.write(f"Accuracy: {acc}\n")
            f.write(f"F1 Macro: {f1}\n\n")
            f.write(report)

        print("High harm ordinal model training done.")
        return model, metrics

    except Exception:
        traceback.print_exc()
        return None, None


# ------------ STANDALONE EXECUTION ------------
if __name__ == "__main__":
    model, metrics = train_harm_ordinal_high()
    if metrics:
        print(json.dumps(metrics, indent=4))
