# train_harm_ordinal_low.py
"""
Ordinal Harm Model (LOW PART)
Predicts harm levels: 1, 2, 3 → mapped to ordinal 0,1,2

Produces:
- Harm_OrdinalLowModel.pkl
- harm_low_report.txt
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


# ---------------- PATH CONFIG ----------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Database path: 2 levels up
DB_PATH = (SCRIPT_DIR / ".." / ".." / "patient_feedback_ml.db").resolve()

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE  = "table_feedback_test"

EMBED_COL  = "embedding_text123"
TARGET_COL = "harm_level"

MODEL_PATH  = SCRIPT_DIR / "Harm_OrdinalLowModel.pkl"
REPORT_PATH = SCRIPT_DIR / "harm_low_report.txt"


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
def train_harm_ordinal_low():
    """
    Train ordinal logistic regression for harm levels 1–3.
    Returns:
        model: trained Mord LogisticIT model
        metrics: dict containing accuracy, f1, and classification report
    """
    metrics = {}

    try:
        df_train = load_table(DB_PATH, TRAIN_TABLE)
        df_test  = load_table(DB_PATH, TEST_TABLE)

        # Keep only harm levels 1–3
        df_train = df_train[df_train[TARGET_COL].isin([1, 2, 3])]
        df_test  = df_test[df_test[TARGET_COL].isin([1, 2, 3])]

        X_train = parse_embedding_series(df_train[EMBED_COL])
        X_test  = parse_embedding_series(df_test[EMBED_COL])

        # Convert harm 1→0, 2→1, 3→2
        y_train = df_train[TARGET_COL].astype(int).to_numpy() - 1
        y_test  = df_test[TARGET_COL].astype(int).to_numpy() - 1

        # Ordinal Logistic Regression
        model = mord.LogisticIT()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Metrics ---
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred, zero_division=0)

        metrics = {
            "accuracy": acc,
            "f1_macro": f1,
            "report": report,
        }

        # Save model
        joblib.dump(model, MODEL_PATH)

        # Save report file
        with open(REPORT_PATH, "w") as f:
            f.write("LOW Harm Ordinal Model Results\n")
            f.write(f"Accuracy: {acc}\n")
            f.write(f"F1 Macro: {f1}\n\n")
            f.write(report)

        print("Low harm ordinal model training done.")

        return model, metrics

    except Exception:
        traceback.print_exc()
        return None, None


# ------------ MAIN ----------------
if __name__ == "__main__":
    model, metrics = train_harm_ordinal_low()
    if metrics:
        print(metrics)
