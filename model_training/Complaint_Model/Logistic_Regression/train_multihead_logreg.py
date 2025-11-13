"""
train_multihead_logreg.py

Trains multiple Logistic Regression heads on shared embeddings for predicting:
- domain
- category
- sub_category
- classification_ar

Each head is trained independently using the same BERT embeddings.

Outputs:
    - {target}_logreg.pkl
    - multihead_logreg_metrics.txt
"""

import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import traceback
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

# ---------------- CONFIG ----------------
HERE = Path(__file__).resolve().parent
DB_PATH = HERE.parent.parent / "patient_feedback_ml.db"
TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"
EMBED_COL = "embedding_text1"
TARGET_COLS = ["domain", "category", "sub_category", "classification_ar"]
RANDOM_STATE = 42
REPORT_PATH = HERE / "multihead_logreg_metrics.txt"


# ---------------- HELPERS ----------------
def load_table(db_path: Path, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()
    return df


def parse_embedding_series(series: pd.Series) -> np.ndarray:
    """Convert a pandas Series of JSON arrays (or lists) into a 2D numpy array."""
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


def clean_target(series, target_name, df_name):
    """Convert column to numeric and drop missing values."""
    series = pd.to_numeric(series, errors="coerce")
    n_missing = series.isna().sum()
    if n_missing > 0:
        print(f"⚠️  {n_missing} missing or invalid values found in {df_name}['{target_name}'], dropping those rows.")
        series = series.dropna()
    return series.astype(int)


# ---------------- MAIN ----------------
def main():
    try:
        print(f"\nUsing database: {DB_PATH}")
        print("Loading train/test tables...")

        df_train = load_table(DB_PATH, TRAIN_TABLE)
        df_test = load_table(DB_PATH, TEST_TABLE)

        if EMBED_COL not in df_train.columns:
            raise KeyError(f"Embedding column '{EMBED_COL}' not in train table")

        X_train_all = parse_embedding_series(df_train[EMBED_COL])
        X_test_all = parse_embedding_series(df_test[EMBED_COL])

        print(f"✅ Embeddings loaded -> train: {X_train_all.shape}, test: {X_test_all.shape}\n")

        with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
            report_file.write("Multi-Head Logistic Regression Results\n")
            report_file.write("=" * 70 + "\n\n")

            for TARGET_COL in TARGET_COLS:
                print(f"\n=== Training head for: {TARGET_COL} ===")

                if TARGET_COL not in df_train.columns:
                    print(f"⚠️  Skipping {TARGET_COL}: not found in DB.")
                    continue

                # Clean target columns
                y_train = clean_target(df_train[TARGET_COL], TARGET_COL, "train")
                y_test = clean_target(df_test[TARGET_COL], TARGET_COL, "test")

                # Keep only aligned embeddings
                X_train = X_train_all[y_train.index]
                X_test = X_test_all[y_test.index]

                print(f"→ Shapes: X_train={X_train.shape}, y_train={y_train.shape}")

                if len(np.unique(y_train)) < 2:
                    print(f"⚠️  Skipping {TARGET_COL}: not enough label classes.")
                    continue

                # Train model
                logreg = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
                logreg.fit(X_train, y_train)

                y_pred = logreg.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
                rep = classification_report(y_test, y_pred, zero_division=0)

                # Save model
                model_path = HERE / f"{TARGET_COL}_logreg.pkl"
                joblib.dump(logreg, model_path)
                print(f"✅ Saved model: {model_path.name}")

                # Write to report
                report_file.write(f"=== {TARGET_COL.upper()} ===\n")
                report_file.write(f"Accuracy: {acc:.6f}\n")
                report_file.write(f"F1 macro: {f1_macro:.6f}\n")
                report_file.write(rep + "\n\n")

            report_file.write("All model heads trained successfully.\n")

        print("\n🎉 Multi-Head Logistic Regression training complete!")
        print(f"📄 Metrics saved to: {REPORT_PATH}")

    except Exception:
        print("❌ An error occurred during training:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
