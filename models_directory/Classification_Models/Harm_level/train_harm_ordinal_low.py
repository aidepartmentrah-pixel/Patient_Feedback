# train_harm_ordinal_low.py
"""
Ordinal Harm Model (LOW PART)
Predicts harm levels: 1, 2, 3 → mapped to ordinal 0,1,2

Writes versioned evaluation artifacts + model into run_dir (see
run_versioning.py) — no longer writes to a fixed live path automatically.
"""

import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import traceback
import mord

from project_paths import get_db_path
from models_directory.Classification_Models.Hierarchical_Classification_Model.Helper_Functions import (
    compute_standardized_metrics,
)
from models_directory.Classification_Models.Maintainance import run_versioning


# ---------------- PATH CONFIG ----------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Database path: 2 levels up (default; overridable via base_path)
_DEFAULT_DB_PATH = (SCRIPT_DIR / ".." / ".." / "patient_feedback_ml.db").resolve()

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE  = "table_feedback_test"

EMBED_COL  = "embedding_text123"
TARGET_COL = "harm_level"


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
def train_harm_ordinal_low(base_path: str | None = None, run_dir=None):
    """
    Train ordinal logistic regression for harm levels 1–3.
    Returns:
        model: trained Mord LogisticIT model
        metrics: dict (merged with roc_pr/warnings/artifacts when run_dir is
                 supplied), or None on failure
    """
    try:
        if run_dir is None:
            run_dir = run_versioning.get_run_dir(run_versioning.generate_run_id())

        DB_PATH = base_path or _DEFAULT_DB_PATH

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

        # Compute standardized metrics using original (non-remapped) labels
        y_test_orig = df_test[TARGET_COL].astype(int).to_numpy()
        y_pred_orig = y_pred + 1  # Remap back to 1-3 range
        unique_labels = sorted([1, 2, 3])
        
        model_name = "Harm_Ordinal_Low"
        standardized_metrics = compute_standardized_metrics(
            model_name=model_name,
            y_train=df_train[TARGET_COL].astype(int).to_numpy(),
            y_test=y_test_orig,
            y_pred=y_pred_orig,
            label_names=unique_labels,
        )

        # ---------- Versioned evaluation artifacts ----------
        y_proba = model.predict_proba(X_test)
        proba_class_order = model.classes_.tolist() if hasattr(model, "classes_") else sorted(np.unique(y_train).tolist())

        eval_result = run_versioning.save_evaluation_artifacts(
            run_dir=run_dir,
            model_name=model_name,
            y_true_display=y_test_orig.tolist(),
            y_pred_display=y_pred_orig.tolist(),
            display_labels=unique_labels,
            y_proba=y_proba,
            proba_class_order=proba_class_order,
            y_true_for_curves=y_test,
        )
        model_entry = run_versioning.register_model_artifact(run_dir, model_name, model, serializer="joblib")
        eval_result["artifacts"].append(model_entry)
        standardized_metrics.update(eval_result)

        print(f"Low harm ordinal model training done. Artifacts written to: {run_dir}")

        return model, standardized_metrics

    except Exception:
        traceback.print_exc()
        return None, None


# ------------ MAIN ----------------
if __name__ == "__main__":
    model, metrics = train_harm_ordinal_low()
    if metrics:
        print(metrics)
