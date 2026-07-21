# train_harm_ordinal_high.py
"""
Ordinal Harm Model (HIGH PART)
Predicts harm levels: 4, 5, 6 → mapped to ordinal 0,1,2

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

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE  = "table_feedback_test"

# Use same embedding column as other scripts
EMBED_COL  = "embedding_text1"
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
def train_harm_ordinal_high(base_path: str | None = None, run_dir=None):
    """
    Train ordinal logistic regression for harm levels 4–6.
    Returns:
        model: trained mord.LogisticIT model or None on failure
        metrics: dict (merged with roc_pr/warnings/artifacts when run_dir is
                 supplied) or None on failure
    """
    try:
        if run_dir is None:
            run_dir = run_versioning.get_run_dir(run_versioning.generate_run_id())

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

        # Compute standardized metrics using original (non-remapped) labels
        y_test_orig = df_test[TARGET_COL].astype(int).to_numpy()
        y_pred_orig = y_pred + 4  # Remap back to 4-6 range
        unique_labels = sorted([4, 5, 6])
        
        model_name = "Harm_Ordinal_High"
        standardized_metrics = compute_standardized_metrics(
            model_name=model_name,
            y_train=df_train[TARGET_COL].astype(int).to_numpy(),
            y_test=y_test_orig,
            y_pred=y_pred_orig,
            label_names=unique_labels,
        )

        # ---------- Versioned evaluation artifacts ----------
        # Display space: original 4-6 harm levels. Proba/curve space: the
        # shifted 0-2 space the model was actually fit on (model.classes_,
        # never assumed).
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

        print(f"High harm ordinal model training done. Artifacts written to: {run_dir}")
        return model, standardized_metrics

    except Exception:
        traceback.print_exc()
        return None, None


# ------------ STANDALONE EXECUTION ------------
if __name__ == "__main__":
    model, metrics = train_harm_ordinal_high()
    if metrics:
        print(json.dumps(metrics, indent=4))
