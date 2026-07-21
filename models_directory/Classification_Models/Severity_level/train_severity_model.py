"""
train_severity_model.py

Training script for SEVERITY LEVEL (ordinal 1–4)
using embedding_text123 only.

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

# Import standardized metrics helper
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from models_directory.Classification_Models.Hierarchical_Classification_Model.Helper_Functions import (
    compute_standardized_metrics,
)
from models_directory.Classification_Models.Maintainance import run_versioning


SCRIPT_DIR = Path(__file__).resolve().parent

# TWO LEVELS UP (default; overridable via base_path)
_DEFAULT_DB_PATH = SCRIPT_DIR.parent.parent / "patient_feedback_ml.db"

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

EMBED_COL = "embedding_text123"
TARGET_COL = "severity_level"


# -------------- Helpers -----------------
def load_table(db_path: Path, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()
    return df


def parse_embedding_series(series: pd.Series) -> np.ndarray:
    out = []
    for i, v in enumerate(series):
        try:
            if isinstance(v, np.ndarray):
                arr = v.astype(float)
            elif isinstance(v, (list, tuple)):
                arr = np.asarray(v, dtype=float)
            elif isinstance(v, (bytes, bytearray)):
                arr = np.frombuffer(v, dtype=np.float32).astype(float)
            elif isinstance(v, str):
                arr = np.asarray(json.loads(v), dtype=float)
            else:
                raise ValueError(f"Unknown format: {type(v)}")
            out.append(arr)
        except Exception as e:
            raise ValueError(f"Error parsing embedding at row {i}: {e}")

    lengths = {len(a) for a in out}
    if len(lengths) != 1:
        raise ValueError(f"Embedding lengths not equal: {lengths}")

    return np.vstack(out)


# ---------------------------------------------------------
# 🚀 FUNCTION VERSION — THIS IS WHAT YOU NEED
# ---------------------------------------------------------
def train_severity_model(base_path=None, run_dir=None):
    """
    Trains the severity model and returns:
        model, standardized_metrics_dict (merged with roc_pr/warnings/artifacts
        when run_dir is supplied)
    """
    if run_dir is None:
        run_dir = run_versioning.get_run_dir(run_versioning.generate_run_id())

    DB_PATH = base_path or _DEFAULT_DB_PATH

    try:
        df_train = load_table(DB_PATH, TRAIN_TABLE)
        df_test = load_table(DB_PATH, TEST_TABLE)

        # Filter 1–4
        df_train = df_train[df_train[TARGET_COL].isin([1, 2, 3, 4])]
        df_test = df_test[df_test[TARGET_COL].isin([1, 2, 3, 4])]

        X_train = parse_embedding_series(df_train[EMBED_COL])
        X_test = parse_embedding_series(df_test[EMBED_COL])

        y_train = df_train[TARGET_COL].astype(int).to_numpy() - 1
        y_test = df_test[TARGET_COL].astype(int).to_numpy() - 1

        # Train
        model = mord.LogisticIT()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute standardized metrics using original (non-remapped) labels
        y_test_orig = df_test[TARGET_COL].astype(int).to_numpy()
        y_pred_orig = y_pred + 1  # Remap back to 1-4 range
        unique_labels = sorted([1, 2, 3, 4])
        
        model_name = "Severity_Model"
        standardized_metrics = compute_standardized_metrics(
            model_name=model_name,
            y_train=df_train[TARGET_COL].astype(int).to_numpy(),
            y_test=y_test_orig,
            y_pred=y_pred_orig,
            label_names=unique_labels,
        )

        # ---------- Versioned evaluation artifacts ----------
        # Display space: original 1-4 severity levels (fixes a pre-existing
        # bug where the old confusion-matrix image used the shifted 0-3
        # space while the text report used 1-4 — now both consistently use
        # display space).
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

        print(f"Artifacts written to: {run_dir}")

        return model, standardized_metrics

    except Exception:
        traceback.print_exc()
        raise


# ---------------------------------------------------------
# Legacy CLI entry point
# ---------------------------------------------------------
def main():
    model, metrics = train_severity_model()
    print("\nTraining Completed.")
    print(metrics)


if __name__ == "__main__":
    main()
