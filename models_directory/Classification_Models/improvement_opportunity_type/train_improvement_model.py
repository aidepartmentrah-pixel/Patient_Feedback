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
import traceback
import mord
import sys

# --------------------------------------------------
# Project imports
# --------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from models_directory.Classification_Models.Hierarchical_Classification_Model.Helper_Functions import (
    compute_standardized_metrics,
)
from models_directory.Classification_Models.Maintainance import run_versioning

# --------------------------------------------------
# Paths & Config
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DB_PATH = SCRIPT_DIR.parent.parent / "patient_feedback_ml.db"

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

EMBED_COL = "embedding_text123"
TARGET_COL = "improvement_opportunity_type"


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
        print(f"[WARNING] No samples for class {target_value}. Skipping oversampling.")
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


# --------------------------------------------------
# TRAINING FUNCTION
# --------------------------------------------------
def train_improvement_model(base_path=None, run_dir=None):
    """
    Trains ordinal improvement model using standardized metrics. Returns
    model, standardized_metrics (merged with roc_pr/warnings/artifacts when
    run_dir is supplied).
    """
    if run_dir is None:
        run_dir = run_versioning.get_run_dir(run_versioning.generate_run_id())

    DB_PATH = base_path or _DEFAULT_DB_PATH

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
        model_name = "Improvement_Ordinal_Model"
        standardized_metrics = compute_standardized_metrics(
            model_name=model_name,
            y_train=df_train[TARGET_COL].to_numpy(),
            y_test=y_test_orig,
            y_pred=y_pred_orig,
            label_names=[1, 2, 3],
        )

        # ---------- Versioned evaluation artifacts ----------
        y_proba = model.predict_proba(X_test)
        proba_class_order = model.classes_.tolist() if hasattr(model, "classes_") else sorted(np.unique(y_train).tolist())

        eval_result = run_versioning.save_evaluation_artifacts(
            run_dir=run_dir,
            model_name=model_name,
            y_true_display=y_test_orig.tolist(),
            y_pred_display=y_pred_orig.tolist(),
            display_labels=[1, 2, 3],
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


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    model, metrics = train_improvement_model()
    print("\nTraining completed.")
    print(metrics)
