import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import traceback
from sklearn.linear_model import LogisticRegression
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
# CONFIG
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DB_PATH = SCRIPT_DIR.parent.parent / "patient_feedback_ml.db"

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

EMBED_COL = "embedding_text123"
TARGET_COL = "classification_en"

# IMPORTANT: BASE ID
CLASS_BASE = 78

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


# --------------------------------------------------
# TRAINING
# --------------------------------------------------
def train_classification_en_model(base_path=None, run_dir=None):
    if run_dir is None:
        run_dir = run_versioning.get_run_dir(run_versioning.generate_run_id())

    DB_PATH = base_path or _DEFAULT_DB_PATH

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
            raise RuntimeError("[ERROR] Only one class present. Cannot train classifier.")

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
        model_name = "ClassificationEN_Model"
        labels = sorted(set(y_train_real) | set(y_test_real))
        metrics = compute_standardized_metrics(
            model_name=model_name,
            y_train=y_train_real,
            y_test=y_test_real,
            y_pred=y_pred_real,
            label_names=labels,
        )

        # ---------- Versioned evaluation artifacts ----------
        # Display space: real classification_en IDs. Proba/curve space: the
        # CLASS_BASE-shifted local space the model was actually fit on
        # (model.classes_, never assumed).
        y_proba = model.predict_proba(X_test)
        proba_class_order = model.classes_.tolist() if hasattr(model, "classes_") else sorted(np.unique(y_train).tolist())

        eval_result = run_versioning.save_evaluation_artifacts(
            run_dir=run_dir,
            model_name=model_name,
            y_true_display=y_test_real.tolist(),
            y_pred_display=y_pred_real.tolist(),
            display_labels=labels,
            y_proba=y_proba,
            proba_class_order=proba_class_order,
            y_true_for_curves=y_test,
        )
        model_entry = run_versioning.register_model_artifact(run_dir, model_name, model, serializer="joblib")
        eval_result["artifacts"].append(model_entry)
        metrics.update(eval_result)

        print(f"Training finished successfully. Artifacts written to: {run_dir}")
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
