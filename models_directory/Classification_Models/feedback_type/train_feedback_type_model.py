import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import traceback
import mord
from sklearn.dummy import DummyClassifier
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
# Paths
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR.parent.parent / "patient_feedback_ml.db"

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

EMBED_COL = "embedding_text123"
TARGET_COL = "feedback_type"

FEEDBACK_TYPE_NAMES = {1: "Improvement", 2: "Notice", 3: "Critique", 4: "Other"}


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
def train_feedback_type_model(run_dir=None):
    try:
        if run_dir is None:
            run_dir = run_versioning.get_run_dir(run_versioning.generate_run_id())

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
            print("[WARNING] Only one class found. Using DummyClassifier.")
            model = DummyClassifier(strategy="most_frequent")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # No shift here — DummyClassifier is fit directly on the
            # original 1..4 label space, unlike the ordinal branch below.
            y_true_for_curves = y_test
        else:
            # Ordinal classification. mord.LogisticIT is fit on labels
            # shifted to 0-indexed (y_train - 1); predictions are shifted
            # back (+1) for reporting. Its predict_proba() columns are in
            # the SHIFTED space, so ROC/PR must use the shifted true labels
            # too — never the display-space y_test.
            model = mord.LogisticIT()
            model.fit(X_train, y_train - 1)
            y_pred = model.predict(X_test) + 1
            y_true_for_curves = y_test - 1

        metrics = compute_standardized_metrics(
            model_name="FeedbackType_Ordinal_Model",
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            label_names=[1, 2, 3, 4],
        )

        # Display space: friendly names, matching the original confusion
        # matrix's implicit 1=Improvement/2=Notice/3=Critique/4=Other mapping.
        y_test_named = [FEEDBACK_TYPE_NAMES[v] for v in y_test]
        y_pred_named = [FEEDBACK_TYPE_NAMES[v] for v in y_pred]
        display_labels = [FEEDBACK_TYPE_NAMES[v] for v in [1, 2, 3, 4]]

        # ROC/PR probability space: whatever the model itself actually
        # produced (never assumed) — falls back to sorted unique training
        # labels only if the estimator doesn't expose classes_ at all.
        y_proba = model.predict_proba(X_test)
        raw_class_order = getattr(model, "classes_", None)
        if raw_class_order is None:
            class_order = sorted(np.unique(y_true_for_curves).tolist())
        else:
            class_order = raw_class_order.tolist() if hasattr(raw_class_order, "tolist") else list(raw_class_order)

        eval_result = run_versioning.save_evaluation_artifacts(
            run_dir=run_dir,
            model_name="FeedbackType_Ordinal_Model",
            y_true_display=y_test_named,
            y_pred_display=y_pred_named,
            display_labels=display_labels,
            y_proba=y_proba,
            proba_class_order=class_order,
            y_true_for_curves=y_true_for_curves,
        )
        model_entry = run_versioning.register_model_artifact(run_dir, "feedback_type", model, serializer="joblib")
        eval_result["artifacts"].append(model_entry)
        metrics.update(eval_result)

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
