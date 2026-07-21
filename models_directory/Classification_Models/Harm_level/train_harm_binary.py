#!/usr/bin/env python3
"""
Harm Binary Classification Model
"""

import json
import os
import sys
from pathlib import Path
import traceback

# Add workspace root to Python path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from project_paths import get_db_path
from models_directory.Classification_Models.Hierarchical_Classification_Model.Helper_Functions import (
    load_table,
    parse_embedding_series,
    compute_standardized_metrics,
)
from models_directory.Classification_Models.Maintainance import run_versioning

# ============================
# CONSTANTS
# ============================

SCRIPT_DIR = Path(__file__).resolve().parent

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

EMBED_COL = "embedding_text123"
TARGET_COL = "harm_level"


# ============================
# TRAIN FUNCTION
# ============================

def train_harm_binary(base_path: str | None = None, run_dir=None):
    """
    Train binary harm-level classifier.

    Returns:
        model, standardized_metrics (merged with roc_pr/warnings/artifacts
        when run_dir is supplied — see run_versioning.save_evaluation_artifacts)
    """

    try:
        if run_dir is None:
            run_dir = run_versioning.get_run_dir(run_versioning.generate_run_id())

        db_path = base_path or get_db_path()

        df_train = load_table(db_path, TRAIN_TABLE)
        df_test = load_table(db_path, TEST_TABLE)

        if df_train.empty or df_test.empty:
            raise ValueError("Training or testing dataset is empty")

        # Binary mapping
        df_train["harm_bin"] = df_train[TARGET_COL].apply(lambda x: 1 if x >= 4 else 0)
        df_test["harm_bin"] = df_test[TARGET_COL].apply(lambda x: 1 if x >= 4 else 0)

        X_train = parse_embedding_series(df_train[EMBED_COL])
        X_test = parse_embedding_series(df_test[EMBED_COL])

        y_train = df_train["harm_bin"].values
        y_test = df_test["harm_bin"].values

        # Train model
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Compute standardized metrics
        unique_labels = sorted(np.unique(y_train).tolist())
        standardized_metrics = compute_standardized_metrics(
            model_name="Harm_Binary",
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            label_names=unique_labels,
        )

        if run_dir is not None:
            # Positive class = "High" harm (1) — the clinically actionable class.
            y_proba = model.predict_proba(X_test)
            class_order = model.classes_.tolist()
            eval_result = run_versioning.save_evaluation_artifacts(
                run_dir=run_dir,
                model_name="Harm_Binary",
                y_true_display=y_test,
                y_pred_display=y_pred,
                display_labels=unique_labels,
                y_proba=y_proba,
                proba_class_order=class_order,
                positive_label=1,
            )
            model_entry = run_versioning.register_model_artifact(run_dir, "harm_binary", model, serializer="joblib")
            eval_result["artifacts"].append(model_entry)
            standardized_metrics.update(eval_result)

        print("[OK] Harm binary model trained successfully")

        return model, standardized_metrics

    except Exception as e:
        print(f"[ERROR] train_harm_binary failed: {str(e)}")
        traceback.print_exc()
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        return None, None


# ============================
# STANDALONE EXECUTION
# ============================

if __name__ == "__main__":
    model, metrics = train_harm_binary()
    if metrics:
        print(json.dumps(metrics, indent=4))
