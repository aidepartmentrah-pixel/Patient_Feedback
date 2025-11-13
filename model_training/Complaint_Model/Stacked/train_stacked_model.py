"""
train_stacked_model.py

Sequential stacked models (classical ML) using BERT embeddings.

Hierarchy:
    domain -> category -> sub_category -> classification_ar

Each stage receives:
 - embedding_text1 (768-d float vector)
 - plus predicted probabilities from previous stage(s) (one-hot-like probabilistic features)

Outputs (in same folder):
 - domain_stacked_logreg.pkl
 - category_stacked_logreg.pkl
 - sub_category_stacked_logreg.pkl
 - classification_ar_stacked_logreg.pkl
 - stacked_metrics.txt
 - confusion matrices png files

Usage:
    python train_stacked_model.py
"""

import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import traceback
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ----------------
HERE = Path(__file__).resolve().parent
DB_PATH = HERE.parent.parent / "patient_feedback_ml.db"   # ../../patient_feedback_ml.db
TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"
EMBED_COL = "embedding_text1"

# Hierarchy / order
STAGES = [
    ("domain", "domain_stacked_logreg.pkl", "domain_stacked_metrics.txt"),
    ("category", "category_stacked_logreg.pkl", None),
    ("sub_category", "sub_category_stacked_logreg.pkl", None),
    ("classification_ar", "classification_ar_stacked_logreg.pkl", None),
]

REPORT_PATH = HERE / "stacked_metrics.txt"
RANDOM_STATE = 42


# -------------- Helpers -----------------
def load_table(db_path: Path, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()
    return df


def parse_embedding_series(series: pd.Series) -> np.ndarray:
    """Convert Series of JSON arrays (or lists) into 2D numpy array."""
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


def save_confusion_matrix(cm: np.ndarray, labels, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0 if cm.max() != 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def clean_target(series: pd.Series) -> pd.Series:
    """Convert column to numeric and drop missing values; return int series with original index preserved."""
    s = pd.to_numeric(series, errors="coerce")
    n_missing = s.isna().sum()
    if n_missing > 0:
        print(f"⚠️  Found {n_missing} missing/invalid values in target; dropping them for this stage.")
    s = s.dropna()
    return s.astype(int)


# -------------- Main flow -----------------
def main():
    try:
        print(f"\nUsing database: {DB_PATH}")
        print("Loading train/test tables...")
        df_train = load_table(DB_PATH, TRAIN_TABLE)
        df_test = load_table(DB_PATH, TEST_TABLE)

        # Basic checks
        for table_name, df in (("train", df_train), ("test", df_test)):
            if EMBED_COL not in df.columns:
                raise KeyError(f"Embedding column '{EMBED_COL}' not found in {table_name} table")

        # parse embeddings for full train and test (we'll select rows by index per stage)
        X_train_all = parse_embedding_series(df_train[EMBED_COL])
        X_test_all = parse_embedding_series(df_test[EMBED_COL])

        print(f"Embeddings -> train: {X_train_all.shape}, test: {X_test_all.shape}")

        # Containers for storing stage models and the predicted-proba features for train & test
        trained_models = {}
        # dictionary mapping stage name -> train_proba_full (n_train x n_classes_of_stage)
        train_probas_full = {}
        test_probas_full = {}

        # Open report file
        with open(REPORT_PATH, "w", encoding="utf-8") as rep:
            rep.write("Stacked sequential model results (domain -> category -> sub_category -> classification_ar)\n")
            rep.write("=" * 80 + "\n\n")

            # sequentially train stages
            for i, (target, model_fname, _stage_report) in enumerate(STAGES):
                print(f"\n--- Stage {i+1}/{len(STAGES)}: {target} ---")
                rep.write(f"=== Stage: {target} ===\n")

                if target not in df_train.columns:
                    print(f"⚠️  Skipping {target}: not found in train table.")
                    rep.write(f"Skipped {target} (not present)\n\n")
                    continue

                # Clean target (drop missing)
                y_train_series = clean_target(df_train[target])
                y_test_series = clean_target(df_test[target])

                if y_train_series.shape[0] == 0 or y_test_series.shape[0] == 0:
                    print(f"⚠️  Skipping {target}: no data after cleaning for train or test.")
                    rep.write(f"Skipped {target}: no valid data after cleaning\n\n")
                    continue

                # Build feature matrices for this stage:
                # base features = embeddings (for the indices that have target)
                # extra features = concatenation of previously produced probas (for same indices)
                # For training we must select rows that have target labels; we will align embeddings/probas using index positions.

                train_idx = y_train_series.index
                test_idx = y_test_series.index

                X_train_stage = X_train_all[train_idx]
                X_test_stage = X_test_all[test_idx]

                # collect previous stages' probas as features (if any)
                if i > 0:
                    # concatenate previously produced probas for train and test (for all previous stages)
                    prev_train_feats = []
                    prev_test_feats = []
                    for prev_target in [s[0] for s in STAGES[:i]]:
                        # For train: if prev produced probas_full, take rows corresponding to current train_idx
                        if prev_target in train_probas_full:
                            prev_train_feats.append(train_probas_full[prev_target][train_idx])
                        else:
                            # If previous stage has no train_probas_full (rare), fallback to zeros
                            prev_train_feats.append(np.zeros((len(train_idx), 1)))
                        # For test:
                        if prev_target in test_probas_full:
                            prev_test_feats.append(test_probas_full[prev_target][test_idx])
                        else:
                            prev_test_feats.append(np.zeros((len(test_idx), 1)))

                    # horizontally concat
                    X_train_stage = np.hstack([X_train_stage] + prev_train_feats)
                    X_test_stage = np.hstack([X_test_stage] + prev_test_feats)

                print(f"Stage features shapes -> X_train_stage: {X_train_stage.shape}, X_test_stage: {X_test_stage.shape}")
                rep.write(f"Stage features shapes -> X_train_stage: {X_train_stage.shape}, X_test_stage: {X_test_stage.shape}\n")

                # Train logistic regression for this stage
                clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
                clf.fit(X_train_stage, y_train_series.to_numpy())

                # Save trained model
                model_path = HERE / model_fname
                joblib.dump(clf, model_path)
                trained_models[target] = clf
                print(f"✅ Saved model for {target}: {model_path.name}")
                rep.write(f"Saved model: {model_path.name}\n")

                # Evaluate on test rows
                y_pred = clf.predict(X_test_stage)
                y_prob = clf.predict_proba(X_test_stage)  # shape (n_test_rows, n_classes)
                acc = accuracy_score(y_test_series.to_numpy(), y_pred)
                f1m = f1_score(y_test_series.to_numpy(), y_pred, average="macro", zero_division=0)
                rep.write(f"Accuracy: {acc:.6f}\n")
                rep.write(f"F1 macro: {f1m:.6f}\n")
                rep.write("Classification report:\n")
                rep.write(classification_report(y_test_series.to_numpy(), y_pred, zero_division=0))
                rep.write("\n\n")

                # Save confusion matrix image (for test)
                labels_sorted = sorted(np.unique(np.concatenate((y_test_series.to_numpy(), y_train_series.to_numpy()))).tolist())
                cm = confusion_matrix(y_test_series.to_numpy(), y_pred)
                cm_path = HERE / f"{target}_stacked_confusion_matrix.png"
                save_confusion_matrix(cm, labels_sorted, cm_path, title=f"Confusion Matrix - {target} (stacked)")
                rep.write(f"Confusion matrix saved: {cm_path.name}\n\n")
                print(f"Saved confusion matrix: {cm_path.name}")

                # Now compute predicted probabilities to feed to later stages:
                # - for train: we compute probabilities for ALL train rows by applying the fitted model to X_train_all (+ previous probas)
                # - for test: we compute probabilities for ALL test rows by applying the fitted model to X_test_all (+ previous probas)
                # Build full-train feature matrix (for every train sample) to generate train_probas_full
                # NOTE: for rows without previous-stage probas (if previous stage didn't exist), we used zeros.

                # Build train full features (embedding + previous stage probas for ALL train rows)
                X_train_full = X_train_all.copy()
                if i > 0:
                    prev_feats_full = []
                    for prev_target in [s[0] for s in STAGES[:i]]:
                        if prev_target in train_probas_full:
                            prev_feats_full.append(train_probas_full[prev_target])
                        else:
                            prev_feats_full.append(np.zeros((X_train_all.shape[0], 1)))
                    X_train_full = np.hstack([X_train_full] + prev_feats_full)

                X_test_full = X_test_all.copy()
                if i > 0:
                    prev_feats_test_full = []
                    for prev_target in [s[0] for s in STAGES[:i]]:
                        if prev_target in test_probas_full:
                            prev_feats_test_full.append(test_probas_full[prev_target])
                        else:
                            prev_feats_test_full.append(np.zeros((X_test_all.shape[0], 1)))
                    X_test_full = np.hstack([X_test_full] + prev_feats_test_full)

                # Predict probabilities for all train and test rows (these will be used as features for next stage)
                train_proba_full = clf.predict_proba(X_train_full)
                test_proba_full = clf.predict_proba(X_test_full)

                train_probas_full[target] = train_proba_full
                test_probas_full[target] = test_proba_full

                print(f"Produced probability features for stage '{target}': train_probas_full shape {train_proba_full.shape}")
                rep.write(f"Produced probas shape: train {train_proba_full.shape}, test {test_proba_full.shape}\n\n")

            rep.write("Stacked pipeline training completed.\n")

        print("\n🎉 Stacked training complete. Report saved to:", REPORT_PATH)

    except Exception:
        print("❌ An error occurred during stacked training:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
