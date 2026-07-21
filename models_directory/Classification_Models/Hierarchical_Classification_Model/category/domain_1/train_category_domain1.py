import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from models_directory.Classification_Models.Hierarchical_Classification_Model.Helper_Functions import (
    load_table,
    parse_embedding,
    parse_embedding_series,
    compute_metrics,
    compute_standardized_metrics,
)
from models_directory.Classification_Models.Maintainance import run_versioning
from project_paths import get_db_path


# ============================
# MAIN TRAIN FUNCTION
# ============================

def train_category_domain1(base_path=None, run_dir=None):
    """Train LR, RF, XGB for a given domain and return the winning model +
    standardized_metrics (merged with roc_pr/warnings/artifacts/candidate_selection
    when run_dir is supplied)."""
    if run_dir is None:
        run_dir = run_versioning.get_run_dir(run_versioning.generate_run_id())

    table_train="table_feedback_train"
    table_test="table_feedback_test"
    domain = 1
    db_path = get_db_path() if base_path is None else base_path

    # ---------- Load Data ----------
    df_train = load_table(db_path, table_train)
    df_test = load_table(db_path, table_test)

    df_train = df_train[df_train["domain"] == domain]
    df_test = df_test[df_test["domain"] == domain]

    print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")

    X_train = parse_embedding_series(df_train["embedding_text1"])
    X_test = parse_embedding_series(df_test["embedding_text1"])
    y_train = df_train["category"].astype(int).values
    y_test = df_test["category"].astype(int).values

    print(f"Labels: {np.unique(y_train).tolist()}")

    results = {}
    trained_models = {}
    all_preds = {}

    # ---------- Logistic Regression ----------
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=5000, class_weight="balanced")
    lr.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)
    results["lr"] = compute_metrics(y_test, lr_pred)
    trained_models["lr"] = lr
    all_preds["lr"] = lr_pred

    # ---------- Random Forest ----------
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    results["rf"] = compute_metrics(y_test, rf_pred)
    trained_models["rf"] = rf
    all_preds["rf"] = rf_pred

    # ---------- XGBoost ----------
    print("Training XGBoost...")
    # Map labels to 0..n
    label_to_temp = {v: i for i, v in enumerate(np.unique(y_train))}
    temp_to_label = {i: v for v, i in label_to_temp.items()}

    y_train_temp = np.array([label_to_temp[v] for v in y_train])
    y_test_temp = np.array([label_to_temp[v] for v in y_test])

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_to_temp),
        eval_metric="mlogloss",
        learning_rate=0.1,
        max_depth=6,
        n_estimators=300,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=42
    )
    xgb.fit(X_train, y_train_temp)

    preds_temp = xgb.predict(X_test)
    if preds_temp.ndim == 2:  # probabilities
        preds_temp = np.argmax(preds_temp, axis=1)
    preds_xgb = np.array([temp_to_label[int(v)] for v in preds_temp])
    results["xgb"] = compute_metrics(y_test, preds_xgb)
    trained_models["xgb"] = xgb
    all_preds["xgb"] = preds_xgb

    # ---------- Select Best Model by F1 ----------
    best_model_name = max(results.keys(), key=lambda k: results[k]["f1"])
    best_model = trained_models[best_model_name]
    best_pred = all_preds[best_model_name]
    
    print(f"\n Best model: {best_model_name} (F1={results[best_model_name]['f1']:.4f})")
    print(f"Training complete for domain {domain}")

    # ---------- Compute Standardized Metrics ----------
    unique_labels = sorted(np.unique(y_train).tolist())
    model_name = f"Category_Domain{domain}_{best_model_name}"
    standardized_metrics = compute_standardized_metrics(
        model_name=model_name,
        y_train=y_train,
        y_test=y_test,
        y_pred=best_pred,
        label_names=unique_labels,
    )
    standardized_metrics["candidate_selection"] = {
        name: results[name] for name in ("lr", "rf", "xgb")
    }

    # ---------- Versioned evaluation artifacts (winning model only) ----------
    # Only XGBoost is fit on the remapped 0..n space here (LR/RF are fit
    # directly on the real category values) — proba_class_order/
    # y_true_for_curves must reflect whichever space the winner actually
    # produced probabilities in, never assumed.
    y_proba = best_model.predict_proba(X_test)
    proba_class_order = best_model.classes_.tolist()
    y_true_for_curves = y_test_temp if best_model_name == "xgb" else y_test

    eval_result = run_versioning.save_evaluation_artifacts(
        run_dir=run_dir,
        model_name=model_name,
        y_true_display=y_test.tolist(),
        y_pred_display=best_pred.tolist() if hasattr(best_pred, "tolist") else list(best_pred),
        display_labels=unique_labels,
        y_proba=y_proba,
        proba_class_order=proba_class_order,
        y_true_for_curves=y_true_for_curves,
    )
    serializer = "xgboost_native" if best_model_name == "xgb" else "joblib"
    model_entry = run_versioning.register_model_artifact(run_dir, model_name, best_model, serializer=serializer)
    eval_result["artifacts"].append(model_entry)
    standardized_metrics.update(eval_result)

    print(f"Artifacts written to: {run_dir}")

    return best_model, standardized_metrics

# ============================
# STANDALONE RUN
# ============================

if __name__ == "__main__":
    models, metrics = train_category_domain1()
    print("\nMetrics per model:")
    print(json.dumps(metrics, indent=4))
