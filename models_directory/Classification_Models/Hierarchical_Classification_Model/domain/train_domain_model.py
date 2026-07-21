import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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
# TRAIN FUNCTION
# ============================

def train_domain_models(base_path=None, run_dir=None):
    """
    Train Logistic Regression, Random Forest, XGBoost for domain classification.
    Returns trained model and standardized_metrics (merged with
    roc_pr/warnings/artifacts/candidate_selection when run_dir is supplied).
    """

    if run_dir is None:
        run_dir = run_versioning.get_run_dir(run_versioning.generate_run_id())

    DB_PATH = get_db_path() if base_path is None else base_path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_DIR = os.path.join(SCRIPT_DIR, "vocab_models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    TABLE_TRAIN = "table_feedback_train"
    TABLE_TEST = "table_feedback_test"
    EMBED_COL = "embedding_text1"
    LABEL_COL = "domain"

    LABEL_MAP_FILE = os.path.join(SCRIPT_DIR, "label_map_domain.json")

    # ---------- Load Data ----------
    print(f"Using database: {DB_PATH}")
    print("Loading train/test tables...")

    df_train = load_table(DB_PATH, TABLE_TRAIN)
    df_test = load_table(DB_PATH, TABLE_TEST)

    X_train = parse_embedding_series(df_train[EMBED_COL])
    X_test = parse_embedding_series(df_test[EMBED_COL])

    y_train_raw = df_train[LABEL_COL].astype(int)
    y_test_raw = df_test[LABEL_COL].astype(int)

    # Encode labels
    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    # Save label map
    with open(LABEL_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "global_labels": le.classes_.tolist(),
            "local_labels": le.transform(le.classes_).tolist()
        }, f, indent=4)

    label_names = le.classes_

    # ---------- Manual Class Weights ----------
    manual_weights = {
        1: 1.0,
        2: 0.35,
        3: 0.90
    }
    class_weight_vector = {
        le.transform([cls])[0]: weight
        for cls, weight in manual_weights.items()
    }
    print("Using manual class weights:", class_weight_vector)

    # ---------- Train Models ----------
    trained_models = {}
    results = {}
    all_preds = {}

    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=5000, class_weight=class_weight_vector)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results["lr"] = compute_metrics(y_test, lr_pred)
    trained_models["lr"] = lr
    all_preds["lr"] = lr_pred

    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=500, class_weight=class_weight_vector, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["rf"] = compute_metrics(y_test, rf_pred)
    trained_models["rf"] = rf
    all_preds["rf"] = rf_pred

    # XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        learning_rate=0.1,
        max_depth=6,
        n_estimators=300,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=42
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    results["xgb"] = compute_metrics(y_test, xgb_pred)
    trained_models["xgb"] = xgb
    all_preds["xgb"] = xgb_pred

    # ---------- Select Best Model by F1 ----------
    best_model_name = max(results.keys(), key=lambda k: results[k]["f1"])
    best_model = trained_models[best_model_name]
    best_pred = all_preds[best_model_name]

    print(f"\n Best model: {best_model_name} (F1={results[best_model_name]['f1']:.4f})")

    # ---------- Compute Standardized Metrics ----------
    model_name = f"Domain_{best_model_name}"
    standardized_metrics = compute_standardized_metrics(
        model_name=model_name,
        y_train=y_train,
        y_test=y_test,
        y_pred=best_pred,
        label_names=label_names.tolist(),
    )
    # Candidate-selection transparency: all 3 candidates' selection metrics
    # are recorded even though only the winner gets full run artifacts below.
    standardized_metrics["candidate_selection"] = {
        name: results[name] for name in ("lr", "rf", "xgb")
    }

    # ---------- Versioned evaluation artifacts (winning model only) ----------
    # Display space: original domain label values (le decodes the encoded
    # 0..n-1 space back to what's actually stored in dbo.APP_IncidentCase).
    y_test_display = le.inverse_transform(y_test)
    y_pred_display = le.inverse_transform(best_pred)

    # ROC/PR probability space: whatever the winning model actually produced
    # (model.classes_ is the encoded 0..n-1 space all 3 candidates share here,
    # never assumed to equal range(n) even though it will in practice).
    y_proba = best_model.predict_proba(X_test)
    class_order = best_model.classes_.tolist()

    eval_result = run_versioning.save_evaluation_artifacts(
        run_dir=run_dir,
        model_name=model_name,
        y_true_display=y_test_display.tolist(),
        y_pred_display=y_pred_display.tolist(),
        display_labels=label_names.tolist(),
        y_proba=y_proba,
        proba_class_order=class_order,
        y_true_for_curves=y_test,
    )
    serializer = "xgboost_native" if best_model_name == "xgb" else "joblib"
    model_entry = run_versioning.register_model_artifact(run_dir, model_name, best_model, serializer=serializer)
    eval_result["artifacts"].append(model_entry)
    standardized_metrics.update(eval_result)

    print("\n===============================")
    print(" DOMAIN MODELS TRAINED WITH MANUAL WEIGHTS")
    print("===============================")
    print(f"Winning model: {best_model_name}, artifacts written to: {run_dir}")

    return best_model, standardized_metrics

# ============================
# STANDALONE RUN
# ============================

if __name__ == "__main__":
    models, metrics = train_domain_models()
    print("\nMetrics per model:")
    print(json.dumps(metrics, indent=4))

# Patient_Feedback\models_directory\patient_feedback_ml (The database link from the Top level of the project)
# Patient_Feedback\models_directory\Classification_Models\Hierarchical_Classification_Model\domain (directory of this file)
# Patient_Feedback\models_directory\Classification_Models\Maintainance (Directory of train all)
# What is a programming solution, to refactor the function, telling it where it is being called.
# Or we can at any point go from the Main folder ""Patient_Feedback" in a relative path an reach the database
# How to refactor the function to make it done