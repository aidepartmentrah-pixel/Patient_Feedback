import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import  classification_report,confusion_matrix
import joblib

from models_directory.Classification_Models.Hierarchical_Classification_Model.Helper_Functions import load_table,parse_embedding,parse_embedding_series,compute_metrics
from project_paths import get_db_path


# ============================
# HELPERS
# ============================



def train_subcategory_cat6(base_path=None):
    """
    Train Logistic Regression, Random Forest, and XGBoost for subcategories of CATEGORY=1.
    Returns trained models and metrics dictionary.
    """

    # ---------- Paths ----------
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = get_db_path() if base_path is None else base_path

    MODEL_DIR = os.path.join(SCRIPT_DIR, "vocab_models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    TABLE_TRAIN = "table_feedback_train"
    TABLE_TEST = "table_feedback_test"
    EMBED_COL = "embedding_text1"
    CATEGORY_COL = "category"
    SUBCAT_COL = "sub_category"
    TARGET_CATEGORY = 6

    REPORT_FILE = os.path.join(MODEL_DIR, "report_subcat_cat6.txt")

    # ---------- Load Data ----------
    print("Loading train/test tables...")
    df_train = load_table(DB_PATH, TABLE_TRAIN)
    df_test = load_table(DB_PATH, TABLE_TEST)

    # Filter by category
    df_train = df_train[df_train[CATEGORY_COL] == TARGET_CATEGORY]
    df_test = df_test[df_test[CATEGORY_COL] == TARGET_CATEGORY]

    print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")

    X_train = parse_embedding_series(df_train[EMBED_COL])
    X_test = parse_embedding_series(df_test[EMBED_COL])
    y_train = df_train[SUBCAT_COL].astype(int).values
    y_test = df_test[SUBCAT_COL].astype(int).values

    unique_labels = sorted(np.unique(y_train).tolist())
    mask = np.isin(y_test, unique_labels)
    X_test = X_test[mask]
    y_test = y_test[mask]

    unique_labels = sorted(np.unique(y_train).tolist())
    print(f"Subcategory labels used: {unique_labels}")

    trained_models = {}
    results = {}

    # ---------- Logistic Regression ----------
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=5000, class_weight="balanced")
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(MODEL_DIR, "lr_subcat_cat6.pkl"))
    lr_pred = lr.predict(X_test)
    results["LogisticRegression"] = compute_metrics(y_test, lr_pred, all_labels=unique_labels)

    trained_models["lr"] = lr

    # ---------- Random Forest ----------
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_subcat_cat6.pkl"))
    rf_pred = rf.predict(X_test)
    results["RandomForest"] = compute_metrics(y_test, rf_pred, all_labels=unique_labels)

    trained_models["rf"] = rf

    # ---------- XGBoost ----------
    print("Training XGBoost...")
    label_to_temp = {v: i for i, v in enumerate(unique_labels)}
    temp_to_label = {i: v for v, i in label_to_temp.items()}

    y_train_temp = np.array([label_to_temp[v] for v in y_train])
    y_test_temp = np.array([label_to_temp[v] for v in y_test])

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(unique_labels),
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
    xgb.save_model(os.path.join(MODEL_DIR, "xgb_subcat_cat6.json"))

    preds_temp = xgb.predict(X_test)
    if preds_temp.ndim == 2:
        preds_temp = np.argmax(preds_temp, axis=1)
    preds_xgb = np.array([temp_to_label[int(v)] for v in preds_temp])
    results["XGBoost"] = compute_metrics(y_test, preds_xgb, all_labels=unique_labels)
    trained_models["xgb"] = xgb

    # ---------- Generate Report ----------
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=== SUBCATEGORY MODEL (CATEGORY = 1) ===\n\n")
        for name, model in [("Logistic Regression", lr), ("Random Forest", rf), ("XGBoost", xgb)]:
            f.write(f"\n---- {name} ----\n")
            if name == "XGBoost":
                preds = preds_xgb
            elif name == "Random Forest":
                preds = rf_pred
            else:
                preds = lr_pred

            f.write("\nClassification Report:\n")
            f.write(classification_report(y_test, preds, zero_division=0))
            cm = confusion_matrix(y_test, preds)
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\n")

    print("==========================================")
    print("✔ SUBCATEGORY MODEL (CATEGORY=6) TRAINED")
    print("✔ MODELS SAVED IN:", MODEL_DIR)
    print("✔ REPORT SAVED:", REPORT_FILE)
    print("==========================================\n")

    return trained_models, results

# ============================
# STANDALONE RUN
# ============================

if __name__ == "__main__":
    models, metrics = train_subcategory_cat6()
    print("\nMetrics per model:")
    print(json.dumps(metrics, indent=4))
