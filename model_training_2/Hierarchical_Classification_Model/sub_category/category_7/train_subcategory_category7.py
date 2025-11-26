import os
import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ============================
# PATHS
# ============================

DB_PATH = r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\model_training_2\patient_feedback_ml.db"

TABLE_TRAIN = "table_feedback_train"
TABLE_TEST = "table_feedback_test"

EMBED_COL = "embedding_text1"
CATEGORY_COL = "category"
SUBCAT_COL = "sub_category"

TARGET_CATEGORY = 7     # CATEGORY 7

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "vocab_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================
# HELPERS
# ============================

def load_table(db_path, table):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

def parse_embedding(v):
    if isinstance(v, list):
        return np.asarray(v, dtype=float)
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            return np.asarray(json.loads(s), dtype=float)
        return np.asarray([float(x) for x in s.split(",")], dtype=float)
    if isinstance(v, (bytes, bytearray)):
        arr = np.frombuffer(v, dtype=np.float32)
        return arr.astype(float)
    raise ValueError(f"Unexpected embedding type: {type(v)}")

def parse_embedding_series(series):
    return np.vstack([parse_embedding(v) for v in series])

# ============================
# MAIN
# ============================

def main():

    print("\n===== SUBCATEGORY MODEL (CATEGORY = 7) =====")

    df_train = load_table(DB_PATH, TABLE_TRAIN)
    df_test = load_table(DB_PATH, TABLE_TEST)

    # Filter to category=7
    df_train = df_train[df_train[CATEGORY_COL] == TARGET_CATEGORY]
    df_test = df_test[df_test[CATEGORY_COL] == TARGET_CATEGORY]

    # Keep only valid subcategories
    VALID_LABELS = {5, 15, 16, 18, 22, 29}
    df_train = df_train[df_train[SUBCAT_COL].isin(VALID_LABELS)]
    df_test = df_test[df_test[SUBCAT_COL].isin(VALID_LABELS)]

    print(f"Train rows after filtering → {len(df_train)}")
    print(f"Test rows after filtering → {len(df_test)}")

    # Parse embeddings
    X_train = parse_embedding_series(df_train[EMBED_COL])
    X_test = parse_embedding_series(df_test[EMBED_COL])

    # Labels
    y_train = df_train[SUBCAT_COL].astype(int).values
    y_test = df_test[SUBCAT_COL].astype(int).values

    unique_labels = sorted(np.unique(y_train).tolist())
    print(f"Subcategory labels used in TRAIN: {unique_labels}")

    # ============================
    # TRAIN MODELS
    # ============================

    print("Training LR...")
    lr = LogisticRegression(max_iter=5000, class_weight="balanced")
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(MODEL_DIR, "lr_subcat_cat7.pkl"))

    print("Training RF...")
    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_subcat_cat7.pkl"))

    # XGBoost mapping
    label_to_temp = {v: i for i, v in enumerate(unique_labels)}
    temp_to_label = {i: v for v, i in label_to_temp.items()}

    y_train_temp = np.array([label_to_temp[v] for v in y_train])
    y_test_temp = np.array([label_to_temp[v] for v in y_test])

    print("Training XGB...")
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
    xgb.save_model(os.path.join(MODEL_DIR, "xgb_subcat_cat7.json"))

    # ============================
    # EVALUATION
    # ============================

    report_path = os.path.join(MODEL_DIR, "report_subcat_cat7.txt")
    with open(report_path, "w", encoding="utf-8") as f:

        f.write("=== SUBCATEGORY MODEL (CATEGORY = 7) ===\n\n")

        for name, model in [
            ("Logistic Regression", lr),
            ("Random Forest", rf),
            ("XGBoost", xgb)
        ]:
            f.write(f"\n---- {name} ----\n")

            if name == "XGBoost":
                preds_raw = model.predict(X_test)
                if preds_raw.ndim == 2:
                    preds_temp = np.argmax(preds_raw, axis=1)
                else:
                    preds_temp = preds_raw.astype(int)
                preds = np.array([temp_to_label[int(v)] for v in preds_temp])
            else:
                preds = model.predict(X_test)

            f.write("\nClassification Report:\n")
            f.write(classification_report(y_test, preds))

            cm = confusion_matrix(y_test, preds)
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\n")

    print("==========================================")
    print("✔ SUBCATEGORY MODEL (CATEGORY=7) TRAINED")
    print("✔ MODELS SAVED IN:", MODEL_DIR)
    print("✔ REPORT SAVED:", report_path)
    print("==========================================\n")


if __name__ == "__main__":
    main()
