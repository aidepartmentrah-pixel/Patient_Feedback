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

DB_PATH = r"/models_directory\patient_feedback_ml.db"

TABLE_TRAIN = "table_feedback_train"
TABLE_TEST = "table_feedback_test"

EMBED_COL = "embedding_text1"
DOMAIN_COL = "domain"
CATEGORY_COL = "category"   # <-- labels here are 2 and 3

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
    """Accept: list, JSON string, comma-separated string, bytes."""
    if isinstance(v, list):
        return np.asarray(v, dtype=float)

    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            return np.asarray(json.loads(s), dtype=float)
        return np.asarray([float(x) for x in s.split(",")], dtype=float)

    if isinstance(v, (bytes, bytearray)):
        try:
            arr = np.frombuffer(v, dtype=np.float32)
            return arr.astype(float)
        except:
            raise ValueError("Failed to decode BLOB embedding")

    raise ValueError(f"Unexpected embedding type: {type(v)}")


def parse_embedding_series(series):
    return np.vstack([parse_embedding(v) for v in series])


# ============================
# MAIN
# ============================

def main():


    print("\n===== CATEGORY MODEL (DOMAIN = 1) =====")

    df_train = load_table(DB_PATH, TABLE_TRAIN)
    df_test = load_table(DB_PATH, TABLE_TEST)

    # Filter to domain=1
    df_train = df_train[df_train[DOMAIN_COL] == 2]
    df_test = df_test[df_test[DOMAIN_COL] == 2]

    print(f"Train rows after filtering domain=1 → {len(df_train)}")
    print(f"Test rows after filtering domain=1 → {len(df_test)}")

    # Parse embeddings
    X_train = parse_embedding_series(df_train[EMBED_COL])
    X_test = parse_embedding_series(df_test[EMBED_COL])

    # IMPORTANT: keep labels 5 and 7 exactly as they are
    y_train = df_train[CATEGORY_COL].astype(int).values
    y_test = df_test[CATEGORY_COL].astype(int).values

    print(f"Labels used: {np.unique(y_train).tolist()} (should be [2 , 3])")

    # ============================
    # TRAIN MODELS
    # ============================

    # Logistic Regression
    print("Training LR...")
    lr = LogisticRegression(max_iter=5000, class_weight="balanced")
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(MODEL_DIR, "lr_category_domain2.pkl"))

    # Random Forest
    print("Training RF...")
    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_category_domain2.pkl"))

    # ------------------------------------------
    # XGBoost requires labels 0..n
    # ------------------------------------------

    # Map original labels -> temp 0/1
    label_to_temp = {2: 0, 3: 1}
    temp_to_label = {0: 2, 1: 3}

    y_train_temp = np.array([label_to_temp[v] for v in y_train])
    y_test_temp = np.array([label_to_temp[v] for v in y_test])

    print("Training XGB... (internally using 0 and 1)")

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=2,
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
    xgb.save_model(os.path.join(MODEL_DIR, "xgb_category_domain2.json"))

    # ============================
    # EVALUATION
    # ============================

    report_path = os.path.join(MODEL_DIR, "report_category_domain1.txt")
    with open(report_path, "w", encoding="utf-8") as f:

        f.write("=== CATEGORY MODEL (DOMAIN 1) ===\n\n")

        for name, model in [
            ("Logistic Regression", lr),
            ("Random Forest", rf),
            ("XGBoost", xgb)
        ]:
            f.write(f"\n---- {name} ----\n")

            # XGBoost → probability matrix → convert to 5/7
            if name == "XGBoost":
                preds_temp = model.predict(X_test)  # shape (N,2)
                preds_temp = np.argmax(preds_temp, axis=1)  # 0/1
                preds = np.array([temp_to_label[int(v)] for v in preds_temp])
            else:
                preds = model.predict(X_test)  # already 5/7

            f.write("\nClassification Report:\n")
            f.write(classification_report(y_test, preds))

            cm = confusion_matrix(y_test, preds)
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\n")

    print("==========================================")
    print("✔ CATEGORY MODEL FOR DOMAIN=1 TRAINED")
    print("✔ MODELS SAVED IN:", MODEL_DIR)
    print("✔ REPORT SAVED:", report_path)
    print("==========================================\n")


if __name__ == "__main__":
    main()
