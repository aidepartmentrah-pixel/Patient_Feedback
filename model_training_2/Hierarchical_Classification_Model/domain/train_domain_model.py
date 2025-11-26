import os
import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib
import matplotlib.pyplot as plt


# ============================
# CONFIG
# ============================

DB_PATH = r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\model_training_2\patient_feedback_ml.db"

TABLE_TRAIN = "table_feedback_train"
TABLE_TEST = "table_feedback_test"

EMBED_COL = "embedding_text1"
LABEL_COL = "domain"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "vocab_models")
os.makedirs(MODEL_DIR, exist_ok=True)

LABEL_MAP_FILE = os.path.join(SCRIPT_DIR, "label_map_domain.json")
REPORT_FILE = os.path.join(MODEL_DIR, "domain_metrics.txt")


# ============================
# HELPERS
# ============================

def load_table(conn_path, table):
    conn = sqlite3.connect(conn_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


def parse_embedding(v):
    """Accepts list, JSON string, comma-separated, or BLOB bytes."""
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


def save_confusion_matrix(cm, labels, out_path, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()

    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============================
# MAIN
# ============================

def main():
    print(f"Using database: {DB_PATH}")
    print("Loading train/test tables...")

    df_train = load_table(DB_PATH, TABLE_TRAIN)
    df_test = load_table(DB_PATH, TABLE_TEST)

    # Extract X
    X_train = parse_embedding_series(df_train[EMBED_COL])
    X_test = parse_embedding_series(df_test[EMBED_COL])

    # Extract y
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

    # ============================
    # MANUAL CLASS WEIGHTS
    # ============================

    # domain distribution: {1:90, 2:260, 3:100}
    manual_weights = {
        1: 1.0,
        2: 0.35,
        3: 0.90
    }

    # Convert to encoded label weights
    class_weight_vector = {
        le.transform([cls])[0]: weight
        for cls, weight in manual_weights.items()
    }

    print("Using manual class weights:", class_weight_vector)

    # ============================
    # TRAIN MODELS
    # ============================

    print("Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=5000,
        class_weight=class_weight_vector
    )
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(MODEL_DIR, "lr_domain.pkl"))

    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight=class_weight_vector,
        random_state=42
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_domain.pkl"))

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
    xgb.save_model(os.path.join(MODEL_DIR, "xgb_domain.json"))

    # ============================
    # EVALUATION
    # ============================

    models = {
        "LogisticRegression": lr,
        "RandomForest": rf,
        "XGBoost": xgb
    }

    report_lines = []
    report_lines.append("======== DOMAIN CLASSIFICATION REPORT ========\n")

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        cls_report = classification_report(y_test, y_pred)

        report_lines.append(f"\n===== {name} =====\n")
        report_lines.append(f"Accuracy: {acc:.4f}\n")
        report_lines.append(f"F1 Macro: {f1:.4f}\n")
        report_lines.append("Classification Report:\n")
        report_lines.append(cls_report + "\n")

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join(MODEL_DIR, f"cm_{name.lower()}.png")
        save_confusion_matrix(cm, label_names, cm_path, f"Confusion Matrix - {name}")

    # write metrics file
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n===============================")
    print("✔ DOMAIN MODELS TRAINED WITH MANUAL WEIGHTS")
    print("===============================")
    print(f"Saved vocab_models in : {MODEL_DIR}")
    print(f"Saved report   : {REPORT_FILE}")


if __name__ == "__main__":
    main()
