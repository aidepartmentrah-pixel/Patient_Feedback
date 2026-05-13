import sqlite3
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mord import LogisticAT
import joblib
import os

# ---------------- CONFIG ----------------
HERE = Path(__file__).resolve().parent
DB_PATH = HERE.parent.parent / "models_directory" / "patient_feedback_ml.db"

TRAIN_TABLE = "table_feedback_train"     # <-- correct
TEST_TABLE = "table_feedback_test"       # <-- correct

EMBED_COL = "embedding_text123"
TARGET_COL = "severity_level"

MODEL_OUT = HERE / "severity_ordinal.pkl"
REPORT_OUT = HERE / "severity_eval.txt"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM feedback_data_clean", conn)
    conn.close()
    return df


def prepare_data(df):
    # keep only rows where embeddings + severity exist
    df = df.dropna(subset=[EMB_COL, TARGET_COL])

    # convert embeddings into numpy arrays
    X = df[EMB_COL].apply(lambda x: np.array(eval(x))).tolist()

    # convert target to int
    y = df[TARGET_COL].astype(int)

    return np.array(X), y


def train_logistic(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    report = classification_report(y_test, preds)
    with open("severity_eval_logistic.txt", "w", encoding="utf-8") as f:
        f.write(report)

    joblib.dump(model, "severity_logistic.pkl")
    print("✔ Logistic Regression done.")
    return preds


def train_ordinal(X_train, y_train, X_test, y_test):
    model = OrdinalLogistic()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    report = classification_report(y_test, preds)
    with open("severity_eval_ordinal.txt", "w", encoding="utf-8") as f:
        f.write(report)

    joblib.dump(model, "severity_ordinal.pkl")
    print("✔ Ordinal Logistic Regression done.")
    return preds


def save_conf_matrix(y_test, preds, name="confusion_matrix.png"):
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Severity Level – Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def main():
    print("Loading data…")
    df = load_data()

    print("Preparing data…")
    X, y = prepare_data(df)

    print(f"Total usable rows: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\nTraining logistic model…")
    preds_log = train_logistic(X_train, y_train, X_test, y_test)
    save_conf_matrix(y_test, preds_log, "conf_matrix_logistic.png")

    print("\nTraining ordinal model…")
    preds_ord = train_ordinal(X_train, y_train, X_test, y_test)
    save_conf_matrix(y_test, preds_ord, "conf_matrix_ordinal.png")

    print("\n✔ DONE: All severity vocab_models trained.")
    print("Reports saved:")
    print("  severity_eval_logistic.txt")
    print("  severity_eval_ordinal.txt")
    print("  conf_matrix_logistic.png")
    print("  conf_matrix_ordinal.png")


if __name__ == "__main__":
    main()
