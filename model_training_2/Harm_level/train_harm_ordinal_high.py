# train_harm_ordinal_high.py
"""
Ordinal Harm Model (HIGH PART)
Predicts harm levels: 4, 5, 6 → mapped to ordinal 0,1,2

Produces:
- Harm_OrdinalHighModel.pkl
- harm_high_report.txt
"""

import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import traceback
import mord
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------- CONFIG ----------------
HERE = Path(__file__).resolve().parent
DB_PATH = HERE.parent / "patient_feedback_ml.db"

TRAIN_TABLE = "table_feedback_train"
TEST_TABLE  = "table_feedback_test"

EMBED_COL  = "embedding_text123"
TARGET_COL = "harm_level"

MODEL_PATH  = HERE / "Harm_OrdinalHighModel.pkl"
REPORT_PATH = HERE / "harm_high_report.txt"

# ------------ load --------------
def load_table(db_path, table):
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

# ------------ parse -------------
def parse_embedding_series(series):
    out=[]
    for v in series:
        if isinstance(v,np.ndarray): out.append(v.astype(float))
        elif isinstance(v,(list,tuple)): out.append(np.asarray(v,float))
        elif isinstance(v,(bytes,bytearray)): out.append(np.frombuffer(v, np.float32).astype(float))
        else: out.append(np.asarray(json.loads(v),float))
    return np.vstack(out)

# ------------ main --------------
def main():
    try:
        df_train = load_table(DB_PATH, TRAIN_TABLE)
        df_test  = load_table(DB_PATH, TEST_TABLE)

        df_train = df_train[df_train[TARGET_COL].isin([4,5,6])]
        df_test  = df_test[df_test[TARGET_COL].isin([4,5,6])]

        X_train = parse_embedding_series(df_train[EMBED_COL])
        X_test  = parse_embedding_series(df_test[EMBED_COL])

        y_train = df_train[TARGET_COL].astype(int).to_numpy() - 4  # 4→0,5→1,6→2
        y_test  = df_test[TARGET_COL].astype(int).to_numpy() - 4

        model = mord.LogisticIT()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro")

        report = classification_report(y_test, y_pred, zero_division=0)

        joblib.dump(model, MODEL_PATH)

        with open(REPORT_PATH, "w") as f:
            f.write("HIGH Harm Ordinal Model\n")
            f.write(f"Accuracy: {acc}\n")
            f.write(f"F1 Macro: {f1}\n\n")
            f.write(report)

        print("High harm ordinal model complete.")

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
