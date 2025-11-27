import json
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score


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

def compute_metrics(y_true, y_pred, all_labels=None):
    y_true_df = pd.get_dummies(y_true)
    y_pred_df = pd.get_dummies(y_pred)

    if all_labels is not None:
        for lbl in all_labels:
            if lbl not in y_true_df.columns:
                y_true_df[lbl] = 0
            if lbl not in y_pred_df.columns:
                y_pred_df[lbl] = 0
        y_true_df = y_true_df[all_labels]
        y_pred_df = y_pred_df[all_labels]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mAP": average_precision_score(y_true_df, y_pred_df, average="macro")
    }
    return metrics