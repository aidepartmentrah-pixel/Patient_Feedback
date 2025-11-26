import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# CONFIG
# ============================================================
HERE = Path(__file__).resolve().parent
DB_PATH = HERE.parent / "patient_feedback_ml.db"

TEST_TABLE = "table_feedback_test"

EMBED_COL = "embedding_text123"
TARGET_COL = "harm_level"

# Models
BINARY_MODEL_PATH = HERE / "Harm_BinaryModel.pkl"
LOW_MODEL_PATH    = HERE / "Harm_OrdinalLowModel.pkl"
HIGH_MODEL_PATH   = HERE / "Harm_OrdinalHighModel.pkl"

OUTPUT_CSV = HERE / "harm_predictions_table.csv"


# ============================================================
# Load Table
# ============================================================
def load_test_data():
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(f"SELECT * FROM {TEST_TABLE}", conn)
    conn.close()
    return df


# ============================================================
# Parse Embedding Column
# ============================================================
def parse_embedding_series(series):
    out = []
    for v in series:
        if isinstance(v, np.ndarray):
            out.append(v.astype(float))
        elif isinstance(v, (list, tuple)):
            out.append(np.asarray(v, float))
        elif isinstance(v, (bytes, bytearray)):
            out.append(np.frombuffer(v, dtype=np.float32).astype(float))
        else:
            out.append(np.asarray(json.loads(v), float))
    return np.vstack(out)


# ============================================================
# Prediction Logic
# ============================================================
def predict_binary(model, emb):
    return int(model.predict(emb.reshape(1, -1))[0])


def predict_low(model, emb):
    pred = int(model.predict(emb.reshape(1, -1))[0])
    return pred + 1     # map back 0→1, 1→2, 2→3


def predict_high(model, emb):
    pred = int(model.predict(emb.reshape(1, -1))[0])
    return pred + 4     # map back 0→4, 1→5, 2→6


# ============================================================
# Main Evaluation
# ============================================================
def evaluate_harm_models():

    print("📥 Loading test data...")
    df = load_test_data()

    # filter valid 1–6
    df = df[df[TARGET_COL].isin([1, 2, 3, 4, 5, 6])]
    df = df.reset_index(drop=True)

    print(f"🔢 Rows after filtering: {len(df)}")

    # Embeddings
    X = parse_embedding_series(df[EMBED_COL])
    y_true = df[TARGET_COL].astype(int).to_numpy()

    # Load Models
    print("📦 Loading vocab_models...")
    model_binary = joblib.load(BINARY_MODEL_PATH)
    model_low    = joblib.load(LOW_MODEL_PATH)
    model_high   = joblib.load(HIGH_MODEL_PATH)

    # Prediction containers
    binary_preds = []
    low_preds    = []
    high_preds   = []
    final_preds  = []

    print("🔮 Running predictions...")

    for i in range(len(df)):
        emb = X[i]
        true = y_true[i]

        # Binary prediction (low=0, high=1)
        pb = predict_binary(model_binary, emb)
        binary_preds.append(pb)

        # Choose sub-model
        if pb == 0:
            ph = predict_low(model_low, emb)
        else:
            ph = predict_high(model_high, emb)

        final_preds.append(ph)

        # Also store sub-model outputs
        if true in [1, 2, 3]:
            low_preds.append(predict_low(model_low, emb))
        else:
            low_preds.append(None)

        if true in [4, 5, 6]:
            high_preds.append(predict_high(model_high, emb))
        else:
            high_preds.append(None)

    # Save results
    df["binary_pred"] = binary_preds
    df["harm_pred"] = final_preds
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"📁 Results saved to {OUTPUT_CSV}")

    # =======================================================
    # METRICS
    # =======================================================
    print("\n===========================")
    print("📊 Final Harm Prediction Metrics")
    print("===========================")
    print(classification_report(y_true, final_preds, zero_division=0))
    print("Accuracy:", accuracy_score(y_true, final_preds))

    print("\n===========================")
    print("📊 LOW Model Metrics (1–3)")
    print("===========================")
    mask_low = df[TARGET_COL].isin([1, 2, 3])
    print(classification_report(df[TARGET_COL][mask_low],
                                df["harm_pred"][mask_low],
                                zero_division=0))

    print("\n===========================")
    print("📊 HIGH Model Metrics (4–6)")
    print("===========================")
    mask_high = df[TARGET_COL].isin([4, 5, 6])
    print(classification_report(df[TARGET_COL][mask_high],
                                df["harm_pred"][mask_high],
                                zero_division=0))


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    evaluate_harm_models()
