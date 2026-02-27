import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
from xgboost import XGBClassifier
from models_directory.Classification_Models.Stage.modular_functions import get_embedding


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "vocab_models")

# ============================================================
# LOAD TRAINED MODELS
# ============================================================

lr = joblib.load(os.path.join(MODEL_DIR, "lr_category_domain1.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "rf_category_domain1.pkl"))

xgb = XGBClassifier()
xgb.load_model(os.path.join(MODEL_DIR, "xgb_category_domain1.json"))

# ============================================================
# LABEL MAP (XGB)
# ============================================================

temp_to_label = {0: 5, 1: 7}     # internal → real
label_to_temp = {5: 0, 7: 1}     # real → internal

# ============================================================
# EMBEDDING FUNCTION (Matches training pipeline exactly)
# ============================================================


# ============================================================
# PREDICT FROM EMBEDDING
# ============================================================

def predict_from_embedding(emb: np.ndarray):
    x = emb.reshape(1, -1)

    # LR + RF already trained on real labels
    lr_pred = int(lr.predict(x)[0])
    rf_pred = int(rf.predict(x)[0])

    # XGB: may return raw labels OR probabilities
    raw_xgb = xgb.predict(x)

    if raw_xgb.ndim == 2:         # probability output
        xgb_pred_temp = int(np.argmax(raw_xgb, axis=1)[0])
    else:                         # raw class index
        xgb_pred_temp = int(raw_xgb[0])

    xgb_pred = temp_to_label[xgb_pred_temp]

    return {
        "logistic_regression": lr_pred,
        "random_forest": rf_pred,
        "xgboost": xgb_pred
    }

# ============================================================
# PREDICT FROM RAW TEXT
# ============================================================

def predict_from_text(text: str):
    emb = get_embedding(text)
    return predict_from_embedding(np.frombuffer(emb, dtype=np.float32))

# ============================================================
# TEST EXAMPLE
# ============================================================

if __name__ == "__main__":
    example_text = "The nurse did not respond quickly."

    print("\n==============================")
    print("INPUT TEXT")
    print("==============================")
    print(example_text)

    print("\nRunning predictions ...")
    preds = predict_from_text(example_text)

    print("\n==============================")
    print("PREDICTIONS")
    print("==============================")
    print(json.dumps(preds, indent=4))
