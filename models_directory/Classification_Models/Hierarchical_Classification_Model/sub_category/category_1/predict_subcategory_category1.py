#!/usr/bin/env python3
"""
predict_subcategory_cat1.py
Predicts subcategory for Category 1 using trained LR, RF, XGB vocab_models.
No argparse. No external label_map.json.
"""

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

lr = joblib.load(os.path.join(MODEL_DIR, "lr_subcat_cat1.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "rf_subcat_cat1.pkl"))

xgb = XGBClassifier()
xgb.load_model(os.path.join(MODEL_DIR, "xgb_subcat_cat1.json"))

# ============================================================
# LABEL MAP (HARDCODED)
# 🔥 Replace these numbers with the real labels for category_1:
# Example based on your note: 1 → {2,10,21,24}
# ============================================================

temp_to_label = {
    0: 2,
    1: 10,
    2: 21,
    3: 24
}

label_to_temp = {
    2: 0,
    10: 1,
    21: 2,
    24: 3
}


# ============================================================
# PREDICT FROM EMBEDDING
# ============================================================

def predict_from_embedding(emb: np.ndarray):
    x = emb.reshape(1, -1)

    # LR + RF already return real labels
    lr_pred = int(lr.predict(x)[0])
    rf_pred = int(rf.predict(x)[0])

    # XGB may return probabilities or direct class index
    raw_xgb = xgb.predict(x)

    if getattr(raw_xgb, "ndim", 1) == 2:
        xgb_temp = int(np.argmax(raw_xgb, axis=1)[0])
    else:
        xgb_temp = int(raw_xgb[0])

    xgb_pred = temp_to_label[xgb_temp]

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
# TEST EXAMPLE (RUN DIRECTLY)
# ============================================================

if __name__ == "__main__":
    example_text = "The nurse did not respond quickly and the communication was unclear."
    embedding = get_embedding(example_text)
    prediction_text = predict_from_text(example_text)

    print(f"The input text is {example_text}")
    print(f"Prediction from Text")
    print(json.dumps(prediction_text, indent=4))
    print(f"Predition from Embeddings")

