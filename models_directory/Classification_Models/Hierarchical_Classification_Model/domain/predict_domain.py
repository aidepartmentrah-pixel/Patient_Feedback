#!/usr/bin/env python3
"""
predict_domain.py

Predicts domain using pre-trained Logistic Regression, Random Forest,
and XGBoost models trained on MPNet sentence embeddings.

Loads the MPNet model once, converts raw text to embeddings, and
predicts final GLOBAL labels using label_map_domain.json.
"""

import os
import json
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
from xgboost import XGBClassifier
from pathlib import Path
from models_directory.Classification_Models.Stage.modular_functions import get_embedding

# ============================
# PATHS
# ============================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "vocab_models"
LABEL_MAP_FILE = BASE_DIR / "label_map_domain.json"

# ============================
# LOAD LABEL MAP
# ============================

with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
    label_map = json.load(f)

global_labels = label_map["global_labels"]
local_labels = label_map["local_labels"]
local_to_global = {local: global_ for local, global_ in zip(local_labels, global_labels)}

# ============================
# LOAD MODELS
# ============================

lr = joblib.load(MODEL_DIR / "lr_domain.pkl")
rf = joblib.load(MODEL_DIR / "rf_domain.pkl")

xgb = XGBClassifier()
xgb.load_model(MODEL_DIR / "xgb_domain.json")

# ============================
# PREDICTION FUNCTIONS
# ============================

def predict_domain_from_embedding(embedding: np.ndarray):
    """
    Predict using Logistic Regression, Random Forest & XGBoost.
    Returns GLOBAL labels.
    """
    x = embedding.reshape(1, -1)

    # LR
    lr_local = int(lr.predict(x)[0])
    lr_global = local_to_global[lr_local]

    # RF
    rf_local = int(rf.predict(x)[0])
    rf_global = local_to_global[rf_local]

    # XGB
    xgb_raw = xgb.predict(x)
    if getattr(xgb_raw, "ndim", 1) == 2:
        xgb_local = int(np.argmax(xgb_raw, axis=1)[0])
    else:
        xgb_local = int(xgb_raw[0])
    xgb_global = local_to_global[xgb_local]

    return {
        "logistic_regression": lr_global,
        "random_forest": rf_global,
        "xgboost": xgb_global
    }

def predict_from_text(text: str):
    # Safe embedding load
    emb = get_embedding(text, Troubleshoot=False)
    return predict_domain_from_embedding(np.frombuffer(emb, dtype=np.float32))

# ============================
# TEST
# ============================

if __name__ == "__main__":

    example_text = "The nurse did not respond quickly."

    print("\n==============================")
    print("🔍 INPUT TEXT")
    print("==============================")
    print(example_text)

    print("\nRunning predictions ...")
    preds = predict_from_text(example_text)

    print("\n==============================")
    print("📊 PREDICTIONS")
    print("==============================")
    print(json.dumps(preds, indent=4))
