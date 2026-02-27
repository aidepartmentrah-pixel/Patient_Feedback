#!/usr/bin/env python3
"""
predict_subcategory_cat7.py
Predicts subcategory for Category 7 using trained LR, RF, XGB vocab_models.
No argparse. No external label map.
"""

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
from xgboost import XGBClassifier
from models_directory.Classification_Models.Stage.model_package import get_embedding
from models_directory.Classification_Models.label_mapping_helper import (
    build_temp_to_label_for_category,
    validate_model_mapping,
    log_predictor_init,
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "vocab_models")
CATEGORY_ID = 7
_FILE_NAME = os.path.basename(__file__)

# ============================================================
# LOAD TRAINED MODELS FOR CATEGORY 7
# ============================================================

lr = joblib.load(os.path.join(MODEL_DIR, "lr_subcat_cat7.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "rf_subcat_cat7.pkl"))

xgb = XGBClassifier()
_xgb_model_path = os.path.join(MODEL_DIR, "xgb_subcat_cat7.json")
xgb.load_model(_xgb_model_path)

# ============================================================
# LABEL MAP (DYNAMIC FROM TRAINING DB)
# ============================================================

temp_to_label = build_temp_to_label_for_category(CATEGORY_ID)
label_to_temp = {v: k for k, v in temp_to_label.items()}

# Validate & log at load time
validate_model_mapping(xgb, temp_to_label, _FILE_NAME, _xgb_model_path)
log_predictor_init(_FILE_NAME, _xgb_model_path, xgb.n_classes_, temp_to_label)

# ============================================================
# EMBEDDING FUNCTION
# ============================================================


# ============================================================
# PREDICT FROM EMBEDDING
# ============================================================

def predict_from_embedding(emb: np.ndarray):
    x = emb.reshape(1, -1)

    lr_pred = int(lr.predict(x)[0])
    rf_pred = int(rf.predict(x)[0])

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
    example_text = "The appointment scheduling was smooth and efficient."
    pred_text = predict_from_text(example_text)

    print("Input Text:")
    print(example_text)

    print("\nPrediction from text:")
    print(json.dumps(pred_text, indent=4))

