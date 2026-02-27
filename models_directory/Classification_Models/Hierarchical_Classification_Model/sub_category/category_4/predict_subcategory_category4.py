#!/usr/bin/env python3
"""
predict_subcategory_category4.py
Predicts subcategory for Category 4 using trained LR, RF, XGB vocab_models.
No argparse. No external label_map.json.
"""

import os
import json
import numpy as np
import joblib
from xgboost import XGBClassifier
from models_directory.Classification_Models.Stage.model_package import get_embedding

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "vocab_models")


# ============================================================
# LOAD TRAINED MODELS (CATEGORY 4)
# ============================================================

lr = joblib.load(os.path.join(MODEL_DIR, "lr_subcat_cat4.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "rf_subcat_cat4.pkl"))

xgb = XGBClassifier()
xgb.load_model(os.path.join(MODEL_DIR, "xgb_subcat_cat4.json"))

# ============================================================
# LABEL MAP (CATEGORY 4)
# Real labels: 11, 23
# ============================================================

temp_to_label = {
    0: 11,
    1: 23
}

label_to_temp = {
    11: 0,
    23: 1
}

# ============================================================
# PREDICT FROM EMBEDDING
# ============================================================

def predict_from_embedding(emb: np.ndarray):
    x = emb.reshape(1, -1)

    # LR + RF already output real labels
    lr_pred = int(lr.predict(x)[0])
    rf_pred = int(rf.predict(x)[0])

    # XGB may output index or prob dist
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
# TEST EXAMPLE
# ============================================================

if __name__ == "__main__":
    example_text = "The staff explained the procedure clearly and on time."

    print("Input Text:")
    print(example_text)

    # Predict using only raw text
    prediction_text = predict_from_text(example_text)

    print("\nPrediction from Text:")
    print(json.dumps(prediction_text, indent=4))
