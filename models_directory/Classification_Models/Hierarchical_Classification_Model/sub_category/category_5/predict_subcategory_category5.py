#!/usr/bin/env python3
"""
predict_subcategory_cat5.py
Predicts subcategory for Category 5 using trained LR, RF, XGB vocab_models.
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
# LOAD TRAINED MODELS FOR CATEGORY 5
# ============================================================

lr = joblib.load(os.path.join(MODEL_DIR, "lr_subcat_cat5.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "rf_subcat_cat5.pkl"))

xgb = XGBClassifier()
xgb.load_model(os.path.join(MODEL_DIR, "xgb_subcat_cat5.json"))

# ============================================================
# LABEL MAP (HARDCODED FOR CATEGORY 5)
# Example: Category 5 labels = [1, 19, 26]
# ============================================================

temp_to_label = {
    0: 1,
    1: 19,
    2: 26
}

label_to_temp = {
    1: 0,
    19: 1,
    26: 2
}

# ============================================================
# PREDICT FROM EMBEDDING
# ============================================================

def predict_from_embedding(emb: np.ndarray):
    x = emb.reshape(1, -1)

    # LR + RF already return real labels directly
    lr_pred = int(lr.predict(x)[0])
    rf_pred = int(rf.predict(x)[0])

    # XGB may need postprocessing
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
    example_text = "Patient expected the hospital to handle their complaint quickly."

    pred_text = predict_from_text(example_text)

    print(f"Input text: {example_text}")
    print("Prediction from text:")
    print(json.dumps(pred_text, indent=4))
