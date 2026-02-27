import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
from xgboost import XGBClassifier
from models_directory.Classification_Models.Stage.modular_functions import get_embedding
from models_directory.Classification_Models.label_mapping_helper import (
    build_temp_to_label_for_domain,
    validate_model_mapping,
    log_predictor_init,
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "vocab_models")
DOMAIN_ID = 3
_FILE_NAME = os.path.basename(__file__)

# ============================================================
# LOAD TRAINED MODELS
# ============================================================

lr = joblib.load(os.path.join(MODEL_DIR, "lr_category_domain3.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "rf_category_domain3.pkl"))

xgb = XGBClassifier()
_xgb_model_path = os.path.join(MODEL_DIR, "xgb_category_domain3.json")
xgb.load_model(_xgb_model_path)

# ============================================================
# LABEL MAP (DYNAMIC FROM TRAINING DB)
# ============================================================

temp_to_label = build_temp_to_label_for_domain(DOMAIN_ID)
label_to_temp = {v: k for k, v in temp_to_label.items()}

# Validate & log at load time
validate_model_mapping(xgb, temp_to_label, _FILE_NAME, _xgb_model_path)
log_predictor_init(_FILE_NAME, _xgb_model_path, xgb.n_classes_, temp_to_label)


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
    example_text = "The nurse did not take action quickly, and the follow-up was poor."

    print("Input Text:")
    print(example_text)

    preds = predict_from_text(example_text)

    print("\nPredictions (Real Labels):")
    print(json.dumps(preds, indent=4))
