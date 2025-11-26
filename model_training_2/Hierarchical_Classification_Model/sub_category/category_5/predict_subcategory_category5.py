#!/usr/bin/env python3
"""
predict_subcategory_cat5.py
Predicts subcategory for Category 5 using trained LR, RF, XGB vocab_models.
No argparse. No external label_map.json.
"""

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
from xgboost import XGBClassifier

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "vocab_models")

MPNET_MODEL_PATH = (
    r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback"
    r"\model_training_2\model_storage\mpnet_embeddings"
)

# ============================================================
# LOAD OFFLINE MPNet MODEL (ONCE ONLY)
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MPNET_MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(MPNET_MODEL_PATH, local_files_only=True)
model.eval()

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
# EMBEDDING FUNCTION
# ============================================================

def embed_mpnet(text: str) -> np.ndarray:
    if text is None:
        text = ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return emb.astype(np.float32)

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
    emb = embed_mpnet(text)
    return predict_from_embedding(emb)

# ============================================================
# TEST EXAMPLE (RUN DIRECTLY)
# ============================================================

if __name__ == "__main__":
    example_text = "Patient expected the hospital to handle their complaint quickly."

    emb = embed_mpnet(example_text)
    pred_text = predict_from_text(example_text)
    pred_emb = predict_from_embedding(emb)

    print(f"Input text: {example_text}")
    print("Prediction from text:")
    print(json.dumps(pred_text, indent=4))
    print("Prediction from embedding:")
    print(json.dumps(pred_emb, indent=4))
