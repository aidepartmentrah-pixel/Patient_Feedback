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
# LOAD TRAINED MODELS FOR CATEGORY 7
# ============================================================

lr = joblib.load(os.path.join(MODEL_DIR, "lr_subcat_cat7.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "rf_subcat_cat7.pkl"))

xgb = XGBClassifier()
xgb.load_model(os.path.join(MODEL_DIR, "xgb_subcat_cat7.json"))

# ============================================================
# LABEL MAP (HARDCODED FOR CATEGORY 7)
# Example labels: [5, 15, 16, 18, 22, 29]
# ============================================================

temp_to_label = {
    0: 5,
    1: 15,
    2: 16,
    3: 18,
    4: 22,
    5: 29
}

label_to_temp = {
    5: 0,
    15: 1,
    16: 2,
    18: 3,
    22: 4,
    29: 5
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
    emb = embed_mpnet(text)
    return predict_from_embedding(emb)

# ============================================================
# TEST EXAMPLE (RUN DIRECTLY)
# ============================================================

if __name__ == "__main__":
    example_text = "The appointment scheduling was smooth and efficient."

    emb = embed_mpnet(example_text)
    pred_text = predict_from_text(example_text)
    pred_emb = predict_from_embedding(emb)

    print("Input Text:")
    print(example_text)

    print("\nPrediction from text:")
    print(json.dumps(pred_text, indent=4))

    print("\nPrediction from embedding:")
    print(json.dumps(pred_emb, indent=4))
