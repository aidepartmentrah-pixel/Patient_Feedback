#!/usr/bin/env python3
"""
predict_subcategory_category3.py
Predicts subcategory for Category 3 using trained LR, RF, XGB vocab_models.
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
# LOAD OFFLINE MPNet MODEL
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MPNET_MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(MPNET_MODEL_PATH, local_files_only=True)
model.eval()

# ============================================================
# LOAD TRAINED MODELS
# ============================================================

lr = joblib.load(os.path.join(MODEL_DIR, "lr_subcat_cat3.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "rf_subcat_cat3.pkl"))

xgb = XGBClassifier()
xgb.load_model(os.path.join(MODEL_DIR, "xgb_subcat_cat3.json"))

# ============================================================
# LABEL MAP (CATEGORY 3)
# Real labels: 4, 6, 8, 9, 13, 30
# ============================================================

temp_to_label = {
    0: 4,
    1: 6,
    2: 8,
    3: 9,
    4: 13,
    5: 30
}

label_to_temp = {
    4: 0,
    6: 1,
    8: 2,
    9: 3,
    13: 4,
    30: 5
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

    # LR + RF already output real labels
    lr_pred = int(lr.predict(x)[0])
    rf_pred = int(rf.predict(x)[0])

    # XGB may output index or probability
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
# TEST EXAMPLE
# ============================================================

if __name__ == "__main__":
    example_text = "The nurse did not respond quickly and the communication was unclear."
    embedding = embed_mpnet(example_text)
    prediction_text = predict_from_text(example_text)
    prediction_embedding = predict_from_embedding(embedding)

    print(f"The input text is {example_text}")
    print(f"Prediction from Text")
    print(json.dumps(prediction_text, indent=4))
    print(f"Prediction from Embeddings")
    print(json.dumps(prediction_embedding, indent=4))
