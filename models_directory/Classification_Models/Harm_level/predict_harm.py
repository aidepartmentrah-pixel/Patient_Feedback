#!/usr/bin/env python3
"""
predict_harm.py

Predict harm level from text using:

1) Text → MPNet embedding
2) Binary model → LOW or HIGH
3) If LOW  → ordinal_low model (1–3)
4) If HIGH → ordinal_high model (4–6)

Also supports:
 - Predict directly from embedding
 - Debug printing
"""

import os
import json
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
from models_directory.Classification_Models.Stage.modular_functions import get_embedding

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BINARY_MODEL_PATH = os.path.join(BASE_DIR, "Harm_BinaryModel.pkl")
LOW_MODEL_PATH    = os.path.join(BASE_DIR, "Harm_OrdinalLowModel.pkl")
HIGH_MODEL_PATH   = os.path.join(BASE_DIR, "Harm_OrdinalHighModel.pkl")

MPNET_MODEL_PATH = (
    r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback"
    r"\models_directory\Classification_Models\model_storage\mpnet_embeddings"
)


# ============================================================
# LOAD MODELS
# ============================================================

binary_model = joblib.load(BINARY_MODEL_PATH)
low_model    = joblib.load(LOW_MODEL_PATH)
high_model   = joblib.load(HIGH_MODEL_PATH)




def predict_harm_from_embedding(emb: np.ndarray) -> dict:
    """
    emb (768D numpy vector) → harm classification
    """
    x = emb.reshape(1, -1)

    # 1) Binary: 0 → LOW, 1 → HIGH
    binary_pred = int(binary_model.predict(x)[0])

    if binary_pred == 0:
        raw = int(low_model.predict(x)[0])     # 0,1,2
        harm = raw + 1                         # → 1–3
        group = "LOW (1–3)"
    else:
        raw = int(high_model.predict(x)[0])    # 0,1,2
        harm = raw + 4                         # → 4–6
        group = "HIGH (4–6)"

    return {
        "low_or_high_group": group,
        "harm_level": harm
    }

def predict_harm_from_text(text: str) -> dict:
    emb = get_embedding(text)
    return predict_harm_from_embedding(np.frombuffer(emb, dtype=np.float32))


# ============================================================
# TEST EXAMPLE
# ============================================================

if __name__ == "__main__":
    example_text = "The patient fell and required stitches."
    raw = get_embedding(example_text)
    emd = np.frombuffer(raw, dtype=np.float32)

    print("Input Text:")
    print(example_text)
    result_embedd = predict_harm_from_embedding(emd)
    result_text = predict_harm_from_text(example_text)

    print("\nPredicted Harm Result:")
    print(json.dumps(result_embedd, indent=4))
    print(json.dumps(result_text, indent=4))
