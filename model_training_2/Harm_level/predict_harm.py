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


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BINARY_MODEL_PATH = os.path.join(BASE_DIR, "Harm_BinaryModel.pkl")
LOW_MODEL_PATH    = os.path.join(BASE_DIR, "Harm_OrdinalLowModel.pkl")
HIGH_MODEL_PATH   = os.path.join(BASE_DIR, "Harm_OrdinalHighModel.pkl")

MPNET_MODEL_PATH = (
    r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback"
    r"\model_training_2\model_storage\mpnet_embeddings"
)


# ============================================================
# LOAD MODELS
# ============================================================

binary_model = joblib.load(BINARY_MODEL_PATH)
low_model    = joblib.load(LOW_MODEL_PATH)
high_model   = joblib.load(HIGH_MODEL_PATH)


# ============================================================
# LOAD MPNet FOR EMBEDDINGS
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MPNET_MODEL_PATH, local_files_only=True)
mpnet_model = AutoModel.from_pretrained(MPNET_MODEL_PATH, local_files_only=True)
mpnet_model.eval()


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def embed_mpnet(text: str) -> np.ndarray:
    """
    Convert text → 768D embedding (mean pooled)
    """
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
        outputs = mpnet_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return emb.astype(np.float32)


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


def predict_harm(text: str) -> dict:
    emb = embed_mpnet(text)
    return predict_harm_from_embedding(emb)


# ============================================================
# TEST EXAMPLE
# ============================================================

if __name__ == "__main__":
    example_text = "The patient fell and required stitches."

    print("Input Text:")
    print(example_text)

    # ---------------------------------------------------------
    # Generate embedding + debug
    # ---------------------------------------------------------
    emb = embed_mpnet(example_text)

    print("\n====== EMBEDDING DEBUG ======")
    print("TYPE:", type(emb))
    print("FIRST ELEMENT TYPE:", type(emb[0]) if hasattr(emb, "__getitem__") else None)
    print("SHAPE:", getattr(emb, "shape", None))

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------
    result = predict_harm_from_embedding(emb)

    print("\nPredicted Harm Result:")
    print(json.dumps(result, indent=4))
