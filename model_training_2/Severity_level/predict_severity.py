
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

MODEL_PATH = os.path.join(BASE_DIR, "Severity_OrdinalModel.pkl")

MPNET_MODEL_PATH = (
    r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback"
    r"\model_training_2\model_storage\mpnet_embeddings"
)


# ============================================================
# LOAD ML MODEL
# ============================================================

severity_model = joblib.load(MODEL_PATH)


# ============================================================
# LOAD OFFLINE MPNet MODEL
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MPNET_MODEL_PATH, local_files_only=True)
mpnet_model = AutoModel.from_pretrained(MPNET_MODEL_PATH, local_files_only=True)
mpnet_model.eval()


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def embed_mpnet(text: str) -> np.ndarray:
    """
    Convert text → 768-dim MPNet mean-pooled embedding
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


# ============================================================
# PREDICT FROM EMBEDDING
# ============================================================

def predict_severity_from_embedding(emb: np.ndarray) -> int:
    """
    emb (768D vector) → model → severity level (1–4)
    """
    x = emb.reshape(1, -1)

    pred_ordinal = int(severity_model.predict(x)[0])   # 0–3
    severity_label = pred_ordinal + 1                  # back to 1–4

    return severity_label


# ============================================================
# PREDICT FROM RAW TEXT
# ============================================================

def predict_severity(text: str) -> int:
    emb = embed_mpnet(text)
    return predict_severity_from_embedding(emb)


# ============================================================
# TEST EXAMPLE
# ============================================================

if __name__ == "__main__":
    example_text = "The patient had died."

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
    severity = predict_severity_from_embedding(emb)

    print("\nPredicted Severity Level (1–4):", severity)