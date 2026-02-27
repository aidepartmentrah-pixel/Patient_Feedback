"""
predict_severity_level.py

Loads the trained Random Forest model for "severity_level"
and predicts the class for new input text using Arabert embeddings.

Usage example:
    python predict_severity_level.py "المريض غير راضٍ عن الخدمة المقدمة في المستشفى"
"""

import sys
import joblib
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import numpy as np

# ---------------- CONFIG ----------------
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "severity_level_rf.pkl"   # Random Forest model filename
EMBED_MODEL_NAME = "aubmindlab/bert-base-arabertv2"

# -------------- Embedder -----------------
def generate_embedding(text: str) -> np.ndarray:
    """Generate BERT embedding for a given Arabic text using Arabert."""
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embedding

# -------------- Predictor -----------------
def predict_severity_level(text: str):
    """Load model and predict severity level for input text."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print("Random Forest model loaded successfully.")

    # Generate embedding
    print("Generating embedding from text...")
    embedding = generate_embedding(text).reshape(1, -1)

    # Predict
    prediction = model.predict(embedding)[0]
    print(f"\nPredicted severity level: {prediction}")

    return prediction

# -------------- Main flow -----------------
if __name__ == "__main__":
   print(predict_severity_level("المريض غير راضٍ عن الخدمة المقدمة في المستشفى"))
