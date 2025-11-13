"""
package_harm_level_model.py

Utility for inference using the trained Logistic Regression "harm_level" model.

Usage (standalone test):
    python package_harm_level_model.py
"""

import torch
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# =========================================================
# 1. Model & tokenizer setup
# =========================================================
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "harm_level_logreg.pkl"  # trained model saved by training script
BERT_NAME = "aubmindlab/bert-base-arabertv2"

print("Loading tokenizer and embedding model...")
tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
embedding_model = AutoModel.from_pretrained(BERT_NAME)
embedding_model.eval()

print("Loading logistic regression model...")
harm_level_logreg = joblib.load(MODEL_PATH)

# =========================================================
# 2. Embedding function
# =========================================================
def get_embedding(text: str) -> np.ndarray:
    """
    Convert Arabic text into a 768-dimensional embedding using AraBERT.
    Uses mean pooling over token embeddings.
    """
    if not text or not text.strip():
        raise ValueError("Empty text string received for embedding.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = embedding_model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.reshape(1, -1)

# =========================================================
# 3. Prediction function
# =========================================================
def predict_harm_level(text: str) -> dict:
    """
    Predicts the 'harm_level' class for given Arabic feedback text.

    Returns:
        dict: {
            "model": "logreg",
            "harm_level_class": int,
            "confidence": float,
        }
    """
    emb = get_embedding(text)
    pred = harm_level_logreg.predict(emb)[0]
    prob = harm_level_logreg.predict_proba(emb).max()

    return {
        "model": "logreg",
        "harm_level_class": int(pred),
        "confidence": float(prob),
    }

# =========================================================
# 4. Example test
# =========================================================
if __name__ == "__main__":
    sample_text = "تسبب الخطأ في تأخير تقديم العلاج للمريض."
    result = predict_harm_level(sample_text)
    print("\n=== Sample Prediction ===")
    print(result)
