"""
package_domain_model.py

Utility for inference using the trained Logistic Regression "domain" model.

Usage (standalone test):
    python package_domain_model.py
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
MODEL_PATH = HERE / "domain_logreg.pkl"  # trained model saved by training script
BERT_NAME = "aubmindlab/bert-base-arabertv2"

print("Loading tokenizer and embedding model...")
tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
embedding_model = AutoModel.from_pretrained(BERT_NAME)
embedding_model.eval()

print("Loading logistic regression model...")
domain_logreg = joblib.load(MODEL_PATH)

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

    # mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.reshape(1, -1)  # shape (1, 768)

# =========================================================
# 3. Prediction function
# =========================================================
def predict_domain(text: str) -> dict:
    """
    Predicts the 'domain' class for given Arabic feedback text.

    Returns:
        dict: {
            "model": "logreg",
            "domain_class": int,
            "confidence": float,
        }
    """
    emb = get_embedding(text)
    pred = domain_logreg.predict(emb)[0]
    prob = domain_logreg.predict_proba(emb).max()

    return {
        "model": "logreg",
        "domain_class": int(pred),
        "confidence": float(prob),
    }

# =========================================================
# 4. Optional: quick test
# =========================================================
if __name__ == "__main__":
    sample_text = "تجربتي في المستشفى كانت ممتازة من ناحية التعامل ولكن الانتظار كان طويلاً"
    result = predict_domain(sample_text)
    print("\n=== Sample Prediction ===")
    print(result)
