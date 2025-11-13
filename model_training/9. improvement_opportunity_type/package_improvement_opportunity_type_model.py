"""
package_improvement_opportunity_type_model.py

Utility for inference using the trained Logistic Regression "improvement_opportunity_type" model.

Usage (standalone test):
    python package_improvement_opportunity_type_model.py
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
MODEL_PATH = HERE / "improvement_opportunity_type_logreg.pkl"  # trained model
BERT_NAME = "aubmindlab/bert-base-arabertv2"

print("Loading tokenizer and embedding model...")
tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
embedding_model = AutoModel.from_pretrained(BERT_NAME)
embedding_model.eval()

print("Loading logistic regression model...")
iot_logreg = joblib.load(MODEL_PATH)

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
def predict_improvement_opportunity_type(text: str) -> dict:
    """
    Predicts the 'improvement_opportunity_type' class for given Arabic feedback text.

    Returns:
        dict: {
            "model": "logreg",
            "improvement_opportunity_type_class": int,
            "confidence": float,
        }
    """
    emb = get_embedding(text)
    pred = iot_logreg.predict(emb)[0]
    prob = iot_logreg.predict_proba(emb).max()

    return {
        "model": "logreg",
        "improvement_opportunity_type_class": int(pred),
        "confidence": float(prob),
    }

# =========================================================
# 4. Optional: quick test
# =========================================================
if __name__ == "__main__":
    sample_text = "نقترح تحسين تجربة المرضى في قسم الطوارئ من خلال تقليل وقت الانتظار."
    result = predict_improvement_opportunity_type(sample_text)
    print("\n=== Sample Prediction ===")
    print(result)
