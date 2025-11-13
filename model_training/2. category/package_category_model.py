"""
package_category_model.py

Loads the trained Logistic Regression model for category classification
and provides a function to predict from raw text.

Usage example:
    from package_category_model import predict_category
    result = predict_category("النظافة كانت سيئة في القسم")
    print(result)
"""

import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# ---------------- CONFIG ----------------
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "category_logreg.pkl"

# Load AraBERT model
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)
bert_model.eval()

# Load the trained classifier
category_model = joblib.load(MODEL_PATH)


# ---------------- Functions ----------------
def get_embedding(text: str) -> np.ndarray:
    """Convert Arabic text into 768-dim mean-pooled AraBERT embedding."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.reshape(1, -1)


def predict_category(text: str):
    """Predict category class using the trained Logistic Regression model."""
    emb = get_embedding(text)
    pred = category_model.predict(emb)[0]
    prob = category_model.predict_proba(emb).max()
    return {"model": "logreg", "category_class": int(pred), "confidence": float(prob)}


# ---------------- Test ----------------
if __name__ == "__main__":
    example = "الخدمة في المستشفى كانت بطيئة جداً"
    print(predict_category(example))
