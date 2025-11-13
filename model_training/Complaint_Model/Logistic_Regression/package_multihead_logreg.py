"""
package_multihead_logreg.py

Utility for inference using trained multi-head Logistic Regression models
for: domain, category, sub_category, classification_ar

Usage:
    python package_multihead_logreg.py
"""

import torch
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# ---------------- CONFIG ----------------
HERE = Path(__file__).resolve().parent
BERT_NAME = "aubmindlab/bert-base-arabertv2"

MODEL_PATHS = {
    "domain": HERE / "domain_logreg.pkl",
    "category": HERE / "category_logreg.pkl",
    "sub_category": HERE / "sub_category_logreg.pkl",
    "classification_ar": HERE / "classification_ar_logreg.pkl",
}

print("Loading AraBERT tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
embedding_model = AutoModel.from_pretrained(BERT_NAME)
embedding_model.eval()

# Load all Logistic Regression heads
print("Loading trained models...")
models = {}
for name, path in MODEL_PATHS.items():
    if path.exists():
        models[name] = joblib.load(path)
        print(f"✅ Loaded {name} from {path.name}")
    else:
        print(f"⚠️ Model not found: {path.name} (skipping)")


# ---------------- Embedding ----------------
def get_embedding(text: str) -> np.ndarray:
    """Convert Arabic text to mean-pooled 768-dim embedding."""
    if not text or not text.strip():
        raise ValueError("Empty text string received.")

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return emb.reshape(1, -1)


# ---------------- Prediction ----------------
def predict_feedback(text: str) -> dict:
    """Predict all 4 classification heads from a single Arabic text."""
    emb = get_embedding(text)
    result = {}
    for name, model in models.items():
        pred = model.predict(emb)[0]
        prob = model.predict_proba(emb).max()
        result[name] = {"class": int(pred), "confidence": float(prob)}
    return result


# ---------------- Example Test ----------------
if __name__ == "__main__":
    example_text = "تجربتي في المستشفى كانت ممتازة من ناحية التعامل ولكن الانتظار كان طويلاً"
    prediction = predict_feedback(example_text)
    print("\n=== Multi-Head Prediction ===")
    for k, v in prediction.items():
        print(f"{k}: class={v['class']} (conf={v['confidence']:.3f})")
