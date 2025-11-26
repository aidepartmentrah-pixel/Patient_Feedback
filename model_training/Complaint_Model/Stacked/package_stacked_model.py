"""
package_stacked_model.py

Loads the trained stacked Logistic Regression vocab_models for:
- domain
- category
- sub_category
- classification_ar

Given an Arabic text input, this script:
1. Generates its BERT embedding using aubmindlab/bert-base-arabertv2
2. Passes the embedding through the 4 stacked classifiers
3. Returns predicted labels and confidence scores

Usage:
    python package_stacked_model.py
"""

import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# =========================================================
# 1. Load tokenizer and embedding model
# =========================================================
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
print("🔹 Loading Arabic BERT tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)
bert_model.eval()
print("✅ BERT model loaded successfully.\n")

# =========================================================
# 2. Load trained vocab_models
# =========================================================
HERE = Path(__file__).resolve().parent

MODEL_PATHS = {
    "domain": HERE / "domain_stacked_logreg.pkl",
    "category": HERE / "category_stacked_logreg.pkl",
    "sub_category": HERE / "sub_category_stacked.pkl",
    "classification_ar": HERE / "classification_ar_stacked_logreg.pkl",
}

MODELS = {}
for name, path in MODEL_PATHS.items():
    if not path.exists():
        print(f"⚠️ Warning: Model file not found for '{name}' -> {path.name}")
    else:
        MODELS[name] = joblib.load(path)
        print(f"✅ Loaded model: {path.name}")
print()

# =========================================================
# 3. Helper: Get BERT embedding
# =========================================================
def get_embedding(text: str) -> np.ndarray:
    """Convert text into mean-pooled 768-dim Arabic BERT embedding."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.reshape(1, -1)

# =========================================================
# 4. Prediction function
# =========================================================
def predict_stacked(text: str):
    """
    Predict domain, category, sub_category, and classification_ar
    using stacked logistic regression vocab_models.
    """
    if not text.strip():
        raise ValueError("Input text is empty.")

    emb = get_embedding(text)
    results = {}

    for name, model in MODELS.items():
        try:
            pred = model.predict(emb)[0]
            prob = model.predict_proba(emb).max() if hasattr(model, "predict_proba") else None
            results[name] = {
                "label": int(pred),
                "confidence": float(prob) if prob is not None else None
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results

# =========================================================
# 5. Example usage
# =========================================================
if __name__ == "__main__":
    example_text = "الطبيب كان متعاونًا جدًا ولكن المواعيد كانت متأخرة."
    print("🧠 Example text:", example_text)
    output = predict_stacked(example_text)
    print("\n🔍 Predictions:")
    for key, val in output.items():
        print(f" - {key}: {val}")
