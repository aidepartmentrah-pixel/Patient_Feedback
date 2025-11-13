"""
package_subcategory_model.py

Lightweight loader and predictor for Subcategory model.
Uses logistic regression trained on embedding_text1 (Arabert v2 embeddings).
"""

import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModel

# =========================================================
# 1. Load tokenizer and embedding model
# =========================================================
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# =========================================================
# 2. Load trained model (Logistic Regression)
# =========================================================
subcategory_logreg = joblib.load("subcategory_logreg.pkl")

# =========================================================
# 3. Function to embed raw text
# =========================================================
def get_embedding(text: str) -> np.ndarray:
    """Convert text into 768-dim BERT embedding (mean pooled)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.reshape(1, -1)

# =========================================================
# 4. Prediction function
# =========================================================
def predict_subcategory(text: str):
    """Predict subcategory using Logistic Regression model."""
    emb = get_embedding(text)
    pred = subcategory_logreg.predict(emb)[0]
    prob = subcategory_logreg.predict_proba(emb).max()
    return {"model": "logreg", "subcategory_class": int(pred), "confidence": float(prob)}

# =========================================================
# 5. Example
# =========================================================
if __name__ == "__main__":
    example_text = "الموظفين في الاستقبال لم يتعاونوا معي بالشكل المطلوب"
    print(predict_subcategory(example_text))
