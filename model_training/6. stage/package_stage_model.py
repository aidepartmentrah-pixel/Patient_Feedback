"""
package_stage_model.py

Loads the trained logistic regression model for 'stage'
and predicts the stage label for a given Arabic text input.
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
# 2. Load trained Logistic Regression model
# =========================================================
stage_logreg = joblib.load("stage_logreg.pkl")

# =========================================================
# 3. Helper: get BERT embedding
# =========================================================
def get_embedding(text: str) -> np.ndarray:
    """Convert Arabic text into mean-pooled 768-dim Arabic BERT embedding."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.reshape(1, -1)

# =========================================================
# 4. Prediction function
# =========================================================
def predict_stage(text: str):
    """Predict the stage using the trained Logistic Regression model."""
    emb = get_embedding(text)
    pred = stage_logreg.predict(emb)[0]
    prob = stage_logreg.predict_proba(emb).max()

    return {
        "model": "logreg",
        "stage_class": int(pred),
        "confidence": float(prob)
    }

# =========================================================
# 5. Example test
# =========================================================
if __name__ == "__main__":
    example_text = "تم اتخاذ الإجراءات بسرعة لكن المتابعة لم تكن كاملة"
    print(predict_stage(example_text))
