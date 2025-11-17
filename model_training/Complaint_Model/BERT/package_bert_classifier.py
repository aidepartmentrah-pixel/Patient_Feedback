"""
package_bert_classifier.py

Loads the trained Multi-Head Arabic BERT model and predicts:
domain, category, sub_category, classification_ar
for a given Arabic text.
"""

import torch
import torch.nn as nn
import joblib
from transformers import AutoTokenizer, AutoModel
import os

# =========================================================
# CONFIG
# =========================================================
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "multihead_bert_classifier.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")

DEVICE = torch.device("cpu")

# =========================================================
# MODEL DEFINITION
# =========================================================
class MultiHeadBERT(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        hidden_size = self.bert.config.hidden_size

        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, n_cls)
            for name, n_cls in num_classes.items()
        })

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        pooled = self.dropout(pooled)
        return {name: head(pooled) for name, head in self.heads.items()}

# =========================================================
# LOAD EVERYTHING
# =========================================================
print("Loading tokenizer, encoders, and model...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
encoders = joblib.load(ENCODERS_PATH)

num_classes = {
    "domain": len(encoders["domain"].classes_),
    "category": len(encoders["category"].classes_),
    "sub_category": len(encoders["sub_category"].classes_),
    "classification_ar": len(encoders["classification_ar"].classes_),
}

model = MultiHeadBERT(MODEL_NAME, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded successfully!")

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_feedback(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    results = {}
    for name, logits in outputs.items():
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        pred_idx = probs.argmax()
        label = encoders[name].inverse_transform([pred_idx])[0]
        results[name] = {"label": label, "confidence": float(probs[pred_idx])}
    return results

# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    example_text = "الطبيب كان متعاون ولكن الانتظار طويل جداً."
    print("Prediction:", predict_feedback(example_text))
