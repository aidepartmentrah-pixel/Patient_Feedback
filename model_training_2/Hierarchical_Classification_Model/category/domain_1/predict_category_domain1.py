import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import joblib
from xgboost import XGBClassifier

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "vocab_models")

MPNET_MODEL_PATH = (
    r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback"
    r"\model_training_2\model_storage\mpnet_embeddings"
)

# ============================================================
# LOAD OFFLINE MPNet MODEL (ONCE ONLY)
# ============================================================


tokenizer = AutoTokenizer.from_pretrained(MPNET_MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(MPNET_MODEL_PATH, local_files_only=True)
model.eval()


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

lr = joblib.load(os.path.join(MODEL_DIR, "lr_category_domain1.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "rf_category_domain1.pkl"))

xgb = XGBClassifier()
xgb.load_model(os.path.join(MODEL_DIR, "xgb_category_domain1.json"))

# ============================================================
# LABEL MAP (XGB)
# ============================================================

temp_to_label = {0: 5, 1: 7}     # internal → real
label_to_temp = {5: 0, 7: 1}     # real → internal

# ============================================================
# EMBEDDING FUNCTION (Matches training pipeline exactly)
# ============================================================

def embed_mpnet(text: str) -> np.ndarray:
    if text is None:
        text = ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return emb.astype(np.float32)

# ============================================================
# PREDICT FROM EMBEDDING
# ============================================================

def predict_from_embedding(emb: np.ndarray):
    x = emb.reshape(1, -1)

    # LR + RF already trained on real labels
    lr_pred = int(lr.predict(x)[0])
    rf_pred = int(rf.predict(x)[0])

    # XGB: may return raw labels OR probabilities
    raw_xgb = xgb.predict(x)

    if raw_xgb.ndim == 2:         # probability output
        xgb_pred_temp = int(np.argmax(raw_xgb, axis=1)[0])
    else:                         # raw class index
        xgb_pred_temp = int(raw_xgb[0])

    xgb_pred = temp_to_label[xgb_pred_temp]

    return {
        "logistic_regression": lr_pred,
        "random_forest": rf_pred,
        "xgboost": xgb_pred
    }

# ============================================================
# PREDICT FROM RAW TEXT
# ============================================================

def predict_from_text(text: str):
    emb = embed_mpnet(text)
    return predict_from_embedding(emb)

# ============================================================
# TEST EXAMPLE
# ============================================================

if __name__ == "__main__":
    example_text = "The nurse did not take action quickly, and the follow-up was poor."

    print("🔍 Input Text:")
    print(example_text)

    preds = predict_from_text(example_text)

    print("\n📊 Predictions (Real Labels):")
    print(json.dumps(preds, indent=4))
