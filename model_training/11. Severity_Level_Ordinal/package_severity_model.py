"""
package_severity_model.py

Load saved severity pipeline and run inference.

Provides:
 - predict_embedding(embedding)   # embedding = 1D numpy array (768 + engineered dims)
 - predict_text(patient_text, hospital_text)  # optional: builds embedding if MPNet present
"""

import os
import joblib
import json
import numpy as np
from pathlib import Path

# optional transformer embeddings
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "severity"

# Load metadata
meta = json.load(open(MODEL_DIR / "severity_meta.json", encoding="utf-8"))
scaler = joblib.load(MODEL_DIR / meta["scaler"])

# load primary ordinal or regressor model
ord_model_path = MODEL_DIR / meta["ordinal_model"]
ordinal_model = joblib.load(ord_model_path)

# Also load comparison vocab_models if needed
lr = joblib.load(MODEL_DIR / "severity_lr.pkl") if (MODEL_DIR / "severity_lr.pkl").exists() else None
rf = joblib.load(MODEL_DIR / "severity_rf.pkl") if (MODEL_DIR / "severity_rf.pkl").exists() else None
xgb = joblib.load(MODEL_DIR / "severity_xgb.pkl") if (MODEL_DIR / "severity_xgb.pkl").exists() else None

# If you want text->embedding conversion (optional)
MPNET_LOCAL = HERE.parent / "model_storage" / "mpnet_embeddings"  # adjust if you saved offline
if TRANSFORMERS_AVAILABLE and MPNET_LOCAL.exists():
    tokenizer = AutoTokenizer.from_pretrained(str(MPNET_LOCAL))
    embed_model = AutoModel.from_pretrained(str(MPNET_LOCAL))
    embed_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_model.to(device)
else:
    tokenizer = None
    embed_model = None

# same engineered feature builder used during training
def build_engineered_features_single(patient_text, hospital_text):
    harm_kws = ["مضاعف", "مضاعفات", "إصابة", "نقل", "وفاة", "readmit", "readmitted", "re-admit", "عودة", "عدوى", "infection", "bleeding", "ألم", "حالة حرجة"]
    resp_kws = ["تقصير", "مهمل", "إهمال", "تأخير", "غير كاف", "لا يوجد", "رفض", "رفض الاستماع", "غير متعاون", "رفض"]

    pt = (patient_text or "").lower()
    ht = (hospital_text or "").lower()
    len_pat = len(patient_text or "")
    len_hosp = len(hospital_text or "")

    def kw_count_text(txt, kws):
        return sum(txt.count(k) for k in kws)

    harm_count = kw_count_text(pt + " " + ht, harm_kws)
    resp_count = kw_count_text(pt + " " + ht, resp_kws)
    return np.array([len_pat, len_hosp, harm_count, resp_count], dtype=np.float32)

def get_embedding_from_text(patient_text, hospital_text):
    if not TRANSFORMERS_AVAILABLE or embed_model is None:
        raise RuntimeError("Transformers/MPNet not available locally for text->embedding conversion.")
    txt = (patient_text or "") + " " + (hospital_text or "")
    enc = tokenizer([txt], return_tensors="pt", truncation=True, padding=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = embed_model(**enc)
    emb = out.last_hidden_state.mean(dim=1).cpu().numpy().reshape(-1)
    return emb

def predict_embedding(emb):
    """
    emb: 1D numpy array for embedding_text123 (length 768) OR full concatenated vector (embedding+engineered)
    The function will detect if engineered dims missing and will append zeros (not recommended).
    Returns: integer severity prediction
    """
    emb = np.array(emb, dtype=np.float32).reshape(1, -1)

    # if emb length equals embedding only, append zeros for engineered features (but warn)
    expected_feat_count = len(meta["features"])  # first is embedding_text123 then engineered names
    # compute number of engineered dims:
    num_engineered = expected_feat_count - 1
    embedding_dim = emb.shape[1]
    if embedding_dim == 768:
        engineered = np.zeros((1, num_engineered), dtype=np.float32)
        X = np.hstack([emb, engineered])
    else:
        X = emb

    Xs = scaler.transform(X)
    pred = ordinal_model.predict(Xs)[0]
    return int(pred)

def predict_text(patient_text, hospital_text):
    if TRANSFORMERS_AVAILABLE and embed_model is not None:
        emb = get_embedding_from_text(patient_text, hospital_text)
        return predict_embedding(emb)
    else:
        raise RuntimeError("Text->embedding not available in this environment. Use predict_embedding.")

# quick test
if __name__ == "__main__":
    # example usage: you can feed a precomputed embedding or text if model present
    try:
        print("Loaded severity pipeline.")
        # dummy test: load an example embedding from DB if available
        print("Call predict_embedding(np.zeros(768)) to test.")
    except Exception as e:
        print("Error loading pipeline:", e)
