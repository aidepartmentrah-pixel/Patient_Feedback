import os
import numpy as np
import joblib
from models_directory.Classification_Models.Stage.modular_functions import get_embedding

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "ClassificationEN_Model.pkl")

CLASS_BASE = 78   # MUST match training

# Load model once
_model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Core prediction from embedding
# --------------------------------------------------
def predict_classification_en_from_embedding(embedding: np.ndarray) -> int:
    if embedding.ndim != 1:
        raise ValueError("Embedding must be 1D")

    local_pred = int(_model.predict(embedding.reshape(1, -1))[0])
    real_pred = local_pred + CLASS_BASE
    return int(real_pred)


# --------------------------------------------------
# Prediction from text (unified signature)
# --------------------------------------------------
def predict_classification_en(text: str) -> int:
    emb = get_embedding(text)

    if isinstance(emb, (bytes, bytearray)):
        emb = np.frombuffer(emb, dtype=np.float32)
    else:
        emb = np.asarray(emb, dtype=np.float32)

    return predict_classification_en_from_embedding(emb)


# --------------------------------------------------
# Manual test (won't run in server unless executed directly)
# --------------------------------------------------
if __name__ == "__main__":
    text = "The patient complained about waiting time and delays."
    print("Prediction (REAL ID):", predict_classification_en(text))
