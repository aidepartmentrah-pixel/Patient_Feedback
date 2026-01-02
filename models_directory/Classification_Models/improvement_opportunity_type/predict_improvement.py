import os
import numpy as np
import joblib
from models_directory.Classification_Models.Stage.modular_functions import get_embedding

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Improvement_OrdinalModel.pkl")

LABEL_MAP = {
    1: "Ordinary Complaint",
    2: "Red Flag",
    3: "Never Event"
}


# --------------------------------------------------
# Core prediction
# --------------------------------------------------
def predict_improvement_from_embedding(embedding: np.ndarray) -> str:
    """
    Returns:
        Human-readable label:
        - Ordinary Complaint
        - Red Flag
        - Never Event
    """

    if embedding.ndim != 1:
        raise ValueError("Embedding must be 1D")

    model = joblib.load(MODEL_PATH)

    x = embedding.reshape(1, -1)

    # model outputs: 0,1,2
    pred_ordinal = int(model.predict(x)[0])

    # convert → 1,2,3
    label_id = pred_ordinal + 1

    return LABEL_MAP[label_id]


def predict_improvement(text: str) -> str:
    emb = get_embedding(text)

    if isinstance(emb, (bytes, bytearray)):
        emb = np.frombuffer(emb, dtype=np.float32)
    else:
        emb = np.asarray(emb, dtype=np.float32)

    return predict_improvement_from_embedding(emb)


# --------------------------------------------------
# Manual test
# --------------------------------------------------
if __name__ == "__main__":
    example_text = "The patient had died."

    result = predict_improvement(example_text)

    print("Input:", example_text)
    print("Prediction:", result)
