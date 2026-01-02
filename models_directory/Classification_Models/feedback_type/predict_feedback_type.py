import os
import numpy as np
import joblib
from models_directory.Classification_Models.Stage.modular_functions import get_embedding

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "FeedbackType_OrdinalModel.pkl"
)

LABEL_MAP = {
    1: "Improvement Opportunity",
    2: "Notice",
    3: "Critique Suggestion",
    4: "Other",
}


def predict_feedback_type_from_embedding(embedding: np.ndarray) -> str:
    model = joblib.load(MODEL_PATH)

    if embedding.ndim != 1:
        raise ValueError("Embedding must be 1D")

    pred = model.predict(embedding.reshape(1, -1))[0]
    return LABEL_MAP[int(pred)]


def predict_feedback_type(text: str) -> str:
    emb = get_embedding(text)

    if isinstance(emb, (bytes, bytearray)):
        emb = np.frombuffer(emb, dtype=np.float32)
    else:
        emb = np.asarray(emb, dtype=np.float32)

    return predict_feedback_type_from_embedding(emb)


# --------------------------------------------------
# Manual test
# --------------------------------------------------
if __name__ == "__main__":
    text = "The patient complained about waiting time."
    print("Prediction:", predict_feedback_type(text))
