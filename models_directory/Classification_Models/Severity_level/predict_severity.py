from models_directory.Classification_Models.Stage.modular_functions import get_embedding
import numpy as np
import joblib
import os





def predict_severity_from_embedding(emb: np.ndarray) -> int:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "Severity_OrdinalModel.pkl")
    severity_model = joblib.load(MODEL_PATH)
    """
    emb (768D vector) → model → severity level (1–4)
    """
    x = emb.reshape(1, -1)
    pred_ordinal = int(severity_model.predict(x)[0])   # 0–3
    severity_label = pred_ordinal + 1                  # back to 1–4
    return severity_label

def predict_severity(text: str) -> int:
    emb = get_embedding(text)
    return predict_severity_from_embedding((np.frombuffer(emb, dtype=np.float32)))


if __name__ == "__main__":
    example_text = "The patient had died."

    print("Input Text:")
    print(example_text)
    emb = get_embedding(example_text)
    emb = np.frombuffer(emb, dtype=np.float32)


    severity_embedding = predict_severity_from_embedding(emb)
    severity_text = predict_severity(example_text)

    print("\nPredicted Severity Level (1–4):", severity_embedding)
    print("\nPredicted Severity Level (1–4):", severity_text)