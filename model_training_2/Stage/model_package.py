from model_training_2.Stage.modular_functions import get_embedding
from model_training_2.Stage.Training_Internal_Metrics.internal_metrics import split_max_score
import numpy as np
import joblib
import json
import os



def classify_stage_Rule_Based(sentence: str):
    """
    Uses split_max_score() to compute metric scores and then applies:

    Rule:
        If two or more of the top 3 agree → pick that
        Else → pick the highest-scoring metric

    Returns:
        {
            "top3": List[(metric, score)],
            "final_metric": str,
            "segment_scores": dict
        }
    """
    # Get sorted metrics and per-segment scores
    sorted_metrics, per_segment_scores = split_max_score(sentence)

    # In case no scores returned
    if len(sorted_metrics) == 0:
        return {
            "top3": [],
            "final_metric": None,
            "segment_scores": per_segment_scores
        }

    # Top 3
    top3 = sorted_metrics[:3]

    # Extract only the metric names
    m1, m2, m3 = [m for m, _ in top3]

    # Apply the combination rule
    if m1 == m2 or m1 == m3:
        # m1 wins because it ties with someone
        final_metric = m1
    elif m2 == m3:
        # m2 wins because it ties with m3
        final_metric = m2
    else:
        # No ties → highest score wins
        final_metric = m1

    return {
        "top3": top3,
        "final_metric": final_metric,
        "segment_scores": per_segment_scores
    }

def classify_stage_Numerical(sentence: str) -> str:
    # Path to the trained model folder
    MODEL_DIR = r"/model_training_2/Stage/Training_Internal_Metrics/vocab_models/train_ML_Metric_Mapper_Numeric"

    # Model and label map filenames
    MODEL_PATH = os.path.join(MODEL_DIR, "mapper_model.pkl")
    LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

    """
    Classify a sentence into one of the Stage categories using the trained
    numerical mapper model. Returns the stage name as a string.
    """

    # Edge case: empty
    if sentence is None or str(sentence).strip() == "":
        return None

    # Load trained classifier
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    clf = joblib.load(MODEL_PATH)

    # Load label mapping
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map not found at: {LABEL_MAP_PATH}")

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    # --- Convert input sentence to embedding ---
    vec = get_embedding(sentence)     # Must return a 1D numpy vector
    if not isinstance(vec, np.ndarray):
        vec = np.array(vec, dtype=np.float32)
    # Predict
    pred = clf.predict(vec.reshape(1, -1))[0]
    # Map numeric → label text
    return label_map.get(str(int(pred)), None)

def classify_stage_Binary(sentence: str):
    pass

def classify_stage_Score_Based(sentence: str, Troubleshoot = False):
    STAGE_GROUPS = {
        "admission": [
            "administration_delay",
            "arrival",
            "bed_room",
            "billing_issues"
        ],
        "care_on_ward": [
            "cleanliness",
            "food",
            "location",
            "staff_security_behavior"
        ],
        "discharge": [
            "billing_issues",
            "disagreement_with_discharge"
        ],
        "examination_diagnosis": [
            "clinical_delay",
            "conflicting_or_wrong_diagnosis"
        ],
        "operation": [
            "communication_scheduling",
            "medical_clinical_errors"
        ]
    }
    STAGE_ENCODINGS = {
        "examination_diagnosis": 1,
        "admission": 2,
        "care_on_ward": 4,
        "discharge": 6,
        "operation": 8,
        "unspecified": 9
    }

    # Run the metric scoring using your function
    sorted_metrics, per_segment_scores = split_max_score(paragraph = sentence,Troubleshoot= Troubleshoot)
    if len(sorted_metrics) == 0:
        return {
            "stage_scores": {},
            "chosen_stage": None,
            "sorted_individual_metrics": [],
            "segment_scores": per_segment_scores
        }

    # Convert sorted list to dict {metric: score}
    metric_dict = {m: s for m, s in sorted_metrics}
    stage_scores = {}

    # Compute score for each stage
    for stage, metrics in STAGE_GROUPS.items():
        values = [metric_dict.get(m, 0.0) for m in metrics]
        if len(values) == 0:
            stage_scores[stage] = {
                "max": 0.0,
                "avg": 0.0,
                "metrics_used": {}
            }
            continue

        stage_scores[stage] = {
            "max": max(values),
            "avg": sum(values) / len(values),
            "metrics_used": {m: metric_dict.get(m, 0.0) for m in metrics}
        }

    # ---- Decision Step ----
    # 1️⃣ Find highest max across stages
    best_max = max(stage_scores[s]["max"] for s in stage_scores)

    # Filter stages that share this max
    candidates = [s for s in stage_scores if stage_scores[s]["max"] == best_max]

    if len(candidates) == 1:
        chosen_stage = candidates[0]
    else:
        # 2️⃣ Use average to break ties
        chosen_stage = max(
            candidates,
            key=lambda s: stage_scores[s]["avg"]
        )

    variable =  {
        "stage_scores": stage_scores,
        "chosen_stage": chosen_stage,
        "sorted_individual_metrics": sorted_metrics[:3],
        "segment_scores": per_segment_scores
    }

    if Troubleshoot:
        # 🔷 PRINT THE FINAL CLASSIFICATION
        print(f'==============================================')
        print(f"\n🧩 Sentence classified as: {chosen_stage}")
        print("Top contributing metrics:")
        for m, score in sorted_metrics[:3]:
            print(f" - {m}: {score:.3f}")
        print(variable)


    stage_encoding = STAGE_ENCODINGS.get(chosen_stage, 9)

    return stage_encoding

text = """
اعترض مرافق المريضة ""أنه بتاريخ 20-4-2025 .حضروا إلى الطوارئ وإعتراضهم حول عدم نظافة الحمامات بالطوارئ(" الحمامات بالطوارئ ابدا مش نظيفة... بقينا يوم ونص بالطوارئ الحمامات ما بينفات عليهن ابدا...")
"""

if __name__ == "__main__":
    number = classify_stage_Score_Based(text, True)
    print(number)
