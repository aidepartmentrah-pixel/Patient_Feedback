import os
import sqlite3

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from models_directory.Classification_Models.Stage.modular_functions import get_embedding, l2_normalize

# metrics_list = ["bed_room", "billing_issues", "cleanliness" , "communication_scheduling",
#                 "conflicting_or_wrong_diagnosis", "disagreement_with_discharge", "food",
#                 "location", "medical_clinical_errors"]

metrics_list = ["staff_security_behavior"]



def Embedd_Normalize(metric_name):
    import os
    import json
    from tqdm import tqdm
    import numpy as np

    # -------------------------------
    # Folder paths
    # -------------------------------
    INPUT_DIR = "vocab_dataset"
    OUTPUT_DIR = "vocab_dataset_normalized"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = os.path.join(INPUT_DIR, f"{metric_name}.json")
    output_path = os.path.join(OUTPUT_DIR, f"{metric_name}.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # -------------------------------
    # Load input dataset
    # -------------------------------
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # CASE 1: dictionary {positive:[], ...}  (rare now)
    if isinstance(data, dict):
        sentences = data.get("positive", [])
    # CASE 2: list of sentences
    elif isinstance(data, list):
        sentences = data
    else:
        raise ValueError("Unsupported JSON structure")

    print(f"[INFO] Loaded {len(sentences)} sentences for metric '{metric_name}'")

    # -------------------------------
    # Embed and normalize
    # -------------------------------
    normalized_vectors = []

    for text in tqdm(sentences, desc=f"Embedding {metric_name}", unit="item"):
        emb = get_embedding(text)
        # If model outputs bytes
        if isinstance(emb, bytes):
            emb = np.frombuffer(emb, dtype=np.float32)

        emb_norm = l2_normalize(emb)
        normalized_vectors.append(emb_norm.tolist())

    # -------------------------------
    # Save output
    # -------------------------------
    result = {"vectors": normalized_vectors}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"[DONE] Saved normalized embeddings to: {output_path}")

def predict_metrics_from_embedding(normalized_vec):
    """
    Takes a NORMALIZED embedding vector (1 x 768 or similar)
    and returns predictions for ALL 12 micro-metrics.

    Returns:
        List of tuples: [(metric_name, probability), ...] sorted descending.
    """

    METRICS = [
        "administration_delay",
        "arrival",
        "bed_room",
        "billing_issues",
        "cleanliness",
        "clinical_delay",
        "communication_scheduling",
        "conflicting_or_wrong_diagnosis",
        "disagreement_with_discharge",
        "food",
        "location",
        "medical_clinical_errors"
    ]

    MODEL_ROOT = "vocab_models"

    results = []

    # Loop over all metrics
    for metric in METRICS:
        model_dir = os.path.join(MODEL_ROOT, metric)

        if not os.path.exists(model_dir):
            # Skip if no vocab_models exist
            continue

        # Possible model files
        model_files = {
            "lr": os.path.join(model_dir, f"{metric}_lr.pkl"),
            "rf": os.path.join(model_dir, f"{metric}_rf.pkl"),
            "xgb": os.path.join(model_dir, f"{metric}_xgb.pkl"),
        }

        preds = []

        # Load each classifier and compute prob
        for _, path in model_files.items():
            if os.path.exists(path):
                model = joblib.load(path)
                prob = model.predict_proba([normalized_vec])[0][1]
                preds.append(prob)

        if len(preds) == 0:
            continue

        # Average probability of the classifiers
        score = sum(preds) / len(preds)

        results.append((metric, score))

    # Sort high → low
    results.sort(key=lambda x: x[1], reverse=True)

    return results



def train_ML_Metric_Reader(metric_name):
    import os
    import json
    import numpy as np
    import random
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    import joblib

    # -----------------------------------------------
    # Helpers
    # -----------------------------------------------
    def clean_vectors(vecs, metric_label):
        """
        Removes None, empty lists, and vectors that don't match
        the correct embedding dimension.
        """
        if len(vecs) == 0:
            return []

        # What is correct dimension?
        # Detect the most common length in this list
        lengths = [len(v) for v in vecs if isinstance(v, (list, np.ndarray))]
        if len(lengths) == 0:
            print(f"[FATAL] All embeddings broken in {metric_label}")
            return []

        correct_dim = max(set(lengths), key=lengths.count)

        cleaned = [v for v in vecs if isinstance(v, (list, np.ndarray)) and len(v) == correct_dim]

        removed = len(vecs) - len(cleaned)
        if removed > 0:
            print(f"[WARN] {metric_label}: Removed {removed} invalid embeddings (kept {len(cleaned)})")
            print(f"[WARN] Expected dim: {correct_dim} | Found dims: {set(lengths)}")

        return cleaned

    # -----------------------------------------------
    # Folder preparation
    # -----------------------------------------------
    DATA_DIR = "vocab_dataset_normalized"
    MODEL_ROOT = "vocab_models"
    os.makedirs(MODEL_ROOT, exist_ok=True)

    target_file = os.path.join(DATA_DIR, f"{metric_name}.json")
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Target dataset not found: {target_file}")

    metrics_list = [
        "bed_room", "billing_issues", "cleanliness",
        "communication_scheduling", "conflicting_or_wrong_diagnosis",
        "disagreement_with_discharge", "food",
        "location", "medical_clinical_errors"
    ]

    other_metrics = [m for m in metrics_list if m != metric_name]

    # -----------------------------------------------
    # Load positive vectors
    # -----------------------------------------------
    with open(target_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "vectors" not in data:
        raise ValueError(f"File missing 'vectors': {target_file}")

    positive_vectors = clean_vectors(data["vectors"], metric_name)
    print(f"[INFO] Loaded {len(positive_vectors)} valid positive samples")

    # -----------------------------------------------
    # Load negative vectors
    # -----------------------------------------------
    negative_vectors_all = []

    for other in other_metrics:
        other_file = os.path.join(DATA_DIR, f"{other}.json")

        if not os.path.exists(other_file):
            print(f"[WARN] Missing dataset: {other_file}")
            continue

        with open(other_file, "r", encoding="utf-8") as f:
            other_data = json.load(f)

        if "vectors" not in other_data:
            print(f"[WARN] No 'vectors' in {other_file}")
            continue

        cleaned = clean_vectors(other_data["vectors"], other)
        negative_vectors_all.extend(cleaned)
        print(f"[INFO] Loaded {len(cleaned)} negatives from {other}")

    if len(negative_vectors_all) == 0:
        raise RuntimeError("No valid negative vectors found.")

    # -----------------------------------------------
    # Balance sampling
    # -----------------------------------------------
    NEG_TARGET = len(positive_vectors)

    if len(negative_vectors_all) <= NEG_TARGET:
        negative_vectors = negative_vectors_all
    else:
        negative_vectors = random.sample(negative_vectors_all, NEG_TARGET)

    print(f"[INFO] Negatives selected: {len(negative_vectors)}")

    # -----------------------------------------------
    # Final training matrix
    # -----------------------------------------------
    try:
        X = np.array(positive_vectors + negative_vectors, dtype=np.float32)
    except Exception as e:
        print("[FATAL] Failed to create numpy matrix!")
        print(
            f"Lengths found: {set(len(v) for v in positive_vectors + negative_vectors if isinstance(v, (list, np.ndarray)))}")
        raise e

    y = np.array(
        [1] * len(positive_vectors) + [0] * len(negative_vectors),
        dtype=int
    )

    print(f"[INFO] Final Training Shape: {X.shape}")
    print(f"[INFO] Positives: {len(positive_vectors)} | Negatives: {len(negative_vectors)}")

    # -----------------------------------------------
    # Train vocab_models
    # -----------------------------------------------
    print("[TRAIN] Logistic Regression")
    lr = LogisticRegression(max_iter=300)
    lr.fit(X, y)

    print("[TRAIN] Random Forest")
    rf = RandomForestClassifier(n_estimators=250)
    rf.fit(X, y)

    print("[TRAIN] XGBoost")
    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist"
    )
    xgb.fit(X, y)

    # -----------------------------------------------
    # Save vocab_models
    # -----------------------------------------------
    model_dir = os.path.join(MODEL_ROOT, metric_name)
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(lr, os.path.join(model_dir, f"{metric_name}_lr.pkl"))
    joblib.dump(rf, os.path.join(model_dir, f"{metric_name}_rf.pkl"))
    joblib.dump(xgb, os.path.join(model_dir, f"{metric_name}_xgb.pkl"))

    print(f"[DONE] Models saved in: {model_dir}")

    # -----------------------------------------------
    # Reports
    # -----------------------------------------------
    def evaluate_and_save(model, name):

        preds = model.predict(X)

        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        cm = confusion_matrix(y, preds)

        report_path = os.path.join(model_dir, f"{metric_name}_{name}_report.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Metric: {metric_name}\n\n")
            f.write(f"Accuracy:  {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall:    {rec:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm))

        print(f"[REPORT] {name} report saved → {report_path}")

    evaluate_and_save(lr, "lr")
    evaluate_and_save(rf, "rf")
    evaluate_and_save(xgb, "xgb")

    print("[DONE] Training and reporting complete.")

def train_ML_Metric_Mapper_Numeric_No_Progress_Bar():

    # =============================
    # PATHS
    # =============================
    DB_PATH = r"/models_directory\patient_feedback_ml.db"

    MODEL_SAVE_ROOT = r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\model_training_2\Stage\Training_Internal_Metrics\vocab_models\train_ML_Metric_Mapper_Numeric"
    os.makedirs(MODEL_SAVE_ROOT, exist_ok=True)

    TRAIN_TABLE = "table_feedback_train"
    TEST_TABLE  = "table_feedback_test"

    # Columns containing embeddings
    EMB_COLS = [
        "sentence_1_embedding",
        "sentence_2_embedding",
        "sentence_3_embedding",
        "sentence_4_embedding",
        "sentence_5_embedding",
        "sentence_6_embedding",
    ]

    METRICS = [
        "administration_delay",
        "arrival",
        "bed_room",
        "billing_issues",
        "cleanliness",
        "clinical_delay",
        "communication_scheduling",
        "conflicting_or_wrong_diagnosis",
        "disagreement_with_discharge",
        "food",
        "location",
        "medical_clinical_errors"
    ]

    # =============================
    # 1. DATABASE READING FUNCTION
    # =============================
    def load_table(table_name):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        cols = ", ".join(EMB_COLS + ["stage"])
        cur.execute(f"SELECT {cols} FROM {table_name}")
        rows = cur.fetchall()
        conn.close()
        return rows

    # =============================
    # 2. CONVERT ROW → FEATURE VECTOR
    # =============================
    def row_to_feature_vector(row):
        """
        row = (6 embeddings..., stage_label)
        """

        sentence_embs = row[:-1]
        stage_label = row[-1]

        # store max prediction per metric across the 6 sentences
        metric_max_scores = {m: [] for m in METRICS}

        for emb in sentence_embs:

            if emb is None:
                continue

            # decode embedding if stored as bytes
            if isinstance(emb, bytes):
                emb = np.frombuffer(emb, dtype=np.float32)

            emb = l2_normalize(emb)

            # Call prediction (returns sorted list)
            metric_scores = predict_metrics_from_embedding(emb)

            score_dict = {name:score for name,score in metric_scores}

            # Add to metric lists
            for m in METRICS:
                metric_max_scores[m].append(score_dict.get(m, 0))

        # If no embeddings are available, fill zeros
        final_vector = [
            max(metric_max_scores[m]) if metric_max_scores[m] else 0
            for m in METRICS
        ]

        return final_vector, stage_label

    # =============================
    # 3. LOAD TRAIN + TEST
    # =============================
    train_rows = load_table(TRAIN_TABLE)
    test_rows  = load_table(TEST_TABLE)

    X_train, y_train = [], []
    for row in train_rows:
        vec, lab = row_to_feature_vector(row)
        X_train.append(vec)
        y_train.append(lab)

    X_test, y_test = [], []
    for row in test_rows:
        vec, lab = row_to_feature_vector(row)
        X_test.append(vec)
        y_test.append(lab)

    X_train = np.array(X_train)
    X_test  = np.array(X_test)

    # =============================
    # 4. TRAIN MODELS
    # =============================

    MODELS = {
        "lr": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(n_estimators=400),
        "xgb": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            learning_rate=0.05,
            eval_metric="logloss"
        )
    }

    for name, model in MODELS.items():

        print(f"\n🚀 Training Model: {name}")

        model.fit(X_train, y_train)

        # Save model
        model_path = os.path.join(MODEL_SAVE_ROOT, f"ML_metric_mapper_{name}.pkl")
        joblib.dump(model, model_path)

        # Predict
        preds = model.predict(X_test)

        # Score
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)

        # Save report
        with open(os.path.join(MODEL_SAVE_ROOT, f"report_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Model: {name}\n\n")
            f.write(f"Accuracy: {acc}\n\n")
            f.write(report)

        print(f"✔ {name} done. Accuracy={acc}")

    print("\n🎉 Training complete. Models and reports saved.")

def train_ML_Metric_Mapper_Numeric():

    import os
    import sqlite3
    import joblib
    import numpy as np
    from tqdm import tqdm
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, classification_report

    # =========================================
    # HELPERS
    # =========================================

    def l2_normalize(vec):
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    def remove_bad_rows(rows):
        """Remove DB rows where stage is None."""
        cleaned = []
        removed = 0
        for r in rows:
            if r[-1] is None:
                removed += 1
                continue
            cleaned.append(r)
        return cleaned, removed

    def safe_to_int(arr, name):
        """Convert labels to int safely."""
        cleaned = []
        removed = 0
        for v in arr:
            if v is None:
                removed += 1
                continue
            cleaned.append(int(v))
        if removed > 0:
            print(f"⚠ Removed {removed} rows from {name} due to missing labels.")
        return np.array(cleaned)

    # =========================================
    # PATHS
    # =========================================

    DB_PATH = r"/models_directory\patient_feedback_ml.db"

    MODEL_SAVE_ROOT = r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\model_training_2\Stage\Training_Internal_Metrics\vocab_models\train_ML_Metric_Mapper_Numeric"
    os.makedirs(MODEL_SAVE_ROOT, exist_ok=True)

    TRAIN_TABLE = "table_feedback_train"
    TEST_TABLE  = "table_feedback_test"

    EMB_COLS = [
        "sentence_1_embedding",
        "sentence_2_embedding",
        "sentence_3_embedding",
        "sentence_4_embedding",
        "sentence_5_embedding",
        "sentence_6_embedding",
    ]

    METRICS = [
        "administration_delay",
        "arrival",
        "bed_room",
        "billing_issues",
        "cleanliness",
        "clinical_delay",
        "communication_scheduling",
        "conflicting_or_wrong_diagnosis",
        "disagreement_with_discharge",
        "food",
        "location",
        "medical_clinical_errors",
        "staff_security_behavior"
    ]

    # =========================================
    # Load all 12 metric classifiers
    # =========================================

    METRIC_MODEL_ROOT = r"vocab_models"
    metric_classifiers = {}

    for metric in METRICS:
        folder = os.path.join(METRIC_MODEL_ROOT, metric)

        if not os.path.exists(folder):
            print(f"[WARN] Missing folder {metric}")
            continue

        models = {}
        for fname in os.listdir(folder):
            if fname.endswith(".pkl"):
                try:
                    models[fname] = joblib.load(os.path.join(folder, fname))
                except:
                    print(f"[WARN] Cannot load model {fname}")

        if models:
            metric_classifiers[metric] = models
        else:
            print(f"[WARN] No vocab_models found for metric {metric}")

    # =========================================
    # Fast metric predictor
    # =========================================

    metric_cache = {}

    def predict_metrics_from_embedding_cached(emb):
        emb_key = emb.tobytes()

        if emb_key in metric_cache:
            return metric_cache[emb_key]

        scores = []

        for metric in METRICS:

            if metric not in metric_classifiers:
                scores.append((metric, 0.0))
                continue

            preds = []
            for _, model in metric_classifiers[metric].items():
                prob = model.predict_proba([emb])[0][1]
                preds.append(prob)

            avg_prob = sum(preds) / len(preds) if preds else 0.0
            scores.append((metric, avg_prob))

        scores.sort(key=lambda x: x[1], reverse=True)
        metric_cache[emb_key] = scores
        return scores

    # =========================================
    # Load DB rows
    # =========================================

    def load_table(tb):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cols = ", ".join(EMB_COLS + ["stage"])
        cur.execute(f"SELECT {cols} FROM {tb}")
        rows = cur.fetchall()
        conn.close()
        return rows

    print("\n📥 Loading DB...")
    train_rows = load_table(TRAIN_TABLE)
    test_rows  = load_table(TEST_TABLE)

    # Remove rows missing stage label
    train_rows, removed_train = remove_bad_rows(train_rows)
    test_rows,  removed_test  = remove_bad_rows(test_rows)

    print(f"✔ TRAIN rows loaded: {len(train_rows)} (removed {removed_train})")
    print(f"✔ TEST rows loaded : {len(test_rows)}  (removed {removed_test})")

    # =========================================
    # Feature extractor
    # =========================================

    def row_to_feature_vector(row):
        sentence_embs = row[:-1]
        stage_label   = row[-1]

        metric_scores = {m: [] for m in METRICS}

        for emb in sentence_embs:

            if emb is None:
                continue

            if isinstance(emb, bytes):
                emb = np.frombuffer(emb, dtype=np.float32)

            emb = l2_normalize(emb)

            preds = predict_metrics_from_embedding_cached(emb)
            pmap = {m: s for m, s in preds}

            for m in METRICS:
                metric_scores[m].append(pmap.get(m, 0.0))

        final_vec = [
            max(metric_scores[m]) if metric_scores[m] else 0.0
            for m in METRICS
        ]

        return final_vec, stage_label

    # =========================================
    # Convert DB → ML vectors
    # =========================================

    print("\n🔁 Extracting features (TRAIN)...")
    X_train, y_train = [], []

    for row in tqdm(train_rows):
        vec, lab = row_to_feature_vector(row)
        X_train.append(vec)
        y_train.append(lab)

    print("\n🔁 Extracting features (TEST)...")
    X_test, y_test = [], []

    for row in tqdm(test_rows):
        vec, lab = row_to_feature_vector(row)
        X_test.append(vec)
        y_test.append(lab)

    X_train = np.array(X_train)
    X_test  = np.array(X_test)

    y_train = safe_to_int(y_train, "TRAIN")
    y_test  = safe_to_int(y_test,  "TEST")

    print("✔ Unique TRAIN labels:", set(y_train))

    # =========================================
    # Train ML vocab_models
    # =========================================

    MODELS = {
        "lr": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(n_estimators=400),
        "xgb": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss"
        )
    }

    print("\n🏋 Training ML vocab_models...\n")

    for name, model in MODELS.items():

        print(f"\n🚀 Training {name}...")

        model.fit(X_train, y_train)

        joblib.dump(model, os.path.join(MODEL_SAVE_ROOT, f"ML_metric_mapper_{name}.pkl"))

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        report_path = os.path.join(MODEL_SAVE_ROOT, f"report_{name}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Accuracy: {acc}\n\n")
            f.write(classification_report(y_test, preds))

        print(f"✔ {name} accuracy: {acc}")

    print("\n🎉 Training finished successfully!")

def train_ML_Metric_Mapper_Numeric_prompt():
    # Go to the database train.
         # The code is at (C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\models_directory\Stage\Training_Internal_Metrics)
         # The database is at : (C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\models_directory)
         # TABLE = "table_feedback_test" "table_feedback_train"
    #
    #     embedding_columns = [
    #         "sentence_1_embedding",
    #         "sentence_2_embedding",
    #         "sentence_3_embedding",
    #         "sentence_4_embedding",
    #         "sentence_5_embedding",
    #         "sentence_6_embedding",
    # get the 6 sentences for each row sentence_1_embedding .. sentence_6_embedding (from 1 to 6)
    # use the function (predict_metrics_from_embedding) to predict the embddings and select for each metric the max of 6
    # Build the table of al these predictions 12 features 371 train and 93 test
    # The classificaiton of each record is the row "stage" from the dtabase
    # Train LR, RF and XB and this table of 12 by 317 and save them in a new folder called "train_ML_Metric_Mapper_Numeric" in : C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\models_directory\Stage\Training_Internal_Metrics\vocab_models
    # Test the model and give me a report on this mapping in the same directory



    pass

def train_ML_Metric_Mapper_Binary():
    pass

# Testing Function
def validate_dataset(X, y, name="DATASET"):
    print(f"\n🔍 Validating {name} ...")

    # If dataset missing
    if X is None or y is None:
        print("❌ ERROR: X or y not provided.")
        return

    # Convert to list if numpy
    try:
        total = len(y)
    except:
        print("❌ ERROR: y has no length.")
        return

    # Count missing labels
    none_labels = sum(1 for l in y if l is None)

    print(f"Total samples: {total}")
    print(f"Missing labels: {none_labels}")

    if none_labels > 0:
        print("⚠ WARNING: Some rows have NULL stage labels.")

    # Check feature vector size
    if len(X) > 0:
        first_shape = len(X[0])
        print(f"Feature vector length: {first_shape}")
    else:
        print("⚠ X is empty (no feature vectors loaded).")

    # Check dimension consistency
    bad_rows = 0
    for i, vec in enumerate(X):
        if len(vec) != len(X[0]):
            bad_rows += 1

    if bad_rows > 0:
        print(f"⚠ {bad_rows} rows have mismatched vector lengths.")
    else:
        print("✔ All vectors consistent.")


def load_data_for_testing():
    import sqlite3
    DB_PATH = r"/models_directory\patient_feedback_ml.db"
    TRAIN_TABLE = "table_feedback_train"
    TEST_TABLE  = "table_feedback_test"

    EMB_COLS = [
        "sentence_1_embedding",
        "sentence_2_embedding",
        "sentence_3_embedding",
        "sentence_4_embedding",
        "sentence_5_embedding",
        "sentence_6_embedding",
    ]

    def load_table(tb):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cols = ", ".join(EMB_COLS + ["stage"])
        cur.execute(f"SELECT {cols} FROM {tb}")
        rows = cur.fetchall()
        conn.close()
        return rows

    print("\n📥 Loading data directly from DB...")

    train_rows = load_table(TRAIN_TABLE)
    test_rows  = load_table(TEST_TABLE)

    print(f"TRAIN rows loaded: {len(train_rows)}")
    print(f"TEST  rows loaded: {len(test_rows)}")

    return train_rows, test_rows



if __name__ == "__main__":

    train_ML_Metric_Mapper_Numeric()



