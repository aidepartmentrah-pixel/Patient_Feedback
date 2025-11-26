import sqlite3
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime

from package_models import classify_feedback

DB_PATH = "patient_feedback_ml.db"
TABLE_NAME = "table_feedback_test"
REPORT_PATH = "model_test_report.txt"


def load_test_data():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    query = f"""
    SELECT
        complaint_text,
        domain,
        category,
        sub_category,
        severity_level,
        stage,
        harm_level
    FROM {TABLE_NAME}
    """

    rows = cur.execute(query).fetchall()
    conn.close()

    return rows


def evaluate_model():
    data = load_test_data()

    true_domain = []
    pred_domain = []

    true_category = []
    pred_category = []

    true_subcategory = []
    pred_subcategory = []

    true_severity = []
    pred_severity = []

    true_stage = []
    pred_stage = []

    true_harm = []
    pred_harm = []

    print(f"Loaded {len(data)} test samples.\n")

    for row in tqdm(data, desc="Evaluating", ncols=80):
        complaint = row[0]
        t_domain = row[1]
        t_category = row[2]
        t_subcategory = row[3]
        t_severity = row[4]
        t_stage = row[5]
        t_harm = row[6]

        result = classify_feedback(
            complaint,
            "",
            "",
            Print=False
        )

        true_domain.append(t_domain)
        pred_domain.append(result["domain"])

        true_category.append(t_category)
        pred_category.append(result["category"])

        true_subcategory.append(t_subcategory)
        pred_subcategory.append(result["sub_category"])

        true_severity.append(t_severity)
        pred_severity.append(result["severity_level"])

        true_stage.append(t_stage)
        pred_stage.append(result["stage"])

        true_harm.append(t_harm)
        pred_harm.append(result["harm_level"])

        # =========================================
        # FIX: Convert predictions and true labels to strings
        # =========================================
    true_domain = [str(x) for x in true_domain]
    pred_domain = [str(x) for x in pred_domain]

    true_category = [str(x) for x in true_category]
    pred_category = [str(x) for x in pred_category]

    true_subcategory = [str(x) for x in true_subcategory]
    pred_subcategory = [str(x) for x in pred_subcategory]

    true_severity = [str(x) for x in true_severity]
    pred_severity = [str(x) for x in pred_severity]

    true_stage = [str(x) for x in true_stage]
    pred_stage = [str(x) for x in pred_stage]

    true_harm = [str(x) for x in true_harm]
    pred_harm = [str(x) for x in pred_harm]

    # SAVE REPORT
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:

        f.write(f"MODEL PERFORMANCE REPORT\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(f"Total Samples: {len(data)}\n\n")

        f.write("=========================================\n")
        f.write("DOMAIN PREDICTION\n")
        f.write("=========================================\n")
        f.write(classification_report(true_domain, pred_domain))
        f.write(f"\nAccuracy: {accuracy_score(true_domain, pred_domain):.4f}\n\n")

        f.write("=========================================\n")
        f.write("CATEGORY PREDICTION\n")
        f.write("=========================================\n")
        f.write(classification_report(true_category, pred_category))
        f.write(f"\nAccuracy: {accuracy_score(true_category, pred_category):.4f}\n\n")

        f.write("=========================================\n")
        f.write("SUBCATEGORY PREDICTION\n")
        f.write("=========================================\n")
        f.write(classification_report(true_subcategory, pred_subcategory))
        f.write(f"\nAccuracy: {accuracy_score(true_subcategory, pred_subcategory):.4f}\n\n")

        f.write("=========================================\n")
        f.write("SEVERITY PREDICTION\n")
        f.write("=========================================\n")
        f.write(classification_report(true_severity, pred_severity))
        f.write(f"\nAccuracy: {accuracy_score(true_severity, pred_severity):.4f}\n\n")

        f.write("=========================================\n")
        f.write("STAGE PREDICTION\n")
        f.write("=========================================\n")
        f.write(classification_report(true_stage, pred_stage))
        f.write(f"\nAccuracy: {accuracy_score(true_stage, pred_stage):.4f}\n\n")

        f.write("=========================================\n")
        f.write("HARM LEVEL PREDICTION\n")
        f.write("=========================================\n")
        f.write(classification_report(true_harm, pred_harm))
        f.write(f"\nAccuracy: {accuracy_score(true_harm, pred_harm):.4f}\n\n")

    print("\nDone!")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    evaluate_model()
