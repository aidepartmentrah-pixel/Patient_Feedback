"""
This function loops through a test table of patient feedback, classifies each
row using the text-based classifier, and generates a report comparing
predictions with the stored true labels in the database. Progress is shown
with a progress bar.
"""
import os
import sqlite3
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from models_directory.Classification_Models.package_models import classify_feedback_encoded

def generate_classification_report_from_text(db_path: str, test_table: str):
    SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))



    """
    Args:
        db_path (str): Path to SQLite database.
        test_table (str): Name of the test table containing text and true labels.

    Returns:
        dict: A dictionary of classification reports per label.
    """
    # Connect to DB
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Fetch text columns and true labels
    cur.execute(f"""
        SELECT rowid, complaint_text, immediate_action, taken_action,
               domain, category, sub_category, severity_level, stage, harm_level
        FROM {test_table}
    """)
    rows = cur.fetchall()
    conn.close()

    # Initialize storage for true and predicted labels
    true_labels = {k: [] for k in ["domain", "category", "sub_category", "severity", "stage", "harm_level"]}
    pred_labels = {k: [] for k in ["domain", "category", "sub_category", "severity", "stage", "harm_level"]}

    # Loop through each row and classify with progress bar
    for row in tqdm(rows, desc="Classifying feedback"):
        complaint_text = row[1] or ""
        immediate_action = row[2] or ""
        taken_action = row[3] or ""

        result = classify_feedback_encoded(
            text_1=complaint_text,
            text_2=immediate_action,
            text_3=taken_action
        )

        # Store predictions
        pred_labels["domain"].append(result["domain_id"])
        pred_labels["category"].append(result["category_id"])
        pred_labels["sub_category"].append(result["sub_category_id"])
        pred_labels["severity"].append(result["severity_id"])
        pred_labels["stage"].append(result["stage_id"])
        pred_labels["harm_level"].append(result["harm_level_id"])

        # Store true labels
        true_labels["domain"].append(row[4])
        true_labels["category"].append(row[5])
        true_labels["sub_category"].append(row[6])
        true_labels["severity"].append(row[7])
        true_labels["stage"].append(row[8])
        true_labels["harm_level"].append(row[9])

    # Generate reports
    reports = {}
    for key in true_labels.keys():
        print(f"\n==== {key.upper()} ====")
        report = classification_report(true_labels[key], pred_labels[key], output_dict=True)
        acc = accuracy_score(true_labels[key], pred_labels[key])
        print(classification_report(true_labels[key], pred_labels[key]))
        print(f"Accuracy: {acc:.4f}")
        reports[key] = {"report": report, "accuracy": acc}

    return reports


if __name__ == "__main__":

    SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(SCRIPT_PATH, "../../patient_feedback_ml.db")
    TEST_TABLE = "table_feedback_test"
    reports = generate_classification_report_from_text(db_path=DB_PATH, test_table=TEST_TABLE)
    for label, info in reports.items():
        print(f"{label.upper()} Accuracy: {info['accuracy']:.4f}")