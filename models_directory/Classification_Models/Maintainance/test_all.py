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


def save_reports_as_txt(reports: dict, save_folder: str):
    """
    Saves all classification reports into a single TXT file.
    File name example: classification_testing_report_13_11_2025.txt
    """
    import datetime

    # Create folder if missing
    os.makedirs(save_folder, exist_ok=True)

    # Current date formatted as DD_MM_YYYY
    today = datetime.datetime.now().strftime("%d_%m_%Y")

    # Build full filename
    filename = f"classification_testing_report_{today}.txt"
    file_path = os.path.join(save_folder, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== CLASSIFICATION PERFORMANCE REPORT ===\n")
        f.write(f"Generated on: {today}\n")
        f.write("=========================================\n\n")

        for label, info in reports.items():
            f.write(f"===== {label.upper()} =====\n")
            f.write(f"Accuracy: {info['accuracy']:.4f}\n\n")

            cr = info["report"]
            cr_text = ""

            cr_text += f"{'Label':<20}{'Precision':<12}{'Recall':<12}{'F1-score':<12}{'Support':<12}\n"
            cr_text += "-" * 70 + "\n"

            for cls, metrics in cr.items():
                if cls in ["accuracy", "macro avg", "weighted avg"]:
                    continue
                if isinstance(metrics, dict):
                    cr_text += f"{cls:<20}{metrics.get('precision', 0):<12.4f}{metrics.get('recall', 0):<12.4f}{metrics.get('f1-score', 0):<12.4f}{metrics.get('support', 0):<12}\n"

            # Macro + weighted averages
            for avg_type in ["macro avg", "weighted avg"]:
                m = cr[avg_type]
                cr_text += f"\n{avg_type:<20}{m['precision']:<12.4f}{m['recall']:<12.4f}{m['f1-score']:<12.4f}{m['support']:<12}\n"

            f.write(cr_text)
            f.write("\n" + "=" * 70 + "\n\n")

    print(f"\nTXT report saved at: {file_path}\n")

if __name__ == "__main__":
    SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(SCRIPT_PATH, "../../patient_feedback_ml.db")
    TEST_TABLE = "table_feedback_test"

    reports = generate_classification_report_from_text(db_path=DB_PATH, test_table=TEST_TABLE)

    # Folder beside script
    SAVE_FOLDER = os.path.join(SCRIPT_PATH, "Performance_Reporting")

    save_reports_as_txt(reports, SAVE_FOLDER)

    # Also print summary
    for label, info in reports.items():
        print(f"{label.upper()} Accuracy: {info['accuracy']:.4f}")
