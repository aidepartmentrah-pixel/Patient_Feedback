import os
import sqlite3
import pandas as pd
import json
from sklearn.metrics import accuracy_score, classification_report
from hierarchical_predictor import hierarchical_predict_text


# ================================================================
# CONFIG
# ================================================================
DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "patient_feedback_ml.db")
)
TEST_TABLE = "table_feedback_test"

TEXT1 = "complaint_text"

TRUE_DOMAIN = "domain"
TRUE_CATEGORY = "category"
TRUE_SUBCAT = "sub_category"

OUTPUT_CSV = "hierarchical_predictions_from_db.csv"


# ================================================================
# LOAD TEST DATA
# ================================================================
def load_test_data():
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT 
            rowid,
            {TEXT1},
            {TRUE_DOMAIN},
            {TRUE_CATEGORY},
            {TRUE_SUBCAT}
        FROM {TEST_TABLE}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Replace None → empty text
    df[TEXT1] = df[TEXT1].fillna("")

    return df


# ================================================================
# EVALUATION LOGIC
# ================================================================
def evaluate_hierarchical_model():

    print("📥 Loading test data from database...")
    df = load_test_data()

    print(f"🔢 Loaded {len(df)} test rows.")

    preds_domain = []
    preds_category = []
    preds_subcat = []

    print("🔮 Running hierarchical predictions...")
    for t1 in df[TEXT1]:

        # Only text1 is used
        full_text = t1.strip()
        result = hierarchical_predict_text(full_text)
        preds_domain.append(result["domain"])
        preds_category.append(result["category"])
        preds_subcat.append(result["subcategory"])

    # Store predictions
    df["pred_domain"] = preds_domain
    df["pred_category"] = preds_category
    df["pred_sub_category"] = preds_subcat

    # Save table
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"📁 Predictions saved → {OUTPUT_CSV}")

    # ============================================================
    # METRICS
    # ============================================================
    print("\n============================")
    print("📊 DOMAIN METRICS")
    print("============================")
    print(classification_report(df[TRUE_DOMAIN], df["pred_domain"]))
    print("Accuracy:", accuracy_score(df[TRUE_DOMAIN], df["pred_domain"]))

    print("\n============================")
    print("📊 CATEGORY METRICS")
    print("============================")
    print(classification_report(df[TRUE_CATEGORY], df["pred_category"]))
    print("Accuracy:", accuracy_score(df[TRUE_CATEGORY], df["pred_category"]))

    print("\n============================")
    print("📊 SUBCATEGORY METRICS")
    print("============================")
    print(classification_report(df[TRUE_SUBCAT], df["pred_sub_category"]))
    print("Accuracy:", accuracy_score(df[TRUE_SUBCAT], df["pred_sub_category"]))


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    evaluate_hierarchical_model()
