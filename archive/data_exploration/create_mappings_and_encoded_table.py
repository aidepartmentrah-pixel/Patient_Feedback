"""
create_mappings_and_encoded_table.py

Purpose:
--------
1. Create an Excel and JSON mapping of old → new column names.
2. Rename columns in the database to consistent English names.
3. Create a new table (`patient_feedback_clean`) in the existing SQLite database.

Input:
------
- patient_feedback.db (must exist in the same folder)
- Existing table: patient_feedback

Output:
-------
- column_name_mapping.xlsx  (Human-readable)
- column_name_mapping.json  (Programmatic use)
- New table: patient_feedback_clean (with standardized names)
"""

import os
import sqlite3
import pandas as pd
import json

# === CONFIG ===
DB_PATH = "patient_feedback.db"
OLD_TABLE = "patient_feedback"
NEW_TABLE = "patient_feedback_clean"
MAPPING_EXCEL = "column_name_mapping.xlsx"
MAPPING_JSON = "column_name_mapping.json"

# === COLUMN NAME MAPPING ===
COLUMN_MAPPING = {
    "تاريخ_تلقي_الملاحظة": "feedback_received_date",
    "الرقم": "record_id",
    "P__Full_Name": "patient_full_name",
    "قسم_الصادر": "issuing_department",
    "قسم_المعني": "target_department",
    "المصدر1": "source_1",
    "النوع": "feedback_type",
    "Domain__CLINICAL_MANAGEMENT_RELATIONAL_": "domain",
    "Category__Safety_Quality_Environment_": "category",
    "Sub_Category": "subcategory",
    "New_Classifiaction_in_Arabic": "classification_ar",
    "New_Classifiaction_in_English": "classification_en",
    "محتوى_الشكوى__Raw_Content_": "complaint_text",
    "Immediate_Action__خدمات_المرضى_القسم_": "immediate_action",
    "الإجراءات_المتخذة__القسم_الدائرة_الإدارة_": "taken_action",
    "Severity___1r_High__2r_Medium_3r_Low": "severity_level",
    "Stage___1r_Admissions__2r_Examination__Diagnosis_3r_Care_on_the_Ward_41Operation": "stage",
    "Harm___1r_No_Harm__2r_Minor_3r_Moderate_4r_Severe_5r_Death": "harm_level",
    "Status___1r_Closed__2r_In_Progress_3r_Open": "status",
    "نوع_'فرصة_التحسين'___1r_Ordinary_Complaint_2r_Red_Flag_3r_Never_Event": "improvement_opportunity_type",
    "embedding_text1": "embedding_text1",
    "embedding_text2": "embedding_text2",
    "embedding_text3": "embedding_text3",
    "embedding_text123": "embedding_text123",
    "embedding_text23": "embedding_text23",
    "embedding_model": "embedding_model"
}

# === MAIN LOGIC ===
def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    print(f"✅ Connected to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    # Load the old table into pandas
    df = pd.read_sql_query(f"SELECT * FROM {OLD_TABLE}", conn)
    print(f"Loaded table '{OLD_TABLE}' with {len(df)} rows and {len(df.columns)} columns")

    # Apply column renaming
    renamed_cols = {col: COLUMN_MAPPING.get(col, col) for col in df.columns}
    df.rename(columns=renamed_cols, inplace=True)
    print(f"✅ Columns renamed to standardized English names.")

    # Save Excel mapping for documentation
    mapping_df = pd.DataFrame(list(COLUMN_MAPPING.items()), columns=["Old Name", "New Name"])
    mapping_df.to_excel(MAPPING_EXCEL, index=False)
    print(f"📘 Excel mapping saved to: {os.path.abspath(MAPPING_EXCEL)}")

    # Save JSON mapping for programmatic use
    with open(MAPPING_JSON, "w", encoding="utf-8") as jf:
        json.dump(COLUMN_MAPPING, jf, ensure_ascii=False, indent=4)
    print(f"🧩 JSON mapping saved to: {os.path.abspath(MAPPING_JSON)}")

    # Create new clean table in database
    df.to_sql(NEW_TABLE, conn, if_exists="replace", index=False)
    print(f"🗃️ New table '{NEW_TABLE}' created in the database with {len(df.columns)} columns.")

    conn.close()
    print("✅ Done. The database now has a clean, standardized version of your table!")

# === ENTRY POINT ===
if __name__ == "__main__":
    main()
