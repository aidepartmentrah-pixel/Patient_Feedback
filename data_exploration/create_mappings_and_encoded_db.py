"""
create_mappings_and_encoded_db.py (final, fixed)

- Reads:  patient_feedback_clean from ../data_exploration/patient_feedback.db
- Writes: categorical_value_mapping.xlsx / .json into ../data_exploration/
- Creates: patient_feedback_encoded table (excluding feedback_received_date from encoding)
"""

import os
import json
import pandas as pd
import sqlite3

# === CONFIG ===
DATA_DIR = "../data_exploration"
DB_PATH = os.path.join(DATA_DIR, "patient_feedback.db")
SOURCE_TABLE = "patient_feedback_clean"
ENCODED_TABLE = "patient_feedback_encoded"
EXCEL_MAPPING = os.path.join(DATA_DIR, "categorical_value_mapping.xlsx")
JSON_MAPPING = os.path.join(DATA_DIR, "categorical_value_mapping.json")

# === COLUMNS TO EXCLUDE FROM ENCODING ===
TEXT_COLUMNS = {
    "patient_full_name",
    "complaint_text",
    "immediate_action",
    "taken_action",
    "embedding_text1",
    "embedding_text2",
    "embedding_text3",
    "embedding_text123",
    "embedding_text23",
    "embedding_model",
    "feedback_received_date",  # ✅ explicitly exclude date column
}

def main():
    # Verify data directory and DB exist
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"❌ Database not found at {os.path.abspath(DB_PATH)}")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE}", conn)
    print(f"✅ Loaded '{SOURCE_TABLE}' with {len(df)} rows and {len(df.columns)} columns.\n")

    # === Detect categorical columns ===
    categorical_cols = []
    for col in df.columns:
        if col in TEXT_COLUMNS:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_datetime64_dtype(df[col]):
            continue
        if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
            continue
        if df[col].dtype == "object":
            categorical_cols.append(col)

    print("📊 Columns selected for encoding:")
    print(", ".join(categorical_cols) if categorical_cols else "None found.\n")

    # === Build categorical mappings ===
    mapping_dict = {}
    for col in categorical_cols:
        unique_vals = df[col].dropna().astype(str).unique().tolist()
        unique_vals = sorted(unique_vals, key=lambda s: s.lower())  # deterministic ordering
        mapping = {val: i + 1 for i, val in enumerate(unique_vals)}
        mapping_dict[col] = mapping
        df[col] = df[col].map(mapping)

    # === Save Excel mapping for documentation ===
    rows = []
    for col, mapping in mapping_dict.items():
        for orig, code in mapping.items():
            rows.append({"Column": col, "Original Value": orig, "Encoded Value": code})
    mapping_df = pd.DataFrame(rows or [{"Column": "", "Original Value": "", "Encoded Value": ""}])
    mapping_df.to_excel(EXCEL_MAPPING, index=False)
    print(f"📘 Excel mapping saved to: {os.path.abspath(EXCEL_MAPPING)}")

    # === Save JSON mapping for programmatic decoding ===
    with open(JSON_MAPPING, "w", encoding="utf-8") as jf:
        json.dump(mapping_dict, jf, ensure_ascii=False, indent=4)
    print(f"🧩 JSON mapping saved to: {os.path.abspath(JSON_MAPPING)}")

    # === Create encoded table ===
    df.to_sql(ENCODED_TABLE, conn, if_exists="replace", index=False)
    conn.close()
    print(f"🗃️ Encoded table '{ENCODED_TABLE}' created successfully.")
    print("✅ Finished. Encoded data and mappings are ready for ML development.\n")

if __name__ == "__main__":
    main()
