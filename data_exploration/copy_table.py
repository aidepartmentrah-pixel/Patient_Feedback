"""
export_encoded_table_for_training.py

Copies only the 'patient_feedback_encoded' table
from the exploration database into a clean ML training database.
"""

import os
import sqlite3
import pandas as pd

# === CONFIG ===
SOURCE_DB = os.path.join( "patient_feedback.db")
DEST_DB = os.path.join("model_training", "patient_feedback_ml.db")
TABLE_NAME = "patient_feedback_encoded"

def main():
    if not os.path.exists(SOURCE_DB):
        raise FileNotFoundError(f"Source DB not found: {SOURCE_DB}")
    os.makedirs("model_training", exist_ok=True)

    # Connect to source DB
    conn_src = sqlite3.connect(SOURCE_DB)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn_src)
    conn_src.close()
    print(f"✅ Loaded {len(df)} rows from '{TABLE_NAME}'.")

    # Write to new DB
    conn_dest = sqlite3.connect(DEST_DB)
    df.to_sql(TABLE_NAME, conn_dest, if_exists="replace", index=False)
    conn_dest.close()

    print(f"📦 Copied '{TABLE_NAME}' to new DB at: {os.path.abspath(DEST_DB)}")

if __name__ == "__main__":
    main()
