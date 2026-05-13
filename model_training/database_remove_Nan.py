import sqlite3
from pathlib import Path
import pandas as pd

# ============================
# 1. CONNECT TO SQLITE DATABASE
# ============================
DB_PATH = str(Path(__file__).resolve().parent.parent / "models_directory" / "patient_feedback_ml.db")
TABLE = "patient_feedback_encoded"

conn = sqlite3.connect(DB_PATH)

print("Loading table...")
df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
print(f"Original rows: {len(df)}")

# ============================
# 2. DROP ROWS WITH MISSING LABELS
# ============================
columns_to_check = ["domain", "category", "sub_category"]

df_clean = df.dropna(subset=columns_to_check)

print(f"Cleaned rows: {len(df_clean)}")
print(f"Deleted rows: {len(df) - len(df_clean)}")

# ============================
# 3. BACKUP ORIGINAL TABLE
# ============================
backup_table = f"{TABLE}_backup_before_cleanup"
print(f"Creating backup table: {backup_table}")

df.to_sql(backup_table, conn, if_exists="replace", index=False)

# ============================
# 4. SAVE CLEANED TABLE BACK
# ============================
print(f"Writing cleaned data back into {TABLE}...")

df_clean.to_sql(TABLE, conn, if_exists="replace", index=False)

conn.close()

print("DONE — Database cleaned successfully.")
