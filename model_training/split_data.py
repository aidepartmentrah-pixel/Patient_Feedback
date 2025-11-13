import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split

# === 1. Database and table configuration ===
DB_PATH = "patient_feedback_ml.db"   # path to your database
SOURCE_TABLE = "patient_feedback_encoded"
TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"

# === 2. Connect to SQLite database ===
conn = sqlite3.connect(DB_PATH)

# === 3. Load full dataset ===
print(f"Loading data from {SOURCE_TABLE} ...")
df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE}", conn)
print(f"Total rows loaded: {len(df)}")

# === 4. Split data ===
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
print(f"Training rows: {len(train_df)} | Testing rows: {len(test_df)}")

# === 5. Save new tables ===
train_df.to_sql(TRAIN_TABLE, conn, if_exists="replace", index=False)
test_df.to_sql(TEST_TABLE, conn, if_exists="replace", index=False)

# === 6. Confirm completion ===
print(f"✅ Split complete! New tables saved in {DB_PATH}:")
print(f"  - {TRAIN_TABLE} ({len(train_df)} rows)")
print(f"  - {TEST_TABLE} ({len(test_df)} rows)")

conn.close()
