import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split

# ======================================================
# Configuration
# ======================================================
DB_PATH = "patient_feedback_ml.db"
SOURCE_TABLE = "patient_feedback_encoded"
TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"
TEST_SIZE = 0.2
RANDOM_STATE = 42

print("\n=== Patient Feedback Dataset Split ===\n")

# ======================================================
# Connect to the database safely
# ======================================================
with sqlite3.connect(DB_PATH) as conn:

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    print(f"📥 Loading data from '{SOURCE_TABLE}' ...")
    df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE}", conn)

    if df.empty:
        raise ValueError(f"❌ Source table '{SOURCE_TABLE}' is empty!")

    print(f"✔ Total rows loaded: {len(df)}")

    # --------------------------------------------------
    # Train/Test split
    # --------------------------------------------------
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    print(f"📊 Training rows: {len(train_df)}")
    print(f"📊 Testing rows : {len(test_df)}")

    # --------------------------------------------------
    # Save new tables (replace if already exist)
    # --------------------------------------------------
    print("\n💾 Saving split tables...")

    train_df.to_sql(TRAIN_TABLE, conn, if_exists="replace", index=False)
    test_df.to_sql(TEST_TABLE, conn, if_exists="replace", index=False)

    print("\n✅ Split complete!")
    print(f"   → {TRAIN_TABLE} ({len(train_df)} rows)")
    print(f"   → {TEST_TABLE} ({len(test_df)} rows)")
