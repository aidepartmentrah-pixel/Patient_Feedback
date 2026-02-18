import sqlite3
import os
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


def split_data(db_path: str = None) -> dict:
    """
    Split the patient_feedback_encoded table into train/test tables.

    Args:
        db_path: Optional explicit path to the SQLite DB.
                 Defaults to patient_feedback_ml.db inside this directory.

    Returns:
        Dict with train_rows, test_rows counts.
    """
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_PATH)

    print("\n=== Patient Feedback Dataset Split ===\n")

    with sqlite3.connect(db_path) as conn:

        # --------------------------------------------------
        # Load dataset
        # --------------------------------------------------
        print(f"Loading data from '{SOURCE_TABLE}' ...")
        df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE}", conn)

        if df.empty:
            raise ValueError(f"Source table '{SOURCE_TABLE}' is empty!")

        print(f"Total rows loaded: {len(df)}")

        # --------------------------------------------------
        # Train/Test split
        # --------------------------------------------------
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        print(f"Training rows: {len(train_df)}")
        print(f"Testing rows : {len(test_df)}")

        # --------------------------------------------------
        # Save new tables (replace if already exist)
        # --------------------------------------------------
        print("\nSaving split tables...")

        train_df.to_sql(TRAIN_TABLE, conn, if_exists="replace", index=False)
        test_df.to_sql(TEST_TABLE, conn, if_exists="replace", index=False)

        print("\nSplit complete!")
        print(f"   -> {TRAIN_TABLE} ({len(train_df)} rows)")
        print(f"   -> {TEST_TABLE} ({len(test_df)} rows)")

    return {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "source_rows": len(df),
    }


# Allow running as standalone script
if __name__ == "__main__":
    split_data()
