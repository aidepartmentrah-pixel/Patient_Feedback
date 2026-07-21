import json
import sqlite3
import os
from datetime import datetime
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

# ml.CaseTrainingRecord (SQL Server) column -> legacy SQLite
# patient_feedback_encoded column name. Every train_*.py script reads
# table_feedback_train/test by these legacy names, so aliasing here means
# NONE of those 18 scripts need to change for Stage 7 (see
# ML_ARCHITECTURE_DECISION_RECORD.md, Workstream 1 Stage 7).
_SQL_SERVER_COLUMN_ALIAS_MAP = {
    "CaseTrainingRecordID": "id",
    "ComplaintText": "complaint_text",
    "ImmediateActionText": "immediate_action",
    "TakenActionText": "taken_action",
    "FeedbackTypeID": "feedback_type",
    "DomainID": "domain",
    "CategoryID": "category",
    "SubCategoryID": "sub_category",
    "ClassificationID": "classification_en",
    "SeverityLevelID": "severity_level",
    "StageID": "stage",
    "HarmLevelID": "harm_level",
    "ImprovementOpportunityTypeID": "improvement_opportunity_type",
    "ComplaintEmbedding": "embedding_text1",
    "CombinedTextEmbedding": "embedding_text123",
}

# Same target column names as above, sourced from ml.HistoricalTrainingExample
# instead (its embedding columns are named EmbeddingText1/EmbeddingText123
# rather than ComplaintEmbedding/CombinedTextEmbedding, but hold the same
# raw float32 bytes format).
_HISTORICAL_COLUMN_ALIAS_MAP = {
    "HistoricalTrainingExampleID": "id",
    "ComplaintText": "complaint_text",
    "ImmediateActionText": "immediate_action",
    "TakenActionText": "taken_action",
    "FeedbackTypeID": "feedback_type",
    "DomainID": "domain",
    "CategoryID": "category",
    "SubCategoryID": "sub_category",
    "ClassificationID": "classification_en",
    "SeverityLevelID": "severity_level",
    "StageID": "stage",
    "HarmLevelID": "harm_level",
    "ImprovementOpportunityTypeID": "improvement_opportunity_type",
    "EmbeddingText1": "embedding_text1",
    "EmbeddingText123": "embedding_text123",
}


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


def _fetch_sql_server_training_dataframe() -> pd.DataFrame:
    """
    Pull embeddings + labels from SQL Server, from two sources:

    1. ml.CaseTrainingRecord — cases linked to a live operational case,
       only rows the Stage 6 worker has actually finished processing
       (ProcessingStatus='Completed'), so both embedding columns are
       guaranteed populated.
    2. ml.HistoricalTrainingExample, filtered to LegacySourceTable =
       'patient_feedback_encoded' ONLY — patient_feedback_encoded_Old is
       deliberately excluded per explicit user decision (2026-07-17): it's
       a separate legacy snapshot the user does not want folded into
       training. Also EXCLUDES rows with LinkConfidence IN ('Exact','High'):
       those cases already have a row in ml.CaseTrainingRecord with
       fresher, live-model-computed embeddings, and including both would
       silently duplicate the same case in the training set.

    patient_feedback_encoded and patient_feedback_encoded_Old were confirmed
    to hold real, verbatim-duplicate complaints (~460 text groups, same
    complaint preserved in both during an earlier partial migration) — one
    more reason _Old is excluded here rather than merged in. Even scoped to
    patient_feedback_encoded alone, the same complaint could in principle
    land in both the train and test split, silently
    inflating reported test accuracy. ROW_NUMBER()-per-ComplaintText below
    keeps exactly one copy of each distinct complaint.

    Both are aliased to the same legacy SQLite column names via
    _SQL_SERVER_COLUMN_ALIAS_MAP / _HISTORICAL_COLUMN_ALIAS_MAP so every
    existing train_*.py script keeps working unchanged.
    """
    from backend.core.database import get_connection

    current_select = ", ".join(f"{sql_col} AS {legacy_col}" for sql_col, legacy_col in _SQL_SERVER_COLUMN_ALIAS_MAP.items())
    historical_select = ", ".join(f"{sql_col} AS {legacy_col}" for sql_col, legacy_col in _HISTORICAL_COLUMN_ALIAS_MAP.items())
    output_cols = ", ".join(_SQL_SERVER_COLUMN_ALIAS_MAP.values())

    query = f"""
        WITH combined AS (
            SELECT {current_select} FROM ml.CaseTrainingRecord WHERE ProcessingStatus = 'Completed'
            UNION ALL
            SELECT {historical_select} FROM ml.HistoricalTrainingExample
            WHERE LegacySourceTable = 'patient_feedback_encoded'
              AND LinkConfidence NOT IN ('Exact', 'High')
              AND EmbeddingText1 IS NOT NULL AND EmbeddingText123 IS NOT NULL
        ),
        deduped AS (
            SELECT {output_cols},
                   ROW_NUMBER() OVER (PARTITION BY complaint_text ORDER BY id) AS rn
            FROM combined
        )
        SELECT {output_cols} FROM deduped WHERE rn = 1
    """

    conn = get_connection()
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df


def split_data_from_sql_server(sqlite_db_path: str = None) -> dict:
    """
    Stage 7 of the ML architecture consolidation (see
    ML_ARCHITECTURE_DECISION_RECORD.md): pull training data from SQL
    Server's ml.CaseTrainingRecord instead of the legacy SQLite
    patient_feedback_encoded table, proving the new store is trainable
    before the Stage 8 historical-migration cutover.

    This does NOT change where training scripts read from — the resulting
    split is still materialized into the same SQLite table_feedback_train/
    table_feedback_test tables every train_*.py script already expects (via
    Helper_Functions.load_table()), so none of those scripts need to change.

    Note: as of Stage 7, ml.CaseTrainingRecord only contains cases created/
    edited since Stage 3-6 landed — real historical volume doesn't exist
    there until Stage 8 migrates it. A meaningful side-by-side accuracy
    comparison against the SQLite-sourced training run is deferred until
    then; this function's job right now is to prove the mechanism (data
    shape, embedding parseability, end-to-end trainability) works correctly.
    """
    if sqlite_db_path is None:
        sqlite_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_PATH)

    print("\n=== Patient Feedback Dataset Split (source: SQL Server ml.CaseTrainingRecord) ===\n")

    df = _fetch_sql_server_training_dataframe()
    if df.empty:
        raise ValueError(
            "ml.CaseTrainingRecord has no Completed rows yet — nothing to train on. "
            "This is expected before Stage 8's historical migration populates it with volume."
        )

    print(f"Total rows loaded from SQL Server: {len(df)}")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows : {len(test_df)}")

    with sqlite3.connect(sqlite_db_path) as conn:
        train_df.to_sql(TRAIN_TABLE, conn, if_exists="replace", index=False)
        test_df.to_sql(TEST_TABLE, conn, if_exists="replace", index=False)

    print("\nSplit complete (materialized into SQLite table_feedback_train/test)!")
    print(f"   -> {TRAIN_TABLE} ({len(train_df)} rows)")
    print(f"   -> {TEST_TABLE} ({len(test_df)} rows)")

    metadata = {
        "method": "train_test_split",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "source": "sql_server (ml.CaseTrainingRecord, ProcessingStatus='Completed')",
        "source_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    metadata_path = os.path.join(os.path.dirname(sqlite_db_path), "split_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"   -> split method/seed recorded in {metadata_path}")

    return {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "source_rows": len(df),
        "source": "sql_server",
    }


# Allow running as standalone script
if __name__ == "__main__":
    split_data()
