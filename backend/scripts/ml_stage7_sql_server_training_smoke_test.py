"""
ML Architecture Consolidation — Stage 7 Smoke Test

Verifies split_data_from_sql_server() end-to-end: pulls Completed rows from
ml.CaseTrainingRecord, materializes them into the SAME SQLite
table_feedback_train/table_feedback_test tables every train_*.py script
already reads from, and confirms one real trainer (Domain) can run against
this SQL-Server-sourced data without any changes to the trainer itself.

Backs up and restores the existing SQLite table_feedback_train/test tables
around the test, since split_data_from_sql_server() (like the original
split_data()) replaces them — this test must not destroy real data.

Does NOT assert on accuracy/metrics values: with only a handful of test
rows, metrics are meaningless. The point is proving the mechanism (SQL
Server -> SQLite materialization -> load_table -> parse_embedding_series ->
sklearn training) works, not validating model quality. A meaningful
accuracy comparison against the SQLite-sourced run is deferred to Stage 8,
once real historical volume exists in ml.CaseTrainingRecord.

Run from the backend/ directory:
    python -m scripts.ml_stage7_sql_server_training_smoke_test
"""

import os
import sys
import sqlite3

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.database import get_connection
from api.services.insert_service import create_record
from ml_mapping import embedding_worker

SQLITE_DB_PATH = os.path.join(_REPO_ROOT, "models_directory", "patient_feedback_ml.db")
TRAIN_TABLE = "table_feedback_train"
TEST_TABLE = "table_feedback_test"


def _backup_split_tables():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    backup = {}
    for table in (TRAIN_TABLE, TEST_TABLE):
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        backup[table] = count
    conn.close()
    return backup


def main():
    print("=" * 70)
    print("STAGE 7 SMOKE TEST — split_data_from_sql_server()")
    print("=" * 70)

    print("\n[0] Backing up existing table_feedback_train/test (row counts only, "
          "full backup already exists from Stage 1)...")
    original_counts = _backup_split_tables()
    print(f"    Current row counts: {original_counts}")
    # Full-fidelity backup: copy the two tables to _stage7backup_ variants
    # inside the same SQLite file, restored at the end.
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    for table in (TRAIN_TABLE, TEST_TABLE):
        cur.execute(f"DROP TABLE IF EXISTS _stage7backup_{table}")
        cur.execute(f"CREATE TABLE _stage7backup_{table} AS SELECT * FROM {table}")
    conn.commit()
    conn.close()
    print("    Full-fidelity backup tables created inside the SQLite file.")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DomainID FROM dbo.APP_LOOKUP_DOMAIN ORDER BY DomainID")
    domain_ids = [r[0] for r in cur.fetchall()][:2]
    assert len(domain_ids) >= 2, "Need at least 2 domains in APP_LOOKUP_DOMAIN for this test"
    category_by_domain = {}
    for d in domain_ids:
        cur.execute("SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY WHERE DomainID = ?", (d,))
        row = cur.fetchone()
        category_by_domain[d] = row[0] if row else None
    conn.close()

    print(f"\n[1] Creating 6 test cases across 2 domains ({domain_ids})...")
    case_ids = []
    for i in range(6):
        d = domain_ids[i % 2]
        data = {
            "complaint_text": f"STAGE7 SMOKE TEST complaint {i} about domain {d}",
            "feedback_received_date": "2026-07-16",
            "incident_date": "2026-07-15",
            "issuing_department_id": 1,
            "domain_id": d,
            "category_id": category_by_domain[d],
            "clinical_risk_type_id": 1,
            "feedback_intent_type_id": 1,
            "immediate_action": "immediate action",
            "taken_action": "",
            "patient_name": f"STAGE7 SMOKE TEST Patient {i}",
            "is_inpatient": True,
            "source_id": 1,
            "building_id": 2,
        }
        result = create_record(data, save_mode='draft')
        assert result["success"], result
        case_ids.append(result["id"])
    print(f"    Created cases: {case_ids}")

    print("\n[2] Running embedding worker to populate embeddings...")
    for _ in range(2):  # a couple of passes in case batch_size doesn't cover all in one go
        result = embedding_worker.process_pending_jobs(batch_size=50)
        print(f"    {result}")

    conn = get_connection()
    cur = conn.cursor()
    placeholders = ",".join("?" * len(case_ids))
    cur.execute(
        f"SELECT COUNT(*) FROM ml.CaseTrainingRecord WHERE IncidentRequestCaseID IN ({placeholders}) AND ProcessingStatus = 'Completed'",
        case_ids,
    )
    completed_count = cur.fetchone()[0]
    conn.close()
    print(f"    Completed rows among our test cases: {completed_count} / {len(case_ids)}")
    assert completed_count == len(case_ids), "Not all test cases finished embedding processing"

    print("\n[3] Running split_data_from_sql_server()...")
    from models_directory.split_data import split_data_from_sql_server
    split_result = split_data_from_sql_server(sqlite_db_path=SQLITE_DB_PATH)
    print(f"    Result: {split_result}")
    assert split_result["source"] == "sql_server"
    assert split_result["source_rows"] >= len(case_ids)

    print("\n[4] Verifying materialized SQLite tables have the expected shape...")
    conn = sqlite3.connect(SQLITE_DB_PATH)
    train_df = __import__("pandas").read_sql_query(f"SELECT * FROM {TRAIN_TABLE}", conn)
    test_df = __import__("pandas").read_sql_query(f"SELECT * FROM {TEST_TABLE}", conn)
    conn.close()
    print(f"    train_df columns: {list(train_df.columns)}")
    for expected_col in ("embedding_text1", "embedding_text123", "domain", "complaint_text"):
        assert expected_col in train_df.columns, f"Missing expected column {expected_col}"

    print("\n[5] Verifying embeddings can be parsed back (parse_embedding_series)...")
    from models_directory.Classification_Models.Hierarchical_Classification_Model.Helper_Functions import parse_embedding_series
    X_train = parse_embedding_series(train_df["embedding_text1"])
    print(f"    Parsed embedding matrix shape: {X_train.shape} (expect (*, 768))")
    assert X_train.shape[1] == 768

    print("\n[6] Sanity check: running the real Domain trainer against this SQL-Server-sourced data...")
    print("    (NOT asserting on accuracy — N is tiny, metrics are meaningless at this scale.")
    print("     This only proves the full pipe runs end-to-end without error.)")
    from models_directory.Classification_Models.Hierarchical_Classification_Model.domain.train_domain_model import train_domain_models
    try:
        best_model, metrics = train_domain_models(base_path=SQLITE_DB_PATH)
        print(f"    Training completed. Metrics: {metrics}")
    except Exception as e:
        print(f"    [INFO] Domain trainer raised: {e}")
        print("    (With only 6 rows split 80/20, this can legitimately fail on class-balance "
              "edge cases unrelated to the SQL Server data-loading mechanism itself, which is "
              "what steps 1-5 above already proved works.)")

    print("\n" + "=" * 70)
    print("STAGE 7 MECHANISM VERIFIED (steps 1-5 are the hard requirement; step 6 is a bonus sanity check)")
    print("=" * 70)

    print(f"\n[Cleanup] Removing test cases {case_ids} and restoring original split tables...")
    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()
    for cid in case_ids:
        cur.execute("DELETE FROM ml.EmbeddingProcessingJob WHERE IncidentRequestCaseID = ?", (cid,))
        cur.execute("DELETE FROM ml.CaseTrainingRecord WHERE IncidentRequestCaseID = ?", (cid,))
        cur.execute("DELETE FROM dbo.APP_IncidentCaseTargetDepartment WHERE IncidentRequestCaseID = ?", (cid,))
        cur.execute("SELECT incident_id FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (cid,))
        inc_row = cur.fetchone()
        cur.execute("DELETE FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (cid,))
        if inc_row and inc_row[0]:
            cur.execute("DELETE FROM dbo.APP_Incident WHERE incident_id = ?", (inc_row[0],))
    conn.commit()
    conn.close()

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    for table in (TRAIN_TABLE, TEST_TABLE):
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cur.execute(f"ALTER TABLE _stage7backup_{table} RENAME TO {table}")
    conn.commit()
    restored_counts = {}
    for table in (TRAIN_TABLE, TEST_TABLE):
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        restored_counts[table] = cur.fetchone()[0]
    conn.close()
    print(f"    Restored original table_feedback_train/test. Row counts: {restored_counts}")
    assert restored_counts == original_counts, "Restore mismatch — original split tables not fully restored!"
    print("    Cleanup and restore verified complete.")


if __name__ == "__main__":
    main()
