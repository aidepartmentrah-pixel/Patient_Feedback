"""
ML Architecture Consolidation — Stage 8: Legacy Training-Run History Copy

Straight 1:1 archival copy of the legacy SQLite training_runs / model_metrics
/ ml_db_size_history tables (backend/data/training_metadata.db) into the new
ml.LegacyTrainingRunHistory / ml.LegacyModelMetricHistory /
ml.LegacyDbSizeHistory tables (see phase_ml_s8_historical_migration_schema.sql).

Not a redesign — Stage 9 introduces the new forward-looking, per-run-folder
tracking design. This just preserves what already exists so it isn't lost
before SQLite is eventually retired (Stage 13).

Idempotent: skips rows already present (RunID / RecordDate are unique keys
in the destination tables; model_metrics has no natural unique key in the
source, so it's keyed here by LegacyMetricID instead and only re-copied if
missing).

Run from the backend/ directory:
    python -m scripts.ml_stage8_copy_training_history
"""

import os
import sqlite3
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.database import get_connection

TRAINING_DB_PATH = os.path.join(_REPO_ROOT, "backend", "data", "training_metadata.db")


def copy_training_runs(sqlite_cur, sql_cursor):
    sqlite_cur.execute("SELECT run_id, started_at, finished_at, status, models_trained, created_at FROM training_runs")
    rows = sqlite_cur.fetchall()
    inserted = 0
    for run_id, started_at, finished_at, status, models_trained, created_at in rows:
        sql_cursor.execute("SELECT 1 FROM ml.LegacyTrainingRunHistory WHERE RunID = ?", (run_id,))
        if sql_cursor.fetchone():
            continue
        sql_cursor.execute(
            """
            INSERT INTO ml.LegacyTrainingRunHistory
                (RunID, StartedAt, FinishedAt, Status, ModelsTrained, LegacyCreatedAt)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, started_at, finished_at, status, models_trained, created_at),
        )
        inserted += 1
    return len(rows), inserted


def copy_model_metrics(sqlite_cur, sql_cursor):
    sqlite_cur.execute(
        "SELECT id, run_id, model_name, num_records, accuracy, `precision`, recall, f1, last_trained FROM model_metrics"
    )
    rows = sqlite_cur.fetchall()
    inserted = 0
    for legacy_id, run_id, model_name, num_records, accuracy, precision, recall, f1, last_trained in rows:
        sql_cursor.execute("SELECT 1 FROM ml.LegacyModelMetricHistory WHERE LegacyMetricID = ?", (legacy_id,))
        if sql_cursor.fetchone():
            continue
        sql_cursor.execute(
            """
            INSERT INTO ml.LegacyModelMetricHistory
                (LegacyMetricID, RunID, ModelName, NumRecords, Accuracy, Precision_, Recall_, F1, LastTrained)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (legacy_id, run_id, model_name, num_records, accuracy, precision, recall, f1, last_trained),
        )
        inserted += 1
    return len(rows), inserted


def copy_db_size_history(sqlite_cur, sql_cursor):
    sqlite_cur.execute("SELECT record_date, record_count, recorded_at FROM ml_db_size_history")
    rows = sqlite_cur.fetchall()
    inserted = 0
    for record_date, record_count, recorded_at in rows:
        sql_cursor.execute("SELECT 1 FROM ml.LegacyDbSizeHistory WHERE RecordDate = ?", (record_date,))
        if sql_cursor.fetchone():
            continue
        sql_cursor.execute(
            """
            INSERT INTO ml.LegacyDbSizeHistory (RecordDate, RecordCount, LegacyRecordedAt)
            VALUES (?, ?, ?)
            """,
            (record_date, record_count, recorded_at),
        )
        inserted += 1
    return len(rows), inserted


def main():
    print("=" * 70)
    print("ML ARCHITECTURE CONSOLIDATION — STAGE 8: LEGACY TRAINING HISTORY COPY")
    print("=" * 70)

    sqlite_conn = sqlite3.connect(f"file:{TRAINING_DB_PATH}?mode=ro", uri=True)
    sqlite_cur = sqlite_conn.cursor()

    conn = get_connection()
    cursor = conn.cursor()

    source, inserted = copy_training_runs(sqlite_cur, cursor)
    print(f"\ntraining_runs: {source} source rows, {inserted} newly inserted into ml.LegacyTrainingRunHistory")

    source, inserted = copy_model_metrics(sqlite_cur, cursor)
    print(f"model_metrics: {source} source rows, {inserted} newly inserted into ml.LegacyModelMetricHistory")

    source, inserted = copy_db_size_history(sqlite_cur, cursor)
    print(f"ml_db_size_history: {source} source rows, {inserted} newly inserted into ml.LegacyDbSizeHistory")

    conn.commit()
    cursor.close()
    conn.close()
    sqlite_conn.close()

    print("\nDone. Source SQLite file untouched (read-only connection).")
    print("=" * 70)


if __name__ == "__main__":
    main()
