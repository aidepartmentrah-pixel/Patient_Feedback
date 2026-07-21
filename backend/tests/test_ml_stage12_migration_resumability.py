"""
Stage 12 Rollback/Recovery Test — Migration Resumability

Proves the exact idempotency mechanism ml_stage8_historical_migration.py
relies on (app-level already_migrated() pre-check + the DB-level filtered
unique index UQ_ml_HistoricalTrainingExample_LegacySource) survives a
simulated interrupted-then-resumed run, using the migration script's own
functions rather than reimplementing the logic.

Uses a synthetic LegacySourceTable value ('ML_STAGE12_TEST_SOURCE') that is
never one of the real SOURCE_TABLES the live migration script reads, so this
test can never collide with or duplicate real production migration data.
Simulates "crash after row 1, before row 2" by only calling
insert_historical_example() for the first row, then a "resumed run" that
re-classifies all rows through already_migrated() exactly like main() does,
confirming row 1 is correctly skipped and only row 2 gets inserted.

Run from the backend/ directory:
    python -m tests.test_ml_stage12_migration_resumability
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.database import get_connection
from scripts.ml_stage8_historical_migration import already_migrated, insert_historical_example

TEST_SOURCE_TABLE = "ML_STAGE12_TEST_SOURCE"


def _fake_row(row_identity, complaint_text):
    return {
        "id": row_identity,
        "_row_identity": row_identity,
        "complaint_text": complaint_text,
        "immediate_action": "stage12 test immediate action",
        "taken_action": "stage12 test taken action",
    }


def main():
    print("=" * 70)
    print("STAGE 12 — Migration Resumability (interrupted-then-resumed run)")
    print("=" * 70)

    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()

    row1 = _fake_row(900001, "STAGE12 TEST resumability row one unique marker aa11")
    row2 = _fake_row(900002, "STAGE12 TEST resumability row two unique marker bb22")
    batch_id = "ML_S12_TEST_INITIAL"

    try:
        print("\n[1] Simulating a 'first run' that crashes after row 1 commits...")
        assert not already_migrated(cur, TEST_SOURCE_TABLE, row1["_row_identity"])
        insert_historical_example(cur, TEST_SOURCE_TABLE, row1, "Unmatched", 0.0, None, [], batch_id)
        conn.commit()
        print(f"    Row 1 (identity={row1['_row_identity']}) committed. 'Crash' before row 2.")

        print("\n[2] Simulating the 'resumed run' — re-classifies BOTH rows through the same "
              "pre-check main() itself uses...")
        resumed_batch_id = "ML_S12_TEST_RESUMED"
        inserted_this_run = []
        for row in (row1, row2):
            if already_migrated(cur, TEST_SOURCE_TABLE, row["_row_identity"]):
                print(f"    Row identity={row['_row_identity']}: already migrated, skipping (correct).")
                continue
            insert_historical_example(cur, TEST_SOURCE_TABLE, row, "Unmatched", 0.0, None, [], resumed_batch_id)
            inserted_this_run.append(row["_row_identity"])
            print(f"    Row identity={row['_row_identity']}: inserted fresh (correct).")
        conn.commit()

        assert inserted_this_run == [row2["_row_identity"]], (
            f"Expected only row 2 to be inserted on resume, got {inserted_this_run}"
        )

        print("\n[3] Verifying final state: exactly one row per identity, no duplicates...")
        cur.execute(
            "SELECT LegacySourceRowID, COUNT(*) FROM ml.HistoricalTrainingExample "
            "WHERE LegacySourceTable = ? GROUP BY LegacySourceRowID",
            (TEST_SOURCE_TABLE,),
        )
        counts = {r[0]: r[1] for r in cur.fetchall()}
        print(f"    Row counts by identity: {counts}")
        assert counts == {row1["_row_identity"]: 1, row2["_row_identity"]: 1}, (
            "Expected exactly 1 row for each of the 2 identities, no duplicates"
        )

        print("\n[4] Confirming the DB-level unique index itself rejects a raw duplicate insert "
              "(belt-and-suspenders, independent of the app-level pre-check)...")
        try:
            insert_historical_example(cur, TEST_SOURCE_TABLE, row1, "Unmatched", 0.0, None, [], "ML_S12_TEST_FORCED")
            conn.commit()
            raise AssertionError("Expected a duplicate-key error from the unique index, but insert succeeded")
        except AssertionError:
            raise
        except Exception as e:
            conn.rollback()
            print(f"    Duplicate insert correctly rejected by the DB: {type(e).__name__}")

        print("\n" + "=" * 70)
        print("ALL E2E ASSERTIONS PASSED — Migration Resumability")
        print("=" * 70)

    finally:
        print(f"\n[Cleanup] Removing test rows for LegacySourceTable={TEST_SOURCE_TABLE}...")
        conn.rollback()
        cur.execute("DELETE FROM ml.HistoricalTrainingExample WHERE LegacySourceTable = ?", (TEST_SOURCE_TABLE,))
        conn.commit()
        conn.close()
        print("    Cleanup complete.")


if __name__ == "__main__":
    main()
