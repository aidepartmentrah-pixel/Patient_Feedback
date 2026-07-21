"""
Stage 11 End-to-End Test — Bulk Import -> Embedding -> Training

test_smoke_import.py already proves bulk import registers an
ml.EmbeddingProcessingJob per imported case (the Stage 4 gate). This test
chains the next two legs that were never exercised: running the embedding
worker on those imported cases, and confirming the result is actually
training-eligible via the real training data-access query.

Reuses test_smoke_import.py's own Excel-building helpers rather than
duplicating them.

Run from the backend/ directory:
    python -m tests.test_ml_e2e_bulk_import_to_training
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.database import get_connection
from api.services.import_service import process_upload
from ml_mapping import embedding_worker
from models_directory.split_data import _fetch_sql_server_training_dataframe
from tests.test_smoke_import import fetch_test_values, build_test_excel


def main():
    print("=" * 70)
    print("STAGE 11 E2E — Bulk Import -> Embedding -> Training")
    print("=" * 70)

    print("\n[1] Fetching real lookup values and building test Excel...")
    tv = fetch_test_values()
    missing = [k for k, v in tv.items() if v is None]
    assert not missing, f"Missing lookup values: {missing}"
    excel_bytes = build_test_excel(tv)

    print("\n[2] Running process_upload()...")
    report = process_upload(excel_bytes, created_by_user_id=1)
    s = report["summary"]
    print(f"    imported_groups={s['imported_groups']}, imported_rows={s['imported_rows']}")
    assert s["imported_rows"] >= 1, "Expected at least 1 imported row to test against"

    # report["imported"][*]["incident_id"] is the PARENT incident id, not the
    # per-row case id — a group can have multiple case rows under one
    # incident. Find the actual per-row IncidentRequestCaseID values the
    # same way test_smoke_import.py's own ML-gate check does: via
    # ml.ImportSourceRecordMap, keyed by this upload's import_batch_id.
    import_batch_id = s["import_batch_id"]
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT IncidentRequestCaseID FROM ml.ImportSourceRecordMap WHERE ImportBatchID = ?",
        (import_batch_id,),
    )
    imported_case_ids = [r[0] for r in cur.fetchall()]
    conn.close()
    print(f"    Imported case IDs: {imported_case_ids}")
    assert len(imported_case_ids) == s["imported_rows"]

    conn = get_connection()
    cur = conn.cursor()
    placeholders = ",".join("?" * len(imported_case_ids))
    cur.execute(
        f"SELECT IncidentRequestCaseID, Status, JobType FROM ml.EmbeddingProcessingJob "
        f"WHERE IncidentRequestCaseID IN ({placeholders})",
        imported_case_ids,
    )
    jobs = cur.fetchall()
    conn.close()
    print(f"\n[3] Jobs registered: {[(j.IncidentRequestCaseID, j.Status, j.JobType) for j in jobs]}")
    assert len(jobs) == len(imported_case_ids)
    assert all(j.Status == 'Pending' and j.JobType == 'Create' for j in jobs)

    print("\n[4] Running embedding_worker.process_pending_jobs()...")
    result = embedding_worker.process_pending_jobs(batch_size=10)
    print(f"    Result: {result}")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        f"SELECT IncidentRequestCaseID, ProcessingStatus, ComplaintEmbedding FROM ml.CaseTrainingRecord "
        f"WHERE IncidentRequestCaseID IN ({placeholders})",
        imported_case_ids,
    )
    recs = {r.IncidentRequestCaseID: r for r in cur.fetchall()}
    conn.close()
    for cid in imported_case_ids:
        rec = recs.get(cid)
        print(f"    case={cid}: status={rec.ProcessingStatus if rec else None}, "
              f"embedding_present={rec.ComplaintEmbedding is not None if rec else False}")
        assert rec is not None and rec.ProcessingStatus == 'Completed'
        assert rec.ComplaintEmbedding is not None

    print("\n[5] Confirming imported cases are training-eligible via the REAL training query...")
    df = _fetch_sql_server_training_dataframe()
    imported_texts = {g["group_key"]: True for g in report["imported"]}
    # Match by the known marker text used in the test Excel's valid group (IMP-SMOKE-001)
    matches = df[df["complaint_text"].str.contains("row 1 of group 001", na=False)]
    print(f"    Rows in training dataframe matching imported group: {len(matches)}")
    assert len(matches) >= 1

    print("\n" + "=" * 70)
    print("ALL E2E ASSERTIONS PASSED — Bulk Import -> Embedding -> Training")
    print("=" * 70)

    print(f"\n[Cleanup] Removing imported cases {imported_case_ids}...")
    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()

    # Multiple case rows can share one parent incident_id (one group -> one
    # incident, N case rows) — delete every case first, then only delete an
    # incident once none of its case rows remain, to avoid an FK conflict.
    incident_ids = set()
    for cid in imported_case_ids:
        cur.execute("SELECT incident_id FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (cid,))
        inc_row = cur.fetchone()
        if inc_row and inc_row[0]:
            incident_ids.add(inc_row[0])
        cur.execute("DELETE FROM ml.EmbeddingProcessingJob WHERE IncidentRequestCaseID = ?", (cid,))
        cur.execute("DELETE FROM ml.CaseTrainingRecord WHERE IncidentRequestCaseID = ?", (cid,))
        cur.execute("DELETE FROM ml.ImportSourceRecordMap WHERE IncidentRequestCaseID = ?", (cid,))
        cur.execute("DELETE FROM dbo.APP_IncidentCaseTargetDepartment WHERE IncidentRequestCaseID = ?", (cid,))
        cur.execute("DELETE FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (cid,))

    for iid in incident_ids:
        cur.execute("SELECT COUNT(*) FROM dbo.APP_IncidentCase WHERE incident_id = ?", (iid,))
        if cur.fetchone()[0] == 0:
            cur.execute("DELETE FROM dbo.APP_Incident WHERE incident_id = ?", (iid,))

    cur.execute("DELETE FROM ml.ImportBatch WHERE ImportBatchID = ?", (report["summary"]["import_batch_id"],))
    cur.execute("DELETE FROM APP_RESERVE_PATIENT WHERE FullName LIKE 'SMOKE TEST%'")
    conn.commit()
    conn.close()
    print("    Cleanup complete.")


if __name__ == "__main__":
    main()
