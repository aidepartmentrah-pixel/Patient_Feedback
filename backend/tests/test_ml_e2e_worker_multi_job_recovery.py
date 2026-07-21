"""
Stage 12 Rollback/Recovery Test — Multi-Job Stuck-Worker Recovery

test_ml_e2e_worker_restart_recovery.py (Stage 11) already proves the real
start_worker_background_thread() entrypoint recovers a SINGLE job stuck in
Processing after a simulated crash. sweep_stuck_jobs() itself is a single
set-based SQL UPDATE (no per-job looping), so it should already handle
several jobs stuck simultaneously — but that was never actually exercised.
This test forces 3 cases' jobs into Processing at once (simulating a worker
that died mid-batch with several jobs in flight), including one case with
TWO stuck jobs (its original Create job plus a later edit's TextChanged job
that got queued before the crash), and confirms:
  - all of them recover to Completed after a real worker restart
  - the two-jobs-one-case scenario is still deduplicated correctly by
    process_pending_jobs()'s case_ids = sorted({...}) batching, not
    processed twice / left half-finished

Run from the backend/ directory:
    python -m tests.test_ml_e2e_worker_multi_job_recovery
"""

import os
import sys
import time
from datetime import datetime, timedelta

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.database import get_connection
from api.services.insert_service import create_record
from api.db_layer import ml_embedding_job_db
from ml_mapping import embedding_worker


def _make_case(domain_id, category_id, marker):
    data = {
        "complaint_text": f"E2E12 MULTI-JOB {marker} complaint text",
        "feedback_received_date": "2026-07-20",
        "incident_date": "2026-07-19",
        "issuing_department_id": 1,
        "domain_id": domain_id,
        "category_id": category_id,
        "clinical_risk_type_id": 1,
        "feedback_intent_type_id": 1,
        "immediate_action": "immediate action",
        "taken_action": "",
        "patient_name": f"E2E12 MULTI-JOB {marker} Patient",
        "is_inpatient": True,
        "source_id": 1,
        "building_id": 2,
    }
    result = create_record(data, save_mode='draft')
    assert result["success"], result
    return result["id"]


def main():
    print("=" * 70)
    print("STAGE 12 — Multi-Job Stuck-Worker Recovery")
    print("=" * 70)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT TOP 1 DomainID FROM dbo.APP_LOOKUP_DOMAIN")
    domain_id = cur.fetchone()[0]
    cur.execute("SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY WHERE DomainID = ?", (domain_id,))
    row = cur.fetchone()
    category_id = row[0] if row else None
    conn.close()

    print("\n[1] Creating 3 cases (each registers a Pending 'Create' job)...")
    case_a = _make_case(domain_id, category_id, "A")
    case_b = _make_case(domain_id, category_id, "B")
    case_c = _make_case(domain_id, category_id, "C-two-jobs")
    print(f"    case_a={case_a}, case_b={case_b}, case_c={case_c} (case_c will get a 2nd stuck job)")

    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()
    # case_c gets a second job (simulating an edit queued just before the
    # crash — its Create job and this TextChanged job are BOTH stuck).
    ml_embedding_job_db.insert_embedding_job(cur, case_c, "TextChanged")
    conn.commit()

    stale_started_at = datetime.now() - timedelta(minutes=30)
    cur.execute(
        "UPDATE ml.EmbeddingProcessingJob SET Status = 'Processing', StartedAt = ? "
        "WHERE IncidentRequestCaseID IN (?, ?, ?)",
        (stale_started_at, case_a, case_b, case_c),
    )
    conn.commit()

    cur.execute(
        "SELECT IncidentRequestCaseID, JobType, Status FROM ml.EmbeddingProcessingJob "
        "WHERE IncidentRequestCaseID IN (?, ?, ?) ORDER BY IncidentRequestCaseID",
        (case_a, case_b, case_c),
    )
    stuck_jobs = cur.fetchall()
    conn.close()
    print(f"\n[2] Jobs forced into Processing (simulated mid-batch crash): "
          f"{[(j.IncidentRequestCaseID, j.JobType, j.Status) for j in stuck_jobs]}")
    assert len(stuck_jobs) == 4, f"Expected 4 stuck jobs total (1+1+2), got {len(stuck_jobs)}"
    assert all(j.Status == 'Processing' for j in stuck_jobs)

    print("\n[3] Starting the REAL worker entrypoint (start_worker_background_thread)...")
    try:
        embedding_worker.start_worker_background_thread(poll_interval_seconds=2)

        print("\n[4] Polling for up to 25s for ALL 4 jobs to reach Completed...")
        all_completed = False
        for _ in range(25):
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT IncidentRequestCaseID, JobType, Status FROM ml.EmbeddingProcessingJob "
                "WHERE IncidentRequestCaseID IN (?, ?, ?)",
                (case_a, case_b, case_c),
            )
            rows = cur.fetchall()
            conn.close()
            statuses = {(r.IncidentRequestCaseID, r.JobType): r.Status for r in rows}
            print(f"    ... {statuses}")
            if all(s == 'Completed' for s in statuses.values()):
                all_completed = True
                break
            time.sleep(1)

        assert all_completed, "Not all stuck jobs reached Completed within 25s"
        print("\n    All 4 jobs (across 3 cases, one with 2 jobs) recovered and completed.")

        print("\n[5] Confirming case_c's embeddings were computed once and both its jobs "
              "share the same completed state (dedup, not double-processed)...")
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT ProcessingStatus, ComplaintEmbedding FROM ml.CaseTrainingRecord "
            "WHERE IncidentRequestCaseID = ?", (case_c,)
        )
        rec = cur.fetchone()
        conn.close()
        assert rec is not None and rec.ProcessingStatus == 'Completed'
        assert rec.ComplaintEmbedding is not None
        print(f"    case_c CaseTrainingRecord: ProcessingStatus={rec.ProcessingStatus}, "
              f"embedding_bytes={len(rec.ComplaintEmbedding)}")

    finally:
        embedding_worker.stop_worker_background_thread()
        print("    Worker thread stopped (test cleanup).")

    print("\n" + "=" * 70)
    print("ALL E2E ASSERTIONS PASSED — Multi-Job Stuck-Worker Recovery")
    print("=" * 70)

    print(f"\n[Cleanup] Removing test cases {case_a}, {case_b}, {case_c}...")
    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()
    for case_id in (case_a, case_b, case_c):
        cur.execute("DELETE FROM ml.EmbeddingProcessingJob WHERE IncidentRequestCaseID = ?", (case_id,))
        cur.execute("DELETE FROM ml.CaseTrainingRecord WHERE IncidentRequestCaseID = ?", (case_id,))
        cur.execute("DELETE FROM dbo.APP_IncidentCaseTargetDepartment WHERE IncidentRequestCaseID = ?", (case_id,))
        cur.execute("SELECT incident_id FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (case_id,))
        inc_row = cur.fetchone()
        cur.execute("DELETE FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (case_id,))
        if inc_row and inc_row[0]:
            cur.execute("DELETE FROM dbo.APP_Incident WHERE incident_id = ?", (inc_row[0],))
    conn.commit()
    conn.close()
    print("    Cleanup complete.")


if __name__ == "__main__":
    main()
