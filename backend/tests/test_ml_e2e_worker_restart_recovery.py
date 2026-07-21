"""
Stage 11 End-to-End Test — Backend Restart -> Job Recovery

ml_stage6_embedding_worker_smoke_test.py already proves the DB-layer sweep
function (sweep_stuck_jobs) correctly recovers a stuck job. This test goes
one level up: it exercises the REAL startup entrypoint
(start_worker_background_thread()) that backend/main.py actually calls on
process start — proving the sweep-then-resume path works end-to-end, not
just the sweep function in isolation.

Run from the backend/ directory:
    python -m tests.test_ml_e2e_worker_restart_recovery
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
from ml_mapping import embedding_worker


def main():
    print("=" * 70)
    print("STAGE 11 E2E — Backend Restart -> Job Recovery")
    print("=" * 70)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT TOP 1 DomainID FROM dbo.APP_LOOKUP_DOMAIN")
    domain_id = cur.fetchone()[0]
    cur.execute("SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY WHERE DomainID = ?", (domain_id,))
    row = cur.fetchone()
    category_id = row[0] if row else None
    conn.close()

    data = {
        "complaint_text": "E2E11 WORKER-RESTART complaint text",
        "feedback_received_date": "2026-07-20",
        "incident_date": "2026-07-19",
        "issuing_department_id": 1,
        "domain_id": domain_id,
        "category_id": category_id,
        "clinical_risk_type_id": 1,
        "feedback_intent_type_id": 1,
        "immediate_action": "immediate action",
        "taken_action": "",
        "patient_name": "E2E11 WORKER-RESTART Patient",
        "is_inpatient": True,
        "source_id": 1,
        "building_id": 2,
    }

    print("\n[1] Creating case, then simulating a crash mid-processing...")
    result = create_record(data, save_mode='draft')
    assert result["success"], result
    case_id = result["id"]

    # Force the job into Processing with an old StartedAt, simulating a
    # worker that crashed mid-batch before it could mark the job Completed.
    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute(
        "UPDATE ml.EmbeddingProcessingJob SET Status = 'Processing', StartedAt = ? "
        "WHERE IncidentRequestCaseID = ?",
        (datetime.now() - timedelta(minutes=30), case_id),
    )
    conn.commit()
    conn.close()
    print(f"    case={case_id}: job forced into Processing (StartedAt 30 min ago)")

    print("\n[2] Starting the REAL worker entrypoint (start_worker_background_thread)...")
    print("    This is the exact function backend/main.py calls on startup — it")
    print("    sweeps stuck jobs first, then starts polling for real.")
    try:
        embedding_worker.start_worker_background_thread(poll_interval_seconds=2)

        print("\n[3] Polling for up to 20s for the recovered job to reach Completed...")
        completed = False
        for _ in range(20):
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT Status FROM ml.EmbeddingProcessingJob WHERE IncidentRequestCaseID = ?",
                (case_id,),
            )
            status = cur.fetchone()[0]
            conn.close()
            print(f"    ... status={status}")
            if status == 'Completed':
                completed = True
                break
            time.sleep(1)

        assert completed, "Job was never recovered and completed by the restarted worker within 20s"
        print("\n    Job successfully recovered and completed by the background worker.")

    finally:
        embedding_worker.stop_worker_background_thread()
        print("    Worker thread stopped (test cleanup).")

    print("\n" + "=" * 70)
    print("ALL E2E ASSERTIONS PASSED — Backend Restart -> Job Recovery")
    print("=" * 70)

    print(f"\n[Cleanup] Removing test case {case_id}...")
    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()
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
