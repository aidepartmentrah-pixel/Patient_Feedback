"""
Stage 11 End-to-End Test — Failed Job -> Retry -> Completion

Genuinely new coverage: no existing test calls mark_job_failed() or proves
a job that failed once actually reaches Completed on a later attempt.
ml_stage6_embedding_worker_smoke_test.py only exercises the stuck-job
sweep (Processing -> RetryPending), never a real failure -> retry cycle.

Failures are simulated deterministically via mark_job_failed() directly
(not by breaking the real embedding model), and NextRetryAt is backdated
after the call so the retry is immediately eligible for claim_pending_jobs()
rather than waiting on a real timer.

Run from the backend/ directory:
    python -m tests.test_ml_e2e_job_retry_completion
"""

import os
import sys
from datetime import datetime, timedelta

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.database import get_connection
from api.services.insert_service import create_record
from api.db_layer import ml_embedding_job_db
from ml_mapping import embedding_worker


def create_test_case(marker):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT TOP 1 DomainID FROM dbo.APP_LOOKUP_DOMAIN")
    domain_id = cur.fetchone()[0]
    cur.execute("SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY WHERE DomainID = ?", (domain_id,))
    row = cur.fetchone()
    category_id = row[0] if row else None
    conn.close()

    data = {
        "complaint_text": f"E2E11 JOB-RETRY {marker} complaint text",
        "feedback_received_date": "2026-07-20",
        "incident_date": "2026-07-19",
        "issuing_department_id": 1,
        "domain_id": domain_id,
        "category_id": category_id,
        "clinical_risk_type_id": 1,
        "feedback_intent_type_id": 1,
        "immediate_action": "immediate action",
        "taken_action": "",
        "patient_name": f"E2E11 JOB-RETRY {marker} Patient",
        "is_inpatient": True,
        "source_id": 1,
        "building_id": 2,
    }
    result = create_record(data, save_mode='draft')
    assert result["success"], result
    return result["id"]


def get_job(case_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT EmbeddingProcessingJobID, Status, AttemptCount, MaximumAttempts, NextRetryAt "
        "FROM ml.EmbeddingProcessingJob WHERE IncidentRequestCaseID = ?",
        (case_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row


def backdate_next_retry(job_id):
    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute(
        "UPDATE ml.EmbeddingProcessingJob SET NextRetryAt = ? WHERE EmbeddingProcessingJobID = ?",
        (datetime.now() - timedelta(minutes=1), job_id),
    )
    conn.commit()
    conn.close()


def main():
    print("=" * 70)
    print("STAGE 11 E2E — Failed Job -> Retry -> Completion")
    print("=" * 70)

    # -----------------------------------------------------------
    # Part A: fail once, then succeed on retry
    # -----------------------------------------------------------
    print("\n[1] Creating case for the retry-then-succeed scenario...")
    case_a = create_test_case("RETRY-SUCCEED")
    job_a = get_job(case_a)
    print(f"    case={case_a}, job_id={job_a.EmbeddingProcessingJobID}, status={job_a.Status}")
    assert job_a.Status == 'Pending'

    print("\n[2] Simulating a failure via mark_job_failed()...")
    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()
    ml_embedding_job_db.mark_job_failed(
        cur, job_a.EmbeddingProcessingJobID,
        error_code="SimulatedFailure", error_message="E2E11 simulated failure",
        retry_delay_minutes=0,
    )
    conn.commit()
    conn.close()
    backdate_next_retry(job_a.EmbeddingProcessingJobID)

    job_a = get_job(case_a)
    print(f"    After failure: status={job_a.Status}, AttemptCount={job_a.AttemptCount}")
    assert job_a.Status == 'RetryPending'
    assert job_a.AttemptCount == 1

    print("\n[3] Running the real worker again — job should now succeed (nothing is actually broken)...")
    result = embedding_worker.process_pending_jobs(batch_size=10)
    print(f"    Result: {result}")

    job_a = get_job(case_a)
    print(f"    After retry: status={job_a.Status}")
    assert job_a.Status == 'Completed', f"Expected job to reach Completed after retry, got {job_a.Status}"

    # -----------------------------------------------------------
    # Part B: exhaust all retries -> terminal Failed
    # -----------------------------------------------------------
    print("\n[4] Creating case for the retries-exhausted scenario...")
    case_b = create_test_case("RETRIES-EXHAUSTED")
    job_b = get_job(case_b)
    job_id_b = job_b.EmbeddingProcessingJobID
    max_attempts = job_b.MaximumAttempts
    print(f"    case={case_b}, job_id={job_id_b}, MaximumAttempts={max_attempts}")

    for attempt in range(1, max_attempts + 1):
        conn = get_connection()
        conn.autocommit = False
        cur = conn.cursor()
        ml_embedding_job_db.mark_job_failed(
            cur, job_id_b,
            error_code="SimulatedFailure", error_message=f"E2E11 simulated failure attempt {attempt}",
            retry_delay_minutes=0,
        )
        conn.commit()
        conn.close()
        job_b = get_job(case_b)
        print(f"    After attempt {attempt}: status={job_b.Status}, AttemptCount={job_b.AttemptCount}")
        if attempt < max_attempts:
            assert job_b.Status == 'RetryPending', \
                f"Expected RetryPending before exhausting attempts, got {job_b.Status} at attempt {attempt}"
        else:
            assert job_b.Status == 'Failed', \
                f"Expected terminal Failed after {max_attempts} attempts, got {job_b.Status}"

    print("\n" + "=" * 70)
    print("ALL E2E ASSERTIONS PASSED — Failed Job -> Retry -> Completion (both paths)")
    print("=" * 70)

    print(f"\n[Cleanup] Removing test cases {case_a}, {case_b}...")
    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()
    for cid in (case_a, case_b):
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
    print("    Cleanup complete.")


if __name__ == "__main__":
    main()
