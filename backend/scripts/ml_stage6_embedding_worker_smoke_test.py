"""
ML Architecture Consolidation — Stage 6 Smoke Test

Verifies the embedding worker: claims Pending jobs, batch-generates
embeddings for the two confirmed load-bearing targets (embedding_text1 /
embedding_text123), writes them into ml.CaseTrainingRecord, marks jobs
Completed — and separately, that the stuck-job startup sweep correctly
recovers a job left in 'Processing' from a simulated crash.

Run from the backend/ directory:
    python -m scripts.ml_stage6_embedding_worker_smoke_test
"""

import os
import sys
from datetime import datetime, timedelta

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.database import get_connection
from api.services.insert_service import create_record
from ml_mapping import embedding_worker


def main():
    print("=" * 70)
    print("STAGE 6 SMOKE TEST — embedding worker")
    print("=" * 70)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT TOP 1 DomainID FROM dbo.APP_LOOKUP_DOMAIN")
    domain_id = cur.fetchone()[0]
    cur.execute("SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY WHERE DomainID = ?", (domain_id,))
    row = cur.fetchone()
    category_id = row[0] if row else None
    conn.close()

    # -----------------------------------------------------------
    # Create two test cases so the worker has more than one job to
    # batch together in a single call.
    # -----------------------------------------------------------
    case_ids = []
    for i in range(2):
        data = {
            "complaint_text": f"STAGE6 SMOKE TEST complaint number {i}",
            "feedback_received_date": "2026-07-16",
            "incident_date": "2026-07-15",
            "issuing_department_id": 1,
            "domain_id": domain_id,
            "category_id": category_id,
            "clinical_risk_type_id": 1,
            "feedback_intent_type_id": 1,
            "immediate_action": f"immediate action {i}",
            "taken_action": "",
            "patient_name": f"STAGE6 SMOKE TEST Patient {i}",
            "is_inpatient": True,
            "source_id": 1,
            "building_id": 2,
        }
        result = create_record(data, save_mode='draft')
        assert result["success"], result
        case_ids.append(result["id"])
    print(f"\n[1] Created test cases: {case_ids}")

    def get_jobs_and_records():
        conn = get_connection()
        cur = conn.cursor()
        out = {}
        for cid in case_ids:
            cur.execute(
                "SELECT Status FROM ml.EmbeddingProcessingJob WHERE IncidentRequestCaseID = ?", (cid,)
            )
            job_statuses = [r[0] for r in cur.fetchall()]
            cur.execute(
                "SELECT ComplaintEmbedding, CombinedTextEmbedding, EmbeddingDimension, ProcessingStatus, EmbeddingModelVersionID "
                "FROM ml.CaseTrainingRecord WHERE IncidentRequestCaseID = ?", (cid,)
            )
            rec = cur.fetchone()
            out[cid] = {"job_statuses": job_statuses, "record": rec}
        conn.close()
        return out

    before = get_jobs_and_records()
    print(f"\n[2] Before worker run:")
    for cid, info in before.items():
        print(f"    case={cid}: jobs={info['job_statuses']}, "
              f"embedding_present={info['record'].ComplaintEmbedding is not None if info['record'] else None}, "
              f"status={info['record'].ProcessingStatus if info['record'] else None}")
        assert info["job_statuses"] == ['Pending'], f"Expected 1 Pending job, got {info['job_statuses']}"
        assert info["record"].ComplaintEmbedding is None, "Embedding should not exist yet"

    print("\n[3] Running worker.process_pending_jobs()...")
    print("    (NOTE: this claims ALL currently-pending jobs system-wide, not")
    print("     just this test's — the assertions below check only this")
    print("     test's own case_ids specifically, not the aggregate result,")
    print("     since the job table is shared/not test-isolated by design.)")
    result = embedding_worker.process_pending_jobs(batch_size=10)
    print(f"    Result: {result}")
    assert result["claimed"] >= 2
    assert result["completed"] >= 2

    after = get_jobs_and_records()
    print(f"\n[4] After worker run (this test's cases only):")
    for cid, info in after.items():
        rec = info["record"]
        emb_len = len(rec.ComplaintEmbedding) if rec.ComplaintEmbedding else 0
        print(f"    case={cid}: jobs={info['job_statuses']}, "
              f"embedding_bytes={emb_len} (expect 768*4=3072), "
              f"dim={rec.EmbeddingDimension}, status={rec.ProcessingStatus}, model_version_id={rec.EmbeddingModelVersionID}")
        assert info["job_statuses"] == ['Completed'], f"Expected job Completed, got {info['job_statuses']}"
        assert rec.ComplaintEmbedding is not None
        assert rec.CombinedTextEmbedding is not None
        assert emb_len == 768 * 4, f"Expected 3072 bytes (768 float32), got {emb_len}"
        assert rec.EmbeddingDimension == 768
        assert rec.ProcessingStatus == 'Completed'
        assert rec.EmbeddingModelVersionID is not None

    print("\n[5] Testing stuck-job recovery (simulating a crash mid-processing)...")
    # Create one more job, manually force it into 'Processing' with an old
    # StartedAt (simulating a worker that crashed mid-batch), then verify the
    # startup sweep recovers it.
    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()
    stuck_case_id = case_ids[0]
    cur.execute(
        "INSERT INTO ml.EmbeddingProcessingJob (IncidentRequestCaseID, JobType, Status, StartedAt) "
        "OUTPUT INSERTED.EmbeddingProcessingJobID VALUES (?, 'Reprocess', 'Processing', ?)",
        (stuck_case_id, datetime.now() - timedelta(minutes=30)),
    )
    stuck_job_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    print(f"    Simulated stuck job_id={stuck_job_id} (Processing, started 30 min ago)")

    recovered = embedding_worker.sweep_stuck_jobs_startup(stuck_threshold_minutes=15)
    print(f"    Jobs recovered by startup sweep: {recovered}")
    assert recovered >= 1

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT Status FROM ml.EmbeddingProcessingJob WHERE EmbeddingProcessingJobID = ?", (stuck_job_id,))
    status = cur.fetchone()[0]
    conn.close()
    print(f"    Stuck job status after sweep: {status}")
    assert status == 'RetryPending', f"Expected RetryPending, got {status}"

    print("\n" + "=" * 70)
    print("ALL STAGE 6 SMOKE TESTS PASSED")
    print("=" * 70)

    print(f"\n[Cleanup] Removing test cases {case_ids} and stuck job {stuck_job_id}...")
    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("DELETE FROM ml.EmbeddingProcessingJob WHERE EmbeddingProcessingJobID = ?", (stuck_job_id,))
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
    print("    Cleanup complete.")


if __name__ == "__main__":
    main()
