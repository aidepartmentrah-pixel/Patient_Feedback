"""
Stage 11 End-to-End Test — Manual Insert -> Embedding -> Training

Chains three already-proven pieces together (none re-tested in isolation
here — see ml_stage3_case_service_smoke_test.py and
ml_stage6_embedding_worker_smoke_test.py for those): a manual insert
registers an embedding job, the worker completes it, and the resulting
ml.CaseTrainingRecord row is actually picked up by the real training
data-access query (split_data._fetch_sql_server_training_dataframe()) —
not just "a row exists somewhere."

Run from the backend/ directory:
    python -m tests.test_ml_e2e_manual_insert_to_training
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.database import get_connection
from api.services.insert_service import create_record
from ml_mapping import embedding_worker
from models_directory.split_data import _fetch_sql_server_training_dataframe


def main():
    print("=" * 70)
    print("STAGE 11 E2E — Manual Insert -> Embedding -> Training")
    print("=" * 70)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT TOP 1 DomainID FROM dbo.APP_LOOKUP_DOMAIN")
    domain_id = cur.fetchone()[0]
    cur.execute("SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY WHERE DomainID = ?", (domain_id,))
    row = cur.fetchone()
    category_id = row[0] if row else None
    conn.close()

    unique_text = "E2E11 MANUAL INSERT complaint text unique marker 8f3a1c"
    data = {
        "complaint_text": unique_text,
        "feedback_received_date": "2026-07-20",
        "incident_date": "2026-07-19",
        "issuing_department_id": 1,
        "domain_id": domain_id,
        "category_id": category_id,
        "clinical_risk_type_id": 1,
        "feedback_intent_type_id": 1,
        "immediate_action": "E2E11 immediate action",
        "taken_action": "",
        "patient_name": "E2E11 MANUAL INSERT Patient",
        "is_inpatient": True,
        "source_id": 1,
        "building_id": 2,
    }

    print("\n[1] Creating case via create_record() (manual insert)...")
    result = create_record(data, save_mode='draft')
    assert result["success"], result
    case_id = result["id"]
    print(f"    case_id={case_id}")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT Status, JobType FROM ml.EmbeddingProcessingJob WHERE IncidentRequestCaseID = ?", (case_id,))
    jobs = cur.fetchall()
    conn.close()
    print(f"\n[2] Jobs after insert: {[(j.Status, j.JobType) for j in jobs]}")
    assert len(jobs) == 1 and jobs[0].Status == 'Pending' and jobs[0].JobType == 'Create'

    print("\n[3] Running embedding_worker.process_pending_jobs()...")
    result = embedding_worker.process_pending_jobs(batch_size=10)
    print(f"    Result: {result}")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT ProcessingStatus, ComplaintEmbedding, CombinedTextEmbedding FROM ml.CaseTrainingRecord "
        "WHERE IncidentRequestCaseID = ?", (case_id,)
    )
    rec = cur.fetchone()
    conn.close()
    print(f"    ProcessingStatus={rec.ProcessingStatus}, "
          f"complaint_embedding_bytes={len(rec.ComplaintEmbedding) if rec.ComplaintEmbedding else 0}")
    assert rec.ProcessingStatus == 'Completed'
    assert rec.ComplaintEmbedding is not None
    assert rec.CombinedTextEmbedding is not None

    print("\n[4] Confirming the case is training-eligible via the REAL training query...")
    df = _fetch_sql_server_training_dataframe()
    matches = df[df["complaint_text"] == unique_text]
    print(f"    Rows in training dataframe matching our case: {len(matches)}")
    assert len(matches) == 1, "Expected exactly 1 matching row in the training-eligible dataset"
    assert matches.iloc[0]["embedding_text1"] is not None

    print("\n" + "=" * 70)
    print("ALL E2E ASSERTIONS PASSED — Manual Insert -> Embedding -> Training")
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
