"""
Stage 11 End-to-End Test — Edit -> Reprocessing -> Training

ml_stage5_edit_upsert_smoke_test.py already proves an edit registers the
correct job type (TextChanged vs LabelsChanged) and upserts instead of
appending. This test chains the leg that was never exercised: actually
running the worker to reprocess the edit (confirming the embedding bytes
genuinely change, not just that a job row exists) and confirming the
edited case remains training-eligible afterward.

Run from the backend/ directory:
    python -m tests.test_ml_e2e_edit_reprocessing_to_training
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.database import get_connection
from api.services.insert_service import create_record, update_record
from ml_mapping import embedding_worker
from models_directory.split_data import _fetch_sql_server_training_dataframe


def get_training_record(case_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT ComplaintText, ComplaintEmbedding, ProcessingStatus FROM ml.CaseTrainingRecord "
        "WHERE IncidentRequestCaseID = ?", (case_id,)
    )
    row = cur.fetchone()
    conn.close()
    return row


def main():
    print("=" * 70)
    print("STAGE 11 E2E — Edit -> Reprocessing -> Training")
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
        "complaint_text": "E2E11 EDIT-REPROCESS original complaint text",
        "feedback_received_date": "2026-07-20",
        "incident_date": "2026-07-19",
        "issuing_department_id": 1,
        "domain_id": domain_id,
        "category_id": category_id,
        "clinical_risk_type_id": 1,
        "feedback_intent_type_id": 1,
        "immediate_action": "original immediate action",
        "taken_action": "",
        "patient_name": "E2E11 EDIT-REPROCESS Patient",
        "is_inpatient": True,
        "source_id": 1,
        "building_id": 2,
    }

    print("\n[1] Creating and completing the initial embedding...")
    result = create_record(data, save_mode='draft')
    assert result["success"], result
    case_id = result["id"]
    embedding_worker.process_pending_jobs(batch_size=10)

    before = get_training_record(case_id)
    print(f"    Before edit: status={before.ProcessingStatus}, text={before.ComplaintText!r}")
    assert before.ProcessingStatus == 'Completed'
    original_embedding = bytes(before.ComplaintEmbedding)

    print("\n[2] Editing complaint text (should register TextChanged job)...")
    edited_text = "E2E11 EDIT-REPROCESS — EDITED complaint text, genuinely different content"
    edit_data = dict(data)
    edit_data["complaint_text"] = edited_text
    result = update_record(case_id, edit_data, save_mode='draft')
    assert result["success"], result

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT JobType, Status FROM ml.EmbeddingProcessingJob WHERE IncidentRequestCaseID = ? "
        "ORDER BY EmbeddingProcessingJobID DESC",
        (case_id,),
    )
    latest_job = cur.fetchone()
    conn.close()
    print(f"    Latest job: {latest_job.JobType}, {latest_job.Status}")
    assert latest_job.JobType == 'TextChanged'
    assert latest_job.Status == 'Pending'

    print("\n[3] Running the worker again to actually reprocess the edit...")
    embedding_worker.process_pending_jobs(batch_size=10)

    after = get_training_record(case_id)
    print(f"    After reprocessing: status={after.ProcessingStatus}, text={after.ComplaintText!r}")
    assert after.ProcessingStatus == 'Completed'
    assert after.ComplaintText == edited_text
    new_embedding = bytes(after.ComplaintEmbedding)
    assert new_embedding != original_embedding, \
        "Embedding bytes did not change after editing the complaint text — reprocessing did not actually happen"

    print("\n[4] Confirming exactly one row still exists (no duplicate-append) and it's training-eligible...")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ml.CaseTrainingRecord WHERE IncidentRequestCaseID = ?", (case_id,))
    count = cur.fetchone()[0]
    conn.close()
    assert count == 1, f"Expected exactly 1 ml.CaseTrainingRecord row, got {count}"

    df = _fetch_sql_server_training_dataframe()
    matches = df[df["complaint_text"] == edited_text]
    print(f"    Rows in training dataframe matching the EDITED text: {len(matches)}")
    assert len(matches) == 1

    print("\n" + "=" * 70)
    print("ALL E2E ASSERTIONS PASSED — Edit -> Reprocessing -> Training")
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
