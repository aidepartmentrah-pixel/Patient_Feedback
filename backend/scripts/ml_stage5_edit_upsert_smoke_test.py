"""
ML Architecture Consolidation — Stage 5 Smoke Test

Verifies the edit-path fix: editing a case updates the ONE current
ml.CaseTrainingRecord row instead of appending a duplicate (the exact bug
this stage exists to fix), and that TextChanged vs LabelsChanged jobs are
registered correctly depending on what actually changed.

Run from the backend/ directory:
    python -m scripts.ml_stage5_edit_upsert_smoke_test
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.database import get_connection
from api.services.insert_service import create_record, update_record


def main():
    print("=" * 70)
    print("STAGE 5 SMOKE TEST — edit-path upsert (not append)")
    print("=" * 70)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT TOP 1 DomainID FROM dbo.APP_LOOKUP_DOMAIN")
    domain_id = cur.fetchone()[0]
    cur.execute("SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY WHERE DomainID = ?", (domain_id,))
    row = cur.fetchone()
    category_id = row[0] if row else None
    cur.execute("SELECT DomainID FROM dbo.APP_LOOKUP_DOMAIN WHERE DomainID != ?", (domain_id,))
    row = cur.fetchone()
    other_domain_id = row[0] if row else domain_id
    conn.close()

    data = {
        "complaint_text": "STAGE5 SMOKE TEST original complaint text",
        "feedback_received_date": "2026-07-16",
        "incident_date": "2026-07-15",
        "issuing_department_id": 1,
        "domain_id": domain_id,
        "category_id": category_id,
        "clinical_risk_type_id": 1,
        "feedback_intent_type_id": 1,
        "immediate_action": "original immediate action",
        "taken_action": "",
        "patient_name": "STAGE5 SMOKE TEST Patient",
        "is_inpatient": True,
        "source_id": 1,
        "building_id": 2,
    }

    print("\n[1] Creating test case...")
    result = create_record(data, save_mode='draft')
    assert result["success"], result
    case_id = result["id"]
    print(f"    Created case_id={case_id}")

    def get_training_record():
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT CaseTrainingRecordID, ComplaintText, DomainID, ProcessingStatus "
            "FROM ml.CaseTrainingRecord WHERE IncidentRequestCaseID = ?",
            (case_id,),
        )
        rows = cur.fetchall()
        conn.close()
        return rows

    def get_jobs():
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT JobType, Status FROM ml.EmbeddingProcessingJob WHERE IncidentRequestCaseID = ? ORDER BY EmbeddingProcessingJobID",
            (case_id,),
        )
        rows = cur.fetchall()
        conn.close()
        return rows

    records = get_training_record()
    print(f"\n[2] After create: {len(records)} ml.CaseTrainingRecord row(s)")
    assert len(records) == 1, f"Expected 1 row after create, got {len(records)}"
    assert records[0].ComplaintText == data["complaint_text"]
    jobs = get_jobs()
    print(f"    Jobs so far: {[(j.JobType, j.Status) for j in jobs]}")
    assert jobs[-1].JobType == 'Create'

    print("\n[3] Editing complaint text (should trigger TextChanged, still 1 row)...")
    edit1 = dict(data)
    edit1["complaint_text"] = "STAGE5 SMOKE TEST — EDITED complaint text"
    result = update_record(case_id, edit1, save_mode='draft')
    assert result["success"], result

    records = get_training_record()
    print(f"    ml.CaseTrainingRecord rows: {len(records)} (must stay 1, not 2)")
    assert len(records) == 1, f"UPSERT FAILED — expected exactly 1 row, got {len(records)} (duplicate-append bug reintroduced!)"
    assert records[0].ComplaintText == edit1["complaint_text"], "Row wasn't updated with new text"
    jobs = get_jobs()
    print(f"    Jobs so far: {[(j.JobType, j.Status) for j in jobs]}")
    assert jobs[-1].JobType == 'TextChanged', f"Expected latest job TextChanged, got {jobs[-1].JobType}"

    print("\n[4] Editing ONLY the domain label (same text, should trigger LabelsChanged)...")
    edit2 = dict(edit1)
    edit2["domain_id"] = other_domain_id
    # category_id must still belong to the new domain for FK/hierarchy checks to pass in 'draft' mode
    # (draft mode skips required-field checks but not FK/hierarchy validation)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY WHERE DomainID = ?", (other_domain_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        edit2["category_id"] = row[0]

    result = update_record(case_id, edit2, save_mode='draft')
    assert result["success"], result

    records = get_training_record()
    print(f"    ml.CaseTrainingRecord rows: {len(records)} (must stay 1)")
    assert len(records) == 1
    assert records[0].DomainID == other_domain_id, "DomainID wasn't updated"
    assert records[0].ComplaintText == edit1["complaint_text"], "Text should be unchanged from previous edit"
    jobs = get_jobs()
    print(f"    Jobs so far: {[(j.JobType, j.Status) for j in jobs]}")
    assert jobs[-1].JobType == 'LabelsChanged', f"Expected latest job LabelsChanged, got {jobs[-1].JobType}"

    print("\n" + "=" * 70)
    print("ALL STAGE 5 SMOKE TESTS PASSED — edit path upserts, never appends")
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
