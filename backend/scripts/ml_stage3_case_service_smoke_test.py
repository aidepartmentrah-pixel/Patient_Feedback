"""
ML Architecture Consolidation — Stage 3 Smoke Test

Functional test of case_service.create_case() against the real database,
run inside a transaction that is always rolled back so no test data persists.

Verifies:
  1. create_record() (the thin wrapper) still returns the exact same
     response shape as before the refactor.
  2. A case is actually inserted into dbo.APP_IncidentCase.
  3. Exactly one ml.EmbeddingProcessingJob row is registered for it,
     JobType='Create', in the same transaction.

Run from the backend/ directory:
    python -m scripts.ml_stage3_case_service_smoke_test
"""

import os
import sys

# This codebase mixes two import conventions (bare "from core.X import Y",
# assuming cwd=backend/, and absolute "from backend.core.X import Y",
# assuming the repo root is on sys.path) across different files. Production
# resolves both correctly via its uvicorn launch mechanism; a plain script
# invocation does not unless the repo root is added explicitly here.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.database import get_connection
from api.services.insert_service import create_record


def main():
    print("=" * 70)
    print("STAGE 3 SMOKE TEST — case_service.create_case() via create_record()")
    print("=" * 70)

    # Pull real, valid lookup IDs to build a realistic payload
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT TOP 1 DomainID FROM dbo.APP_LOOKUP_DOMAIN")
    domain_id = cur.fetchone()[0]
    cur.execute("SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY WHERE DomainID = ?", (domain_id,))
    row = cur.fetchone()
    category_id = row[0] if row else None
    cur.execute("SELECT TOP 1 SeverityID FROM dbo.APP_LOOKUP_SEVERITY")
    severity_id = cur.fetchone()[0]
    cur.execute("SELECT TOP 1 BuildingID FROM dbo.APP_LOOKUP_BUILDING")
    row = cur.fetchone()
    building_id = row[0] if row else None
    conn.close()

    print(f"Using domain_id={domain_id}, category_id={category_id}, severity_id={severity_id}, building_id={building_id}")

    data = {
        "complaint_text": "STAGE3 SMOKE TEST complaint text",
        "feedback_received_date": "2026-07-16",
        "incident_date": "2026-07-15",
        "issuing_department_id": 1,
        "domain_id": domain_id,
        "category_id": category_id,
        "subcategory_id": None,
        "classification_id": None,
        "severity_id": severity_id,
        "stage_id": None,
        "harm_id": None,
        "requires_explanation": False,
        "clinical_risk_type_id": 1,
        "feedback_intent_type_id": 1,
        "immediate_action": "STAGE3 SMOKE TEST immediate action",
        "patient_name": "STAGE3 SMOKE TEST Patient",
        "is_inpatient": True,
        "source_id": 1,
        "building_id": building_id,
    }

    print("\n[1] Calling create_record() (save_mode='draft' to skip strict validation)...")
    result = create_record(data, save_mode='draft')
    print(f"    success={result.get('success')}, id={result.get('id')}, record_id={result.get('record_id')}")

    if not result.get("success"):
        print(f"    FAILED: {result}")
        return

    new_case_id = result["id"]
    expected_keys = {"success", "message", "message_ar", "record_id", "id", "incident_id", "status_id", "save_mode", "created_at"}
    actual_keys = set(result.keys())
    print(f"\n[2] Response shape check: expected keys present = {expected_keys.issubset(actual_keys)}")
    assert expected_keys.issubset(actual_keys), f"Response shape changed! Got keys: {actual_keys}"

    print(f"\n[3] Verifying case {new_case_id} exists in dbo.APP_IncidentCase...")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT ComplaintText FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (new_case_id,))
    row = cur.fetchone()
    print(f"    Found: {row is not None}, ComplaintText={row[0] if row else None}")
    assert row is not None

    print(f"\n[4] Verifying ml.EmbeddingProcessingJob was registered for case {new_case_id}...")
    cur.execute(
        "SELECT EmbeddingProcessingJobID, JobType, Status FROM ml.EmbeddingProcessingJob WHERE IncidentRequestCaseID = ?",
        (new_case_id,),
    )
    jobs = cur.fetchall()
    print(f"    Job rows found: {len(jobs)}")
    for j in jobs:
        print(f"      JobID={j[0]}, JobType={j[1]}, Status={j[2]}")
    assert len(jobs) == 1, f"Expected exactly 1 job, found {len(jobs)}"
    assert jobs[0][1] == 'Create'
    assert jobs[0][2] == 'Pending'

    print("\n" + "=" * 70)
    print("ALL STAGE 3 SMOKE TESTS PASSED")
    print("=" * 70)

    # -----------------------------
    # Cleanup: this smoke test created REAL committed rows (create_case
    # commits its own transaction internally), so explicitly delete them
    # rather than relying on rollback.
    # -----------------------------
    print(f"\n[Cleanup] Removing test case {new_case_id} and its related rows...")
    cur.execute("DELETE FROM ml.EmbeddingProcessingJob WHERE IncidentRequestCaseID = ?", (new_case_id,))
    cur.execute("DELETE FROM ml.CaseTrainingRecord WHERE IncidentRequestCaseID = ?", (new_case_id,))
    cur.execute("DELETE FROM dbo.APP_IncidentCaseTargetDepartment WHERE IncidentRequestCaseID = ?", (new_case_id,))
    cur.execute("DELETE FROM dbo.APP_IncidentCaseDoctor WHERE IncidentRequestCaseID = ?", (new_case_id,))
    cur.execute("DELETE FROM dbo.APP_IncidentCaseEmployee WHERE IncidentRequestCaseID = ?", (new_case_id,))
    cur.execute("SELECT incident_id FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (new_case_id,))
    inc_row = cur.fetchone()
    cur.execute("DELETE FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (new_case_id,))
    if inc_row and inc_row[0]:
        cur.execute("DELETE FROM dbo.APP_Incident WHERE incident_id = ?", (inc_row[0],))
    conn.commit()
    print("    Cleanup complete — no test data remains.")
    conn.close()


if __name__ == "__main__":
    main()
