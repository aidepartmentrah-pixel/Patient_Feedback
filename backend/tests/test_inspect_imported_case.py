"""
Regression test for Inspect showing "No Data" on bulk-imported records.

Bulk import (case_service.create_case(..., save_mode='import_closed')) never
creates an APP_AdministrativeSubcase, and GET /api/v2/workflow/incident/{id}/
responses used to drop every case without one for non-privileged users --
so the whole Inspect page came back empty.

Creates one case through the same calls import_service._import_group makes
(import_db.insert_incident -> create_case(BulkImport, import_closed) ->
assign_case_to_incident), copying lookup IDs from an existing case, then
calls the endpoint as:
1. a SECTION_ADMIN scoped to the case's target unit -> 1 case,
   has_workflow = False, target unit name resolved, taken_action returned
2. a SECTION_ADMIN scoped to an unrelated unit -> 0 cases
3. a COMPLAINT_SUPERVISOR -> 1 case
4. regression: an existing workflow case still reports has_workflow = True

Everything created is deleted at the end.

Run: python backend/tests/test_inspect_imported_case.py
"""

import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(backend_dir))
sys.path.insert(0, backend_dir)

from backend.core.database import get_connection
from backend.api.schemas.auth_models import CurrentUser, UserScope
from backend.api.db_layer import import_db
from backend.api.db_layer.incident_parent import assign_case_to_incident
from backend.api.services.case_service import create_case
from backend.api_v2.routers.workflow_router import get_incident_responses


def fetch_one(sql, params=()):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return cursor.fetchone()
    finally:
        conn.close()


def make_user(role_code, unit_id):
    return CurrentUser(
        user_id=1, username=f"test_{role_code.lower()}", is_active=True,
        scopes=[UserScope(role_code=role_code, org_unit_id=unit_id, org_unit_type="SECTION")],
        allowed_unit_ids={unit_id}, roles=[role_code],
    )


def create_imported_case(created, target_unit_id=None):
    template = fetch_one("""
        SELECT TOP 1 c.*, i.feedback_intent_type_id AS intent_id, i.issuing_org_unit_id AS issuing_id
        FROM dbo.APP_IncidentCase c JOIN dbo.APP_Incident i ON i.incident_id = c.incident_id
        ORDER BY c.IncidentRequestCaseID
    """)
    assert template, "Need at least one existing case to copy lookup IDs from"
    if target_unit_id:
        target_unit = fetch_one("SELECT UniqueID, Name FROM dbo.AdminsrationUnit WHERE UniqueID = ?", (target_unit_id,))
    else:
        target_unit = fetch_one(
            "SELECT TOP 1 UniqueID, Name FROM dbo.AdminsrationUnit WHERE Type = 324 ORDER BY UniqueID DESC")
    user_id = fetch_one("SELECT TOP 1 UserID FROM dbo.APP_Users ORDER BY UserID").UserID

    conn = get_connection()
    try:
        cursor = conn.cursor()
        incident_id = import_db.insert_incident(
            patient_name="TEST IMPORT PATIENT",
            feedback_intent_type_id=template.intent_id,
            issuing_org_unit_id=template.issuing_id,
            building_id=template.BuildingID or 1,
            is_inpatient=False,
            created_by_user_id=user_id,
            cursor=cursor,
        )
        conn.commit()
        created["incident_id"] = incident_id
    finally:
        conn.close()

    result = create_case({
        "record_type_id": template.RecordTypeID,
        "patient_name": "TEST IMPORT PATIENT",
        "feedback_received_date": str(template.FeedbackRecievedDate)[:10],
        "incident_date": str(template.IncidentDate)[:10],
        "feedback_intent_type_id": template.intent_id,
        "source_id": template.SourceID,
        "issuing_department_id": template.issuing_id,
        "domain_id": template.DomainID,
        "category_id": template.CategoryID,
        "subcategory_id": template.SubCategoryID,
        "classification_id": template.ClassificationID,
        "severity_id": template.SeverityID,
        "stage_id": template.StageID,
        "harm_id": template.HarmLevelID,
        "clinical_risk_type_id": template.ClinicalRiskTypeID,
        "requires_explanation": False,
        "building_id": template.BuildingID or 1,
        "is_inpatient": False,
        "is_morbidity": False,
        "complaint_text": "TEST imported complaint text",
        "immediate_action": "TEST immediate action",
        "taken_action": "TEST taken action by section",
        "target_department_ids": [target_unit.UniqueID],
        "doctors": [],
        "employees": [],
    }, context='BulkImport', save_mode='import_closed')
    assert result.get("success"), result
    case_id = int(result["id"])
    created["case_id"] = case_id
    assign_case_to_incident(case_id, incident_id)
    return incident_id, case_id, target_unit


def cleanup(incident_id, case_id):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        if case_id:
            # ML rows first (EmbeddingProcessingJob may point at CaseTrainingRecord)
            for table in ("ml.EmbeddingProcessingJob", "ml.ImportSourceRecordMap", "ml.CaseTrainingRecord",
                          "dbo.APP_IncidentCaseTargetDepartment", "dbo.APP_IncidentCaseDoctor",
                          "dbo.APP_IncidentCaseEmployee", "dbo.APP_IncidentCaseFeedback",
                          "dbo.APP_IncidentCaseSatisfaction", "dbo.APP_AdministrativeSubcase",
                          "dbo.APP_SupervisorActionItem"):
                cursor.execute(f"DELETE FROM {table} WHERE IncidentRequestCaseID = ?", (case_id,))
            cursor.execute("DELETE FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (case_id,))
        if incident_id:
            cursor.execute("DELETE FROM dbo.APP_Incident WHERE incident_id = ?", (incident_id,))
        conn.commit()
    finally:
        conn.close()


def run():
    created = {"incident_id": None, "case_id": None}
    try:
        incident_id, case_id, target_unit = create_imported_case(created)
        assert fetch_one("SELECT COUNT(*) FROM dbo.APP_AdministrativeSubcase WHERE IncidentRequestCaseID = ?",
                         (case_id,))[0] == 0, "fixture should mirror import: no subcase"
        print(f"  created imported case {case_id} (incident {incident_id}) targeting unit {target_unit.UniqueID}")

        in_scope = get_incident_responses(incident_id, make_user("SECTION_ADMIN", target_unit.UniqueID))
        assert len(in_scope["cases"]) == 1, in_scope
        c = in_scope["cases"][0]
        assert c["has_workflow"] is False, c
        assert c["target_org_unit_id"] == target_unit.UniqueID, c
        assert c["target_org_unit_name"] == target_unit.Name, c
        assert c["taken_action"] == "TEST taken action by section", c
        assert c["action_items"] == [] and c["status"] is None, c
        print("  OK in-scope section admin sees the imported case (has_workflow=False, target unit resolved)")

        out_scope = get_incident_responses(incident_id, make_user("SECTION_ADMIN", -1))
        assert out_scope["cases"] == [], out_scope
        print("  OK out-of-scope section admin sees nothing")

        supervisor = get_incident_responses(incident_id, make_user("COMPLAINT_SUPERVISOR", -1))
        assert len(supervisor["cases"]) == 1, supervisor
        print("  OK supervisor sees the imported case")

        wf = fetch_one("""SELECT TOP 1 c.incident_id, s.TargetOrgUnitID FROM dbo.APP_AdministrativeSubcase s
                          JOIN dbo.APP_IncidentCase c ON c.IncidentRequestCaseID = s.IncidentRequestCaseID
                          WHERE c.incident_id IS NOT NULL AND s.TargetOrgUnitID IS NOT NULL""")
        if wf:
            wf_resp = get_incident_responses(wf.incident_id, make_user("SECTION_ADMIN", wf.TargetOrgUnitID))
            assert wf_resp["cases"] and all(x["has_workflow"] for x in wf_resp["cases"]), wf_resp
            print(f"  OK workflow incident {wf.incident_id} still has_workflow=True for its target unit")

        print("ALL INSPECT IMPORTED-CASE TESTS PASSED")
    finally:
        cleanup(created["incident_id"], created["case_id"])
        print(f"  cleaned up case {created['case_id']}, incident {created['incident_id']}")


if __name__ == "__main__":
    run()
