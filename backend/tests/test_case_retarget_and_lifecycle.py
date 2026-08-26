"""
Test Suite for the Edit-page case lifecycle work:
  - Retarget mechanism (update_case() when a published case's Target Unit changes)
  - Add Case auto-publish (update_case() when a Draft case's incident already
    has a published sibling)
  - Delete Case (delete_case_from_incident(): last-case guard, hard-delete for
    unpublished cases, soft-delete + subcase retirement for published cases)

Builds its own fixture data (incident + case rows) rather than relying on
hardcoded IDs from an existing dev database, since these tests need cases in
specific, controlled starting states (Draft, or Open with a known target).

Run: python backend/tests/test_case_retarget_and_lifecycle.py
"""

import sys
sys.path.insert(0, 'backend')
# case_service.py (and a few sibling modules) import via `from backend.X import
# Y` absolute paths rather than the bare `X.Y` style most of this codebase
# uses. In the real app this resolves only because an earlier-imported module
# (api.services.classification_service) inserts the repo root onto sys.path
# as an import-time side effect before case_service ever gets imported.
# Replicate that here explicitly rather than relying on import order.
sys.path.insert(0, '.')

from datetime import date
from core.database import get_connection
from api.services import case_service
from api.db_layer.incident_case import create_incident_case
from api.db_layer.incident_parent import create_incident_parent, assign_case_to_incident, add_case_to_incident, delete_case_from_incident
from api_v2.db_layer import administrative_subcase_db as subcase_db
from api_v2.services.case_creation_service import create_subcases_for_incident, _ORG_TYPE_SECTION

tests_run = 0
tests_passed = 0
tests_failed = 0


def test(description):
    def decorator(func):
        def wrapper():
            global tests_run, tests_passed, tests_failed
            tests_run += 1
            try:
                func()
                tests_passed += 1
                print(f"PASS: {description}")
                return True
            except AssertionError as e:
                tests_failed += 1
                print(f"FAIL: {description}")
                print(f"   Error: {e}")
                return False
            except Exception as e:
                tests_failed += 1
                print(f"ERROR: {description}")
                print(f"   Exception: {e}")
                return False
        return wrapper
    return decorator


# ============================================================
# FIXTURE HELPERS
# ============================================================

def _fetch_scalar(query, params=()):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query, params)
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        cursor.close()
        conn.close()


def _get_two_section_unit_ids():
    """Two distinct active Section-type (Type=324) org units, for old/new target."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT TOP 2 UniqueID FROM dbo.AdminsrationUnit WHERE Type = ? ORDER BY UniqueID",
            (_ORG_TYPE_SECTION,)
        )
        ids = [r.UniqueID for r in cursor.fetchall()]
        assert len(ids) >= 2, "Need at least 2 Section-type org units in this DB to run these tests"
        return ids[0], ids[1]
    finally:
        cursor.close()
        conn.close()


def _get_valid_classification_chain():
    """One fully-linked Domain -> Category -> Subcategory -> Classification chain."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT TOP 1 c.DomainID, c.CategoryID, s.SubCategoryID, cl.ClassificationID
            FROM dbo.APP_LOOKUP_CATEGORY c
            INNER JOIN dbo.APP_LOOKUP_SUBCATEGORY s ON s.CategoryID = c.CategoryID
            INNER JOIN dbo.APP_LOOKUP_CLASSIFICATION cl ON cl.SubCategoryID = s.SubCategoryID
        """)
        row = cursor.fetchone()
        assert row is not None, "Need a fully-linked Domain->Category->Subcategory->Classification chain in this DB"
        return row.DomainID, row.CategoryID, row.SubCategoryID, row.ClassificationID
    finally:
        cursor.close()
        conn.close()


def _update_case_payload(*, issuing_org_unit_id, building_id, source_id, target_department_ids):
    """
    Payload shape for case_service.update_case() — snake_case keys, distinct
    from _base_case_payload()'s PascalCase (which matches create_incident_case()'s
    direct DB-layer format instead). Includes every field save_mode='workflow'
    requires for a non-Notice case.
    """
    return {
        "complaint_text": "Test complaint for retarget/lifecycle suite",
        "immediate_action": "Test immediate action",
        "taken_action": "Test taken action",
        "feedback_received_date": date.today().isoformat(),
        "incident_date": date.today().isoformat(),
        "patient_name": "Test Patient",
        "issuing_department_id": issuing_org_unit_id,
        "is_inpatient": True,
        "building_id": building_id,
        "domain_id": domain_id,
        "category_id": category_id,
        "subcategory_id": subcategory_id,
        "classification_id": classification_id,
        "severity_id": severity_id,
        "stage_id": stage_id,
        "harm_id": harm_id,
        "clinical_risk_type_id": 1,  # Ordinary
        "feedback_intent_type_id": feedback_intent_type_id,
        "requires_explanation": False,
        "source_id": source_id,
        "target_department_ids": target_department_ids,
    }


def _base_case_payload(*, issuing_org_unit_id, building_id, source_id, case_status_id):
    return {
        "ComplaintText": "Test complaint for retarget/lifecycle suite",
        "ImmediateAction": "",
        "TakenAction": "",
        "FeedbackRecievedDate": date.today(),
        "IncidentDate": date.today(),
        "PatientName": "Test Patient",
        "IssuingOrgUnitID": issuing_org_unit_id,
        "CreatedByUserID": 1,
        "isINPatient": 1,
        "IsMorbidity": 0,
        "ClinicalRiskTypeID": 1,  # Ordinary
        "FeedbackIntentTypeID": None,
        "BuildingID": building_id,
        "DomainID": None,
        "CategoryID": None,
        "SubCategoryID": None,
        "ClassificationID": None,
        "SeverityID": None,
        "StageID": None,
        "HarmLevelID": None,
        "CaseStatusID": case_status_id,
        "SourceID": source_id,
        "ExplanationStatusID": 4,  # No Explanation Needed
        "RequiresExplanation": 0,
        "RecordTypeID": 1,
    }


def _set_target_departments(case_id, dept_ids):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM dbo.APP_IncidentCaseTargetDepartment WHERE IncidentRequestCaseID = ?", (case_id,))
        for idx, dept_id in enumerate(dept_ids):
            cursor.execute(
                """
                INSERT INTO dbo.APP_IncidentCaseTargetDepartment (IncidentRequestCaseID, DepartmentID, IsPrimary, AssignedByUserID)
                VALUES (?, ?, ?, 1)
                """,
                (case_id, dept_id, 1 if idx == 0 else 0)
            )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def _make_incident_with_case(*, case_status_id, issuing_org_unit_id, building_id, source_id):
    """Create a parent incident with exactly one case, return (incident_id, case_id)."""
    incident_id = create_incident_parent({
        "patient_name": "Test Patient",
        "primary_doctor_name": None,
        "primary_worker_name": None,
        "feedback_intent_type_id": None,
        "issuing_org_unit_id": issuing_org_unit_id,
        "complaint_summary": "Test complaint for retarget/lifecycle suite",
        "building_id": building_id,
        "is_inpatient": True,
        "created_by_user_id": 1,
    })
    payload = _base_case_payload(
        issuing_org_unit_id=issuing_org_unit_id,
        building_id=building_id,
        source_id=source_id,
        case_status_id=case_status_id,
    )
    case_id = create_incident_case(payload)
    assign_case_to_incident(case_id, incident_id)
    return incident_id, case_id


def _hard_delete_case_row(case_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM dbo.APP_IncidentCaseTargetDepartment WHERE IncidentRequestCaseID = ?", (case_id,))
        cursor.execute("DELETE FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (case_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def _delete_incident_row(incident_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM dbo.APP_Incident WHERE incident_id = ?", (incident_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def _cleanup_case(case_id):
    """Remove a case's ML rows, subcases (+ action items), and the case row itself."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM ml.EmbeddingProcessingJob WHERE IncidentRequestCaseID = ?", (case_id,))
        cursor.execute("DELETE FROM ml.CaseTrainingRecord WHERE IncidentRequestCaseID = ?", (case_id,))
        cursor.execute("SELECT SubcaseID FROM dbo.APP_AdministrativeSubcase WHERE IncidentRequestCaseID = ?", (case_id,))
        subcase_ids = [r.SubcaseID for r in cursor.fetchall()]
        for sid in subcase_ids:
            cursor.execute("DELETE FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (sid,))
        for sid in subcase_ids:
            cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (sid,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()
    _hard_delete_case_row(case_id)


print("=" * 70)
print("CASE RETARGET / AUTO-PUBLISH / DELETE-CASE TEST SUITE")
print("=" * 70)
print()

section_a_id, section_b_id = _get_two_section_unit_ids()
building_id = _fetch_scalar("SELECT TOP 1 BuildingID FROM dbo.APP_LOOKUP_BUILDING")
source_id = _fetch_scalar("SELECT TOP 1 SourceID FROM dbo.APP_LOOKUP_SOURCE")
assert building_id is not None, "Need at least 1 row in APP_LOOKUP_BUILDING to run these tests"
assert source_id is not None, "Need at least 1 row in APP_LOOKUP_SOURCE to run these tests"

domain_id, category_id, subcategory_id, classification_id = _get_valid_classification_chain()
severity_id = _fetch_scalar("SELECT TOP 1 SeverityID FROM dbo.APP_LOOKUP_SEVERITY")
stage_id = _fetch_scalar("SELECT TOP 1 StageID FROM dbo.APP_LOOKUP_CASE_STAGE")
harm_id = _fetch_scalar("SELECT TOP 1 HarmID FROM dbo.APP_LOOKUP_HARM_LEVEL")
feedback_intent_type_id = _fetch_scalar("SELECT TOP 1 FeedbackIntentTypeID FROM dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE")
for name, val in [("severity_id", severity_id), ("stage_id", stage_id), ("harm_id", harm_id),
                   ("feedback_intent_type_id", feedback_intent_type_id)]:
    assert val is not None, f"Need at least 1 row providing {name} to run these tests"

print(f"Using section A={section_a_id}, section B={section_b_id}, building={building_id}, source={source_id}")
print(f"Using domain={domain_id}, category={category_id}, subcategory={subcategory_id}, classification={classification_id}")
print(f"Using severity={severity_id}, stage={stage_id}, harm={harm_id}, feedback_intent_type={feedback_intent_type_id}")
print()

_cleanup_incident_ids = []
_cleanup_case_ids = []


# ============================================================
# 1. RETARGET MECHANISM
# ============================================================

@test("1. Retargeting a published case redirects its subcase to the new unit, status reset")
def test_retarget_redirects_subcase():
    incident_id, case_id = _make_incident_with_case(
        case_status_id=1,  # Open — already "published"
        issuing_org_unit_id=section_a_id,
        building_id=building_id,
        source_id=source_id,
    )
    _cleanup_incident_ids.append(incident_id)
    _cleanup_case_ids.append(case_id)

    _set_target_departments(case_id, [section_a_id])
    create_subcases_for_incident(case_id, current_user=None)

    before = subcase_db.get_subcases_by_incident(case_id)
    assert len(before) == 1, "Expected exactly one subcase before retarget"
    subcase_id = before[0]["subcase_id"]
    assert before[0]["target_org_unit_id"] == section_a_id
    assert before[0]["status"] == "SUBMITTED_TO_SECTION"

    # Simulate the old section having already accepted/explained, so we can
    # verify none of it carries over to the new target below.
    subcase_db.update_section_explanation(subcase_id, "Old section's explanation text", 1)
    subcase_db.update_subcase_status(subcase_id, "SECTION_ACCEPTED_PENDING_DEPT", 1)

    result = case_service.update_case(
        case_id,
        _update_case_payload(issuing_org_unit_id=section_a_id, building_id=building_id,
                              source_id=source_id, target_department_ids=[section_b_id]),
        save_mode="workflow",
    )
    assert result["success"], f"update_case should succeed: {result}"

    all_subcases = subcase_db.get_subcases_by_incident(case_id)
    assert len(all_subcases) == 1, \
        f"UQ_APP_AdministrativeSubcase_CaseID allows only one row per case — expected 1, got {len(all_subcases)}"
    row = all_subcases[0]
    assert row["subcase_id"] == subcase_id, "Retarget must update the same row, not create a second one"

    print(f"   Subcase {subcase_id} redirected in place, section A's old acceptance discarded")

test_retarget_redirects_subcase()


@test("2. Retargeted subcase targets the new unit, fresh status, no carried-over text")
def test_retarget_is_blank_for_new_target():
    # Reuses the case from test 1 (already retargeted to section_b_id there)
    incident_id, case_id = _cleanup_incident_ids[-1], _cleanup_case_ids[-1]

    all_subcases = subcase_db.get_subcases_by_incident(case_id)
    assert len(all_subcases) == 1
    row = all_subcases[0]

    assert row["target_org_unit_id"] == section_b_id, "Subcase should now target the new unit"
    assert row["status"] == "SUBMITTED_TO_SECTION", \
        f"Should reset to the new target's initial status, got {row['status']}"
    assert row["section_explanation_text"] is None, "New target must see blank space — no carried-over text"
    assert row["section_rejection_text"] is None
    assert row["department_explanation_text"] is None
    assert row["department_rejection_text"] is None

    print(f"   Subcase {row['subcase_id']} now targets {section_b_id}, completely blank as expected")

test_retarget_is_blank_for_new_target()


@test("3. Saving a published case WITHOUT changing its target does not touch its subcase")
def test_no_target_change_no_retarget():
    incident_id, case_id = _make_incident_with_case(
        case_status_id=1,
        issuing_org_unit_id=section_a_id,
        building_id=building_id,
        source_id=source_id,
    )
    _cleanup_incident_ids.append(incident_id)
    _cleanup_case_ids.append(case_id)

    _set_target_departments(case_id, [section_a_id])
    create_subcases_for_incident(case_id, current_user=None)

    before = subcase_db.get_subcases_by_incident(case_id)
    assert len(before) == 1
    original_subcase_id = before[0]["subcase_id"]

    # Save again with the SAME target — should be a no-op for subcases
    payload = _update_case_payload(issuing_org_unit_id=section_a_id, building_id=building_id,
                                    source_id=source_id, target_department_ids=[section_a_id])
    result = case_service.update_case(case_id, payload, save_mode="workflow")
    assert result["success"]

    after = subcase_db.get_subcases_by_incident(case_id)
    assert len(after) == 1, "No new subcase should be created when the target didn't change"
    assert after[0]["subcase_id"] == original_subcase_id
    assert after[0]["status"] == "SUBMITTED_TO_SECTION", "Untouched subcase should keep its original status"

    print("   Confirmed: same-target save leaves the subcase completely untouched")

test_no_target_change_no_retarget()


# ============================================================
# 4-5. ADD CASE AUTO-PUBLISH
# ============================================================

@test("4. A Draft case added to an incident with a published sibling auto-publishes on save")
def test_add_case_auto_publishes():
    incident_id, published_case_id = _make_incident_with_case(
        case_status_id=1,  # Open sibling already published
        issuing_org_unit_id=section_a_id,
        building_id=building_id,
        source_id=source_id,
    )
    _cleanup_incident_ids.append(incident_id)
    _cleanup_case_ids.append(published_case_id)
    _set_target_departments(published_case_id, [section_a_id])

    add_result = add_case_to_incident(incident_id, created_by_user_id=1)
    new_case_id = add_result
    _cleanup_case_ids.append(new_case_id)

    status_before = _fetch_scalar(
        "SELECT CaseStatusID FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (new_case_id,)
    )
    assert status_before == 4, f"add_case_to_incident should create a Draft case, got status {status_before}"

    payload = _update_case_payload(issuing_org_unit_id=section_a_id, building_id=building_id,
                                    source_id=source_id, target_department_ids=[section_b_id])
    result = case_service.update_case(new_case_id, payload, save_mode="workflow")
    assert result["success"], f"update_case should succeed: {result}"

    status_after = _fetch_scalar(
        "SELECT CaseStatusID FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (new_case_id,)
    )
    assert status_after != 4, "Case should no longer be Draft after saving alongside a published sibling"
    assert status_after in (1, 3), f"Expected Open(1) or Closed(3) per FSM, got {status_after}"

    new_subcases = subcase_db.get_subcases_by_incident(new_case_id)
    assert len(new_subcases) == 1, "Auto-publish should create exactly one subcase for the new case"
    assert new_subcases[0]["target_org_unit_id"] == section_b_id

    print(f"   New case {new_case_id} auto-published to status {status_after}, subcase created")

test_add_case_auto_publishes()


@test("5. A Draft case added to an all-Draft incident stays Draft on save (unchanged behavior)")
def test_add_case_stays_draft_when_no_published_sibling():
    incident_id, draft_case_id = _make_incident_with_case(
        case_status_id=4,  # Draft sibling — nothing published yet
        issuing_org_unit_id=section_a_id,
        building_id=building_id,
        source_id=source_id,
    )
    _cleanup_incident_ids.append(incident_id)
    _cleanup_case_ids.append(draft_case_id)

    add_result = add_case_to_incident(incident_id, created_by_user_id=1)
    new_case_id = add_result
    _cleanup_case_ids.append(new_case_id)

    payload = _update_case_payload(issuing_org_unit_id=section_a_id, building_id=building_id,
                                    source_id=source_id, target_department_ids=[section_a_id])
    result = case_service.update_case(new_case_id, payload, save_mode="workflow")
    assert result["success"]

    status_after = _fetch_scalar(
        "SELECT CaseStatusID FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (new_case_id,)
    )
    assert status_after == 4, f"Case should stay Draft when no sibling is published, got {status_after}"

    subcases = subcase_db.get_subcases_by_incident(new_case_id)
    assert len(subcases) == 0, "No subcase should be created for a case that stayed Draft"

    print("   Confirmed: no auto-publish when the whole incident is still unpublished")

test_add_case_stays_draft_when_no_published_sibling()


# ============================================================
# 6-8. DELETE CASE
# ============================================================

@test("6. Deleting the only case in an incident is rejected")
def test_delete_last_case_rejected():
    incident_id, case_id = _make_incident_with_case(
        case_status_id=4,
        issuing_org_unit_id=section_a_id,
        building_id=building_id,
        source_id=source_id,
    )
    _cleanup_incident_ids.append(incident_id)
    _cleanup_case_ids.append(case_id)

    raised = False
    try:
        delete_case_from_incident(incident_id, case_id, deleted_by_user_id=1)
    except ValueError:
        raised = True
    assert raised, "Deleting an incident's only case must raise ValueError"

    still_there = _fetch_scalar(
        "SELECT COUNT(*) FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (case_id,)
    )
    assert still_there == 1, "Case must not have been deleted"

    print("   Confirmed: last-case guard rejected the delete")

test_delete_last_case_rejected()


@test("7. Deleting a Draft case (with a sibling) hard-deletes it")
def test_delete_draft_case_hard_deletes():
    incident_id, case_a = _make_incident_with_case(
        case_status_id=4,
        issuing_org_unit_id=section_a_id,
        building_id=building_id,
        source_id=source_id,
    )
    _cleanup_incident_ids.append(incident_id)
    _cleanup_case_ids.append(case_a)
    case_b = add_case_to_incident(incident_id, created_by_user_id=1)
    _set_target_departments(case_b, [section_a_id])

    result = delete_case_from_incident(incident_id, case_b, deleted_by_user_id=1)
    assert result["mode"] == "hard", f"Draft case should hard-delete, got mode={result['mode']}"

    row_gone = _fetch_scalar(
        "SELECT COUNT(*) FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (case_b,)
    )
    assert row_gone == 0, "Hard-deleted Draft case row should be gone"

    targets_gone = _fetch_scalar(
        "SELECT COUNT(*) FROM dbo.APP_IncidentCaseTargetDepartment WHERE IncidentRequestCaseID = ?", (case_b,)
    )
    assert targets_gone == 0, "Hard-deleted case's target department rows should be gone too"

    print(f"   Draft case {case_b} correctly hard-deleted, sibling {case_a} untouched")

test_delete_draft_case_hard_deletes()


@test("8. Deleting a published case soft-deletes it and retires its subcase")
def test_delete_published_case_soft_deletes():
    incident_id, case_a = _make_incident_with_case(
        case_status_id=1,  # Open
        issuing_org_unit_id=section_a_id,
        building_id=building_id,
        source_id=source_id,
    )
    _cleanup_incident_ids.append(incident_id)
    _cleanup_case_ids.append(case_a)
    _set_target_departments(case_a, [section_a_id])
    create_subcases_for_incident(case_a, current_user=None)

    # A Draft sibling so case_a isn't the incident's last case
    case_b = add_case_to_incident(incident_id, created_by_user_id=1)
    _cleanup_case_ids.append(case_b)

    subcases_before = subcase_db.get_subcases_by_incident(case_a)
    assert len(subcases_before) == 1
    subcase_id = subcases_before[0]["subcase_id"]

    result = delete_case_from_incident(incident_id, case_a, deleted_by_user_id=1)
    assert result["mode"] == "soft", f"Published case should soft-delete, got mode={result['mode']}"

    status_after = _fetch_scalar(
        "SELECT CaseStatusID FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (case_a,)
    )
    assert status_after == 3, f"Soft-deleted case should be Closed(3), got {status_after}"

    subcases_after = subcase_db.get_subcases_by_incident(case_a)
    retired = next(r for r in subcases_after if r["subcase_id"] == subcase_id)
    assert retired["status"] == "CASE_DELETED", f"Subcase should be retired, got {retired['status']}"

    print(f"   Published case {case_a} soft-deleted (Closed), subcase {subcase_id} retired to CASE_DELETED")

test_delete_published_case_soft_deletes()


# ============================================================
# CLEANUP
# ============================================================

print()
print("CLEANUP: Removing test fixtures...")
try:
    # Each item cleaned up independently — one failure (e.g. an FK this
    # script doesn't know about yet) must not abort the rest of the loop
    # and leak every fixture after it. Bit us once already: a single
    # ml.CaseTrainingRecord FK conflict mid-loop silently left 10 rows
    # behind in the dev DB until they surfaced later in a manual QA pass.
    cleanup_failures = []
    for cid in _cleanup_case_ids:
        try:
            _cleanup_case(cid)
        except Exception as e:
            cleanup_failures.append((cid, str(e)))
    for iid in _cleanup_incident_ids:
        try:
            _delete_incident_row(iid)
        except Exception as e:
            cleanup_failures.append((iid, str(e)))
    if cleanup_failures:
        print(f"Cleanup finished with {len(cleanup_failures)} failure(s):")
        for ident, err in cleanup_failures:
            print(f"   id={ident}: {err}")
    else:
        print("Cleanup complete")
except Exception as e:
    print(f"Cleanup warning: {e}")
    print("(This is not critical — tests still ran)")


# ============================================================
# TEST SUMMARY
# ============================================================

print()
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Total Tests:  {tests_run}")
print(f"Passed:       {tests_passed}")
print(f"Failed:       {tests_failed}")
print()

if tests_failed == 0:
    print("ALL TESTS PASSED.")
    sys.exit(0)
else:
    print(f"{tests_failed} test(s) failed. Please review.")
    sys.exit(1)
