"""
Test suite for the skip-level (no-Department) section workflow escalation
fix. Confirms the originally reported bug is fixed: a case routed to a
Section with no real Department above it (attaches directly to an
Administration -- e.g. "OPD") no longer gets orphaned at a phantom
'SECTION_ACCEPTED_PENDING_DEPT' stage that nobody can ever act on.

Deterministic by construction: creates its own fresh skip-level section
(parented directly under a real, existing Administration) and its own fresh
normal section (parented under a real, existing Department) as throwaway
fixtures, rather than depending on a real section like "OPD" happening to
exist in whatever DB this runs against. Every fixture (org units, incident,
case, subcase) is created at the start and torn down at the end, so this
suite gives the same result on every run regardless of ambient DB state.

Run: python backend/tests/test_skip_level_workflow_escalation.py
"""

import sys
import os
# Absolute, cwd-independent path setup (some sibling modules, e.g.
# incident_parent.py, import via `from backend.X import Y` absolute paths,
# which requires the repo root -- not just backend/ -- on sys.path; see the
# identical note in test_case_retarget_and_lifecycle.py).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_BACKEND_DIR)
sys.path.insert(0, _BACKEND_DIR)
sys.path.insert(0, _REPO_ROOT)

from datetime import date, datetime
from types import SimpleNamespace

from core.database import get_connection
from api.services import section_admin_creator_service
from api.db_layer.incident_case import create_incident_case
from api.db_layer.incident_parent import create_incident_parent, assign_case_to_incident
from api_v2.db_layer import administrative_subcase_db as subcase_db
from api_v2.services import case_creation_service, case_response_service

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


def _user(role_code='COMPLAINT_SUPERVISOR', user_id=1):
    return SimpleNamespace(user_id=user_id, scopes=[SimpleNamespace(role_code=role_code)])


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


def _get_real_administration_id():
    uid = _fetch_scalar("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Type = 323 ORDER BY UniqueID")
    assert uid is not None, "Need at least 1 real Administration (Type=323) org unit in this DB"
    return uid


def _get_real_department_id():
    uid = _fetch_scalar("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Type = 325 ORDER BY UniqueID")
    assert uid is not None, "Need at least 1 real Department (Type=325) org unit in this DB"
    return uid


def _create_test_section(parent_unit_id, name):
    conn = get_connection()
    try:
        section_id = section_admin_creator_service.create_section_core(
            conn, section_name=name, parent_unit_id=parent_unit_id, created_by_user_id=1
        )
        conn.commit()
        return section_id
    finally:
        conn.close()


def _delete_test_section(section_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM dbo.AdminsrationUnit WHERE UniqueID = ?", (section_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def _base_case_payload(*, issuing_org_unit_id, building_id, source_id):
    return {
        "ComplaintText": "Test complaint for skip-level workflow escalation suite",
        "ImmediateAction": "",
        "TakenAction": "",
        "FeedbackRecievedDate": date.today(),
        "IncidentDate": date.today(),
        "PatientName": "Test Patient",
        "IssuingOrgUnitID": issuing_org_unit_id,
        "CreatedByUserID": 1,
        "isINPatient": 1,
        "IsMorbidity": 0,
        "ClinicalRiskTypeID": 1,  # Ordinary -- NOT a complaint auto-routed to Patient Services
        "FeedbackIntentTypeID": None,
        "BuildingID": building_id,
        "DomainID": None,
        "CategoryID": None,
        "SubCategoryID": None,
        "ClassificationID": None,
        "SeverityID": None,
        "StageID": None,
        "HarmLevelID": None,
        "CaseStatusID": 1,  # Open
        "SourceID": source_id,
        "ExplanationStatusID": 4,  # No Explanation Needed
        "RequiresExplanation": 0,
        "RecordTypeID": 1,  # Ordinary complaint, not a Notice
    }


def _make_case(issuing_org_unit_id, building_id, source_id):
    """Create a minimal, real incident+case pair. Returns (incident_id, case_id)."""
    incident_id = create_incident_parent({
        "patient_name": "Test Patient",
        "primary_doctor_name": None,
        "primary_worker_name": None,
        "feedback_intent_type_id": None,
        "issuing_org_unit_id": issuing_org_unit_id,
        "complaint_summary": "Test complaint for skip-level workflow escalation suite",
        "building_id": building_id,
        "is_inpatient": True,
        "created_by_user_id": 1,
    })
    case_id = create_incident_case(_base_case_payload(
        issuing_org_unit_id=issuing_org_unit_id, building_id=building_id, source_id=source_id
    ))
    assign_case_to_incident(case_id, incident_id)
    return incident_id, case_id


def _make_subcase(case_id, target_org_unit_id, initial_status='SUBMITTED_TO_SECTION'):
    subcase_id = subcase_db.create_subcase(
        case_type='INCIDENT_RESPONSE',
        incident_id=case_id,
        seasonal_report_id=None,
        target_org_unit_id=target_org_unit_id,
        created_by_user_id=1,
        initial_status=initial_status,
    )
    assert subcase_id is not None, "create_subcase returned None"
    return subcase_id


def _cleanup_case(case_id, incident_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT SubcaseID FROM dbo.APP_AdministrativeSubcase WHERE IncidentRequestCaseID = ?", (case_id,))
        subcase_ids = [r.SubcaseID for r in cursor.fetchall()]
        for sid in subcase_ids:
            cursor.execute("DELETE FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (sid,))
        for sid in subcase_ids:
            cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (sid,))
        cursor.execute("DELETE FROM dbo.APP_IncidentCaseTargetDepartment WHERE IncidentRequestCaseID = ?", (case_id,))
        cursor.execute("DELETE FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (case_id,))
        cursor.execute("DELETE FROM dbo.APP_Incident WHERE incident_id = ?", (incident_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def _get_status(subcase_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT Status FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
        row = cursor.fetchone()
        return row.Status if row else None
    finally:
        cursor.close()
        conn.close()


def _get_deadline(subcase_id, column):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT {column} FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
        row = cursor.fetchone()
        return getattr(row, column) if row else None
    finally:
        cursor.close()
        conn.close()


# ============================================================
# SETUP
# ============================================================

print("=" * 70)
print("SKIP-LEVEL (NO-DEPARTMENT) SECTION WORKFLOW ESCALATION TEST SUITE")
print("=" * 70)
print()

real_admin_id = _get_real_administration_id()
real_dept_id = _get_real_department_id()
building_id = _fetch_scalar("SELECT TOP 1 BuildingID FROM dbo.APP_LOOKUP_BUILDING")
source_id = _fetch_scalar("SELECT TOP 1 SourceID FROM dbo.APP_LOOKUP_SOURCE")
assert building_id is not None, "Need at least 1 row in APP_LOOKUP_BUILDING to run these tests"
assert source_id is not None, "Need at least 1 row in APP_LOOKUP_SOURCE to run these tests"

skip_level_section_id = _create_test_section(real_admin_id, f"TEST SKIP-LEVEL SECTION {os.getpid()}")
normal_section_id = _create_test_section(real_dept_id, f"TEST NORMAL SECTION {os.getpid()}")

print(f"Fixture: skip-level section id={skip_level_section_id} (parent Administration id={real_admin_id})")
print(f"Fixture: normal section id={normal_section_id} (parent Department id={real_dept_id})")
print()

_cleanup_case_ids = []  # list of (case_id, incident_id)


def _new_case(target_org_unit_id, initial_status='SUBMITTED_TO_SECTION'):
    incident_id, case_id = _make_case(target_org_unit_id, building_id, source_id)
    _cleanup_case_ids.append((case_id, incident_id))
    subcase_id = _make_subcase(case_id, target_org_unit_id, initial_status)
    return subcase_id


# ============================================================
# 1. ROUTING: the literal, direct proof of the originally reported bug
# ============================================================

@test("1a. Skip-level section accept -> lands DIRECTLY on DEPT_ACCEPTED_PENDING_ADMIN (not the phantom dept stage)")
def test_skip_level_accept_routes_to_administration():
    subcase_id = _new_case(skip_level_section_id)
    case_response_service.accept_section_complaint(subcase_id, _user())
    status = _get_status(subcase_id)
    assert status == 'DEPT_ACCEPTED_PENDING_ADMIN', f"expected DEPT_ACCEPTED_PENDING_ADMIN, got {status!r}"


@test("1b. Normal section accept -> still lands on SECTION_ACCEPTED_PENDING_DEPT (no regression)")
def test_normal_accept_unchanged():
    subcase_id = _new_case(normal_section_id)
    case_response_service.accept_section_complaint(subcase_id, _user())
    status = _get_status(subcase_id)
    assert status == 'SECTION_ACCEPTED_PENDING_DEPT', f"expected SECTION_ACCEPTED_PENDING_DEPT, got {status!r}"


# ============================================================
# 2. INBOX VISIBILITY: the core "orphaned case" regression check
# ============================================================

@test("2a. Skip-level case after section accept: visible to Administration inbox, ABSENT from Department inbox")
def test_skip_level_visible_only_to_administration():
    subcase_id = _new_case(skip_level_section_id)
    case_response_service.accept_section_complaint(subcase_id, _user())

    dept_pending_ids = {s['subcase_id'] for s in subcase_db.get_subcases_pending_for_department()}
    admin_pending_ids = {s['subcase_id'] for s in subcase_db.get_subcases_pending_for_administration()}

    assert subcase_id not in dept_pending_ids, "skip-level case must NEVER appear in the Department pending queue"
    assert subcase_id in admin_pending_ids, "skip-level case must appear in the Administration pending queue"


@test("2b. Normal case after section accept: visible to Department inbox, absent from Administration inbox")
def test_normal_visible_only_to_department():
    subcase_id = _new_case(normal_section_id)
    case_response_service.accept_section_complaint(subcase_id, _user())

    dept_pending_ids = {s['subcase_id'] for s in subcase_db.get_subcases_pending_for_department()}
    admin_pending_ids = {s['subcase_id'] for s in subcase_db.get_subcases_pending_for_administration()}

    assert subcase_id in dept_pending_ids, "normal case must appear in the Department pending queue"
    assert subcase_id not in admin_pending_ids, "normal case must not yet appear in the Administration pending queue"


# ============================================================
# 3. DEADLINE SEEDING: prerequisite for auto-force-close to ever fire
# ============================================================

@test("3a. Skip-level section accept seeds AdministrationDeadlineAt (not DepartmentDeadlineAt)")
def test_skip_level_seeds_administration_deadline():
    subcase_id = _new_case(skip_level_section_id)
    case_response_service.accept_section_complaint(subcase_id, _user())
    admin_deadline = _get_deadline(subcase_id, 'AdministrationDeadlineAt')
    dept_deadline = _get_deadline(subcase_id, 'DepartmentDeadlineAt')
    assert admin_deadline is not None, "AdministrationDeadlineAt must be seeded for a skip-level case"
    assert dept_deadline is None, "DepartmentDeadlineAt must stay NULL for a skip-level case (no such stage)"


@test("3b. Normal section accept seeds DepartmentDeadlineAt")
def test_normal_seeds_department_deadline():
    subcase_id = _new_case(normal_section_id)
    case_response_service.accept_section_complaint(subcase_id, _user())
    dept_deadline = _get_deadline(subcase_id, 'DepartmentDeadlineAt')
    assert dept_deadline is not None, "DepartmentDeadlineAt must be seeded for a normal case"


# ============================================================
# 4. ON-BEHALF FILL
# ============================================================

@test("4a. fill_explanation_on_behalf(level='department') is REJECTED for a skip-level subcase")
def test_fill_department_rejected_for_skip_level():
    subcase_id = _new_case(skip_level_section_id)
    case_response_service.fill_explanation_on_behalf(subcase_id, 'section', 'section text', _user())
    try:
        case_response_service.fill_explanation_on_behalf(subcase_id, 'department', 'dept text', _user())
        raise AssertionError("expected an HTTPException rejecting the department-level fill, got none")
    except Exception as e:
        detail = getattr(e, 'detail', str(e))
        assert 'department' in str(detail).lower(), f"unexpected error for department fill rejection: {detail}"


@test("4b. fill_explanation_on_behalf(level='administration') unlocks directly after section for a skip-level subcase")
def test_fill_administration_unlocked_after_section_for_skip_level():
    subcase_id = _new_case(skip_level_section_id)
    case_response_service.fill_explanation_on_behalf(subcase_id, 'section', 'section text', _user())
    # Must NOT raise -- administration is the very next applicable level for a skip-level case
    case_response_service.fill_explanation_on_behalf(subcase_id, 'administration', 'admin text', _user())
    status = _get_status(subcase_id)
    assert status in ('WAITING_PATIENT_SERVICES_DECISION', 'ADMIN_APPROVED'), f"unexpected status after administration fill: {status!r}"


@test("4c. fill_explanation_on_behalf(level='department') still works normally for a normal (non-skip-level) subcase")
def test_fill_department_works_for_normal_section():
    subcase_id = _new_case(normal_section_id)
    case_response_service.fill_explanation_on_behalf(subcase_id, 'section', 'section text', _user())
    case_response_service.fill_explanation_on_behalf(subcase_id, 'department', 'dept text', _user())
    status = _get_status(subcase_id)
    assert status == 'DEPT_ACCEPTED_PENDING_ADMIN', f"expected DEPT_ACCEPTED_PENDING_ADMIN, got {status!r}"


# ============================================================
# 5. FORCE-CLOSE (TIMEOUT) VISIBILITY
# ============================================================

@test("5a. Skip-level section force-close (timeout) surfaces to Administration, NOT Department")
def test_force_closed_at_section_skip_level_visible_to_administration():
    subcase_id = _new_case(skip_level_section_id)
    ok = subcase_db.force_close_section(subcase_id, auto_text="Auto force-closed (test)")
    assert ok, "force_close_section returned False"
    status = _get_status(subcase_id)
    assert status == 'FORCE_CLOSED_AT_SECTION', f"expected FORCE_CLOSED_AT_SECTION, got {status!r}"

    dept_pending_ids = {s['subcase_id'] for s in subcase_db.get_subcases_pending_for_department()}
    admin_pending_ids = {s['subcase_id'] for s in subcase_db.get_subcases_pending_for_administration()}
    assert subcase_id not in dept_pending_ids, "skip-level force-closed case must NEVER surface to Department"
    assert subcase_id in admin_pending_ids, "skip-level force-closed case must surface to Administration"


@test("5b. Normal section force-close (timeout) still surfaces to Department (no regression)")
def test_force_closed_at_section_normal_visible_to_department():
    subcase_id = _new_case(normal_section_id)
    subcase_db.force_close_section(subcase_id, auto_text="Auto force-closed (test)")
    dept_pending_ids = {s['subcase_id'] for s in subcase_db.get_subcases_pending_for_department()}
    admin_pending_ids = {s['subcase_id'] for s in subcase_db.get_subcases_pending_for_administration()}
    assert subcase_id in dept_pending_ids, "normal force-closed case must surface to Department"
    assert subcase_id not in admin_pending_ids, "normal force-closed case must not surface to Administration yet"


# ============================================================
# 6. FORCE-CLOSE COMPLETION (only applicable levels required)
# ============================================================

@test("6. force_close_subcase completes DIRECTLY (no phantom department requirement) once section+administration are filled for a skip-level case")
def test_force_close_completion_skip_level_no_department_required():
    subcase_id = _new_case(skip_level_section_id)
    case_response_service.fill_explanation_on_behalf(subcase_id, 'section', 'section text', _user())
    # Force-close now, with only section filled -- department is inapplicable,
    # so completeness should NOT depend on it, only on administration.
    result_status = case_response_service.force_close_subcase(subcase_id, "manual close (test)", _user())
    # After force close, administration is still unfilled -> DRAFT expected.
    assert result_status == 'FORCE_CLOSED_DRAFT', f"expected FORCE_CLOSED_DRAFT before administration is filled, got {result_status!r}"

    case_response_service.fill_explanation_on_behalf(subcase_id, 'administration', 'admin text', _user())
    case_response_service.complete_force_closed_draft(subcase_id, _user())
    status = _get_status(subcase_id)
    assert status == 'FORCE_CLOSED_COMPLETE', f"expected FORCE_CLOSED_COMPLETE (department was never required), got {status!r}"


# ============================================================
# 7. ADMINISTRATION-REJECT TARGET
# ============================================================

@test("7a. Administration reject on a skip-level case returns DIRECTLY to Section (not the phantom department stage)")
def test_admin_reject_skip_level_returns_to_section():
    subcase_id = _new_case(skip_level_section_id)
    case_response_service.accept_section_complaint(subcase_id, _user())  # -> DEPT_ACCEPTED_PENDING_ADMIN
    case_response_service.reject_administration(subcase_id, "not good enough (test)", _user())
    status = _get_status(subcase_id)
    assert status == 'RETURNED_TO_SECTION_FOR_REVISION', f"expected RETURNED_TO_SECTION_FOR_REVISION, got {status!r}"


@test("7b. Administration reject on a normal case still returns to Department (no regression)")
def test_admin_reject_normal_returns_to_department():
    subcase_id = _new_case(normal_section_id)
    case_response_service.accept_section_complaint(subcase_id, _user())          # -> SECTION_ACCEPTED_PENDING_DEPT
    case_response_service.approve_department(subcase_id, _user())                # -> DEPT_ACCEPTED_PENDING_ADMIN
    case_response_service.reject_administration(subcase_id, "not good enough (test)", _user())
    status = _get_status(subcase_id)
    assert status == 'RETURNED_TO_DEPT_FOR_REVISION', f"expected RETURNED_TO_DEPT_FOR_REVISION, got {status!r}"


# ============================================================
# RUN
# ============================================================

def run_all_tests():
    tests = [
        test_skip_level_accept_routes_to_administration,
        test_normal_accept_unchanged,
        test_skip_level_visible_only_to_administration,
        test_normal_visible_only_to_department,
        test_skip_level_seeds_administration_deadline,
        test_normal_seeds_department_deadline,
        test_fill_department_rejected_for_skip_level,
        test_fill_administration_unlocked_after_section_for_skip_level,
        test_fill_department_works_for_normal_section,
        test_force_closed_at_section_skip_level_visible_to_administration,
        test_force_closed_at_section_normal_visible_to_department,
        test_force_close_completion_skip_level_no_department_required,
        test_admin_reject_skip_level_returns_to_section,
        test_admin_reject_normal_returns_to_department,
    ]
    for t in tests:
        t()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total: {tests_run}  Passed: {tests_passed}  Failed: {tests_failed}")
    return tests_failed == 0


if __name__ == "__main__":
    success = False
    try:
        success = run_all_tests()
    finally:
        print()
        print("Cleaning up fixtures...")
        for case_id, incident_id in _cleanup_case_ids:
            try:
                _cleanup_case(case_id, incident_id)
            except Exception as e:
                print(f"  [Cleanup Warning] case {case_id}: {e}")
        try:
            _delete_test_section(skip_level_section_id)
            _delete_test_section(normal_section_id)
            print(f"  Deleted test sections {skip_level_section_id}, {normal_section_id}")
        except Exception as e:
            print(f"  [Cleanup Warning] test sections: {e}")

    sys.exit(0 if success else 1)
