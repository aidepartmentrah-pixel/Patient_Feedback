"""
Test Suite for Action Item Coordination (Iteration 4) — Session 2
Covers supervisor_action_item_db.py (SQL layer) and
supervisor_action_item_service.py (validation/business logic).

Run: python backend/tests/test_aic_s2_supervisor_action_item.py

Fixtures used (verified to exist in the dev DB before writing this file):
- IncidentRequestCaseID 1            (valid case)
- SubcaseID 1600 -> CaseID 1327      (valid subcase, but NOT under case 1 - mismatch test)
- AdminsrationUnit 12                (Frozen = 1 - inactive unit test)
- AdminsrationUnit 1, 2, 3           (Frozen = 0 - valid administrations, siblings)
- AdminsrationUnit 5 (Dept, Parent=1) -> UserID 5 belongs here (descendant-of-1 test)
- AdminsrationUnit 10 (root Admin)   -> UserID 2 belongs here (sibling of 1/2/3, mismatch test)
"""

import sys
sys.path.insert(0, 'backend')

from types import SimpleNamespace

from api_v2.db_layer import supervisor_action_item_db as db
from api_v2.services import supervisor_action_item_service as service
from api_v2.services.supervisor_action_item_service import SupervisorActionItemError

# Test counters
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
                print(f"✅ PASS: {description}")
                return True
            except AssertionError as e:
                tests_failed += 1
                print(f"❌ FAIL: {description}")
                print(f"   Error: {e}")
                return False
            except Exception as e:
                tests_failed += 1
                print(f"❌ ERROR: {description}")
                print(f"   Exception: {e}")
                return False
        return wrapper
    return decorator


def make_user(user_id, roles, allowed_unit_ids):
    return SimpleNamespace(user_id=user_id, roles=roles, allowed_unit_ids=allowed_unit_ids)


print("=" * 70)
print("AIC-S2: SUPERVISOR_ACTION_ITEM_DB.PY + SERVICE TEST SUITE")
print("=" * 70)
print()

CASE_ID = 1
SUBCASE_WRONG_CASE = 1600  # belongs to case 1327, not CASE_ID
FROZEN_UNIT = 12
VALID_UNIT = 1
SIBLING_UNIT = 2
USER_UNDER_UNIT_1 = 5      # AdminsrationUnit 5's ParentID = 1
USER_UNDER_SIBLING_ROOT = 2  # belongs to root unit 10, unrelated to 1/2/3
SUPERVISOR_USER_ID = 1

created_item_ids = []

# ============================================================
# DB LAYER TESTS
# ============================================================

@test("1. db.create_supervisor_action_item - minimal fields")
def t1():
    item_id = db.create_supervisor_action_item(
        incident_request_case_id=CASE_ID,
        target_org_unit_id=VALID_UNIT,
        created_by_user_id=SUPERVISOR_USER_ID,
        created_by_role_code="COMPLAINT_SUPERVISOR",
        description="DB layer smoke test - minimal",
    )
    assert item_id is not None, "Should return a new ActionItemID"
    created_item_ids.append(item_id)
    print(f"   Created ActionItemID={item_id}")

t1()


@test("2. db.get_supervisor_action_item_by_id - returns joined context")
def t2():
    item = db.get_supervisor_action_item_by_id(created_item_ids[0])
    assert item is not None
    assert item["status"] == "PENDING"
    assert item["target_org_unit_name"] is not None, "Org unit name should be joined in"
    assert item["incident_request_case_id"] == CASE_ID

t2()


@test("3. db.get_supervisor_action_items_by_org_units - scoped list")
def t3():
    items = db.get_supervisor_action_items_by_org_units([VALID_UNIT])
    assert any(i["action_item_id"] == created_item_ids[0] for i in items)

    items_other_unit = db.get_supervisor_action_items_by_org_units([SIBLING_UNIT])
    assert not any(i["action_item_id"] == created_item_ids[0] for i in items_other_unit)

t3()


@test("4. db.get_supervisor_action_items_by_org_units - empty list returns []")
def t4():
    assert db.get_supervisor_action_items_by_org_units([]) == []

t4()


@test("5. db.create_audit_log_entry + get_audit_log_for_action_item")
def t5():
    db.create_audit_log_entry(created_item_ids[0], "CREATED", SUPERVISOR_USER_ID, note="test note")
    log = db.get_audit_log_for_action_item(created_item_ids[0])
    assert len(log) == 1
    assert log[0]["action"] == "CREATED"
    assert log[0]["note"] == "test note"

t5()


@test("6. db.set_supervisor_action_item_completed")
def t6():
    item_id = db.create_supervisor_action_item(
        incident_request_case_id=CASE_ID,
        target_org_unit_id=VALID_UNIT,
        created_by_user_id=SUPERVISOR_USER_ID,
        created_by_role_code="COMPLAINT_SUPERVISOR",
        description="DB layer smoke test - for completion",
    )
    created_item_ids.append(item_id)

    ok = db.set_supervisor_action_item_completed(item_id, SUPERVISOR_USER_ID)
    assert ok is True
    item = db.get_supervisor_action_item_by_id(item_id)
    assert item["status"] == "COMPLETED"
    assert item["completed_at"] is not None

t6()


@test("7. db.set_supervisor_action_item_cancelled")
def t7():
    item_id = db.create_supervisor_action_item(
        incident_request_case_id=CASE_ID,
        target_org_unit_id=VALID_UNIT,
        created_by_user_id=SUPERVISOR_USER_ID,
        created_by_role_code="COMPLAINT_SUPERVISOR",
        description="DB layer smoke test - for cancellation",
    )
    created_item_ids.append(item_id)

    ok = db.set_supervisor_action_item_cancelled(item_id, SUPERVISOR_USER_ID)
    assert ok is True
    item = db.get_supervisor_action_item_by_id(item_id)
    assert item["status"] == "CANCELLED"
    assert item["cancelled_at"] is not None

t7()


@test("8. db.get_org_unit_frozen_status - frozen / active / nonexistent")
def t8():
    assert db.get_org_unit_frozen_status(FROZEN_UNIT) is True
    assert db.get_org_unit_frozen_status(VALID_UNIT) is False
    assert db.get_org_unit_frozen_status(-999999) is None

t8()


@test("9. db.get_org_unit_ids_for_user")
def t9():
    units = db.get_org_unit_ids_for_user(USER_UNDER_UNIT_1)
    assert 5 in units

t9()


# ============================================================
# SERVICE LAYER TESTS
# ============================================================

supervisor = make_user(SUPERVISOR_USER_ID, ["COMPLAINT_SUPERVISOR"], None)


@test("10. service.create_action_item - rejects empty description")
def t10():
    try:
        service.create_action_item(
            current_user=supervisor,
            incident_request_case_id=CASE_ID,
            target_org_unit_id=VALID_UNIT,
            description="   ",
        )
        assert False, "Should have raised SupervisorActionItemError"
    except SupervisorActionItemError as e:
        assert "Description" in str(e)

t10()


@test("11. service.create_action_item - rejects nonexistent case")
def t11():
    try:
        service.create_action_item(
            current_user=supervisor,
            incident_request_case_id=-999999,
            target_org_unit_id=VALID_UNIT,
            description="Should fail - bad case",
        )
        assert False, "Should have raised SupervisorActionItemError"
    except SupervisorActionItemError as e:
        assert "does not exist" in str(e)

t11()


@test("12. service.create_action_item - rejects subcase not belonging to case")
def t12():
    try:
        service.create_action_item(
            current_user=supervisor,
            incident_request_case_id=CASE_ID,
            target_org_unit_id=VALID_UNIT,
            description="Should fail - mismatched subcase",
            subcase_id=SUBCASE_WRONG_CASE,
        )
        assert False, "Should have raised SupervisorActionItemError"
    except SupervisorActionItemError as e:
        assert "does not belong to case" in str(e)

t12()


@test("13. service.create_action_item - rejects frozen target unit")
def t13():
    try:
        service.create_action_item(
            current_user=supervisor,
            incident_request_case_id=CASE_ID,
            target_org_unit_id=FROZEN_UNIT,
            description="Should fail - frozen unit",
        )
        assert False, "Should have raised SupervisorActionItemError"
    except SupervisorActionItemError as e:
        assert "frozen" in str(e).lower()

t13()


@test("14. service.create_action_item - rejects nonexistent target unit")
def t14():
    try:
        service.create_action_item(
            current_user=supervisor,
            incident_request_case_id=CASE_ID,
            target_org_unit_id=-999999,
            description="Should fail - bad unit",
        )
        assert False, "Should have raised SupervisorActionItemError"
    except SupervisorActionItemError as e:
        assert "does not exist" in str(e)

t14()


@test("15. service.create_action_item - rejects target user outside target unit")
def t15():
    try:
        service.create_action_item(
            current_user=supervisor,
            incident_request_case_id=CASE_ID,
            target_org_unit_id=SIBLING_UNIT,
            target_user_id=USER_UNDER_SIBLING_ROOT,
            description="Should fail - user 2 is under unit 10, not unit 2",
        )
        assert False, "Should have raised SupervisorActionItemError"
    except SupervisorActionItemError as e:
        assert "does not belong to org unit" in str(e)

t15()


@test("16. service.create_action_item - accepts valid target user within target unit")
def t16():
    item = service.create_action_item(
        current_user=supervisor,
        incident_request_case_id=CASE_ID,
        target_org_unit_id=VALID_UNIT,
        target_user_id=USER_UNDER_UNIT_1,
        description="Should succeed - user 5 is under unit 1",
    )
    assert item is not None
    assert item["status"] == "PENDING"
    assert item["created_by_role_code"] == "COMPLAINT_SUPERVISOR"
    created_item_ids.append(item["action_item_id"])

    log = db.get_audit_log_for_action_item(item["action_item_id"])
    assert len(log) == 1 and log[0]["action"] == "CREATED"

t16()


@test("17. service.list_action_items_for_user - scoped correctly")
def t17():
    scoped_user = make_user(999, ["DEPARTMENT_ADMIN"], [VALID_UNIT])
    items = service.list_action_items_for_user(scoped_user)
    assert any(i["action_item_id"] == created_item_ids[-1] for i in items)

    other_scoped_user = make_user(999, ["DEPARTMENT_ADMIN"], [SIBLING_UNIT])
    items2 = service.list_action_items_for_user(other_scoped_user)
    assert not any(i["action_item_id"] == created_item_ids[-1] for i in items2)

t17()


@test("18. service.complete_action_item - happy path (target unit in scope)")
def t18():
    item = service.create_action_item(
        current_user=supervisor,
        incident_request_case_id=CASE_ID,
        target_org_unit_id=VALID_UNIT,
        description="For completion via service",
    )
    created_item_ids.append(item["action_item_id"])

    target_scoped_user = make_user(USER_UNDER_UNIT_1, ["SECTION_ADMIN"], [VALID_UNIT])
    completed = service.complete_action_item(target_scoped_user, item["action_item_id"])
    assert completed["status"] == "COMPLETED"

    log = db.get_audit_log_for_action_item(item["action_item_id"])
    assert any(entry["action"] == "COMPLETED" for entry in log)

t18()


@test("19. service.complete_action_item - rejects user outside target scope")
def t19():
    item = service.create_action_item(
        current_user=supervisor,
        incident_request_case_id=CASE_ID,
        target_org_unit_id=VALID_UNIT,
        description="For unauthorized completion attempt",
    )
    created_item_ids.append(item["action_item_id"])

    outsider = make_user(999, ["SECTION_ADMIN"], [SIBLING_UNIT])
    try:
        service.complete_action_item(outsider, item["action_item_id"])
        assert False, "Should have raised SupervisorActionItemError"
    except SupervisorActionItemError as e:
        assert "not authorized" in str(e).lower()

t19()


@test("20. service.complete_action_item - rejects wrong status (already completed)")
def t20():
    item = service.create_action_item(
        current_user=supervisor,
        incident_request_case_id=CASE_ID,
        target_org_unit_id=VALID_UNIT,
        description="For double-completion test",
    )
    created_item_ids.append(item["action_item_id"])

    scoped_user = make_user(USER_UNDER_UNIT_1, ["SECTION_ADMIN"], [VALID_UNIT])
    service.complete_action_item(scoped_user, item["action_item_id"])

    try:
        service.complete_action_item(scoped_user, item["action_item_id"])
        assert False, "Should have raised SupervisorActionItemError"
    except SupervisorActionItemError as e:
        assert "must be 'PENDING'" in str(e)

t20()


@test("21. service.cancel_action_item - happy path (creator only)")
def t21():
    item = service.create_action_item(
        current_user=supervisor,
        incident_request_case_id=CASE_ID,
        target_org_unit_id=VALID_UNIT,
        description="For cancellation via service",
    )
    created_item_ids.append(item["action_item_id"])

    cancelled = service.cancel_action_item(supervisor, item["action_item_id"])
    assert cancelled["status"] == "CANCELLED"

    log = db.get_audit_log_for_action_item(item["action_item_id"])
    assert any(entry["action"] == "CANCELLED" for entry in log)

t21()


@test("22. service.cancel_action_item - rejects non-creator")
def t22():
    item = service.create_action_item(
        current_user=supervisor,
        incident_request_case_id=CASE_ID,
        target_org_unit_id=VALID_UNIT,
        description="For unauthorized cancellation attempt",
    )
    created_item_ids.append(item["action_item_id"])

    not_creator = make_user(999, ["SECTION_ADMIN"], [VALID_UNIT])
    try:
        service.cancel_action_item(not_creator, item["action_item_id"])
        assert False, "Should have raised SupervisorActionItemError"
    except SupervisorActionItemError as e:
        assert "Only the creator" in str(e)

t22()


@test("23. service.complete_action_item - nonexistent item raises")
def t23():
    try:
        service.complete_action_item(supervisor, -999999)
        assert False, "Should have raised SupervisorActionItemError"
    except SupervisorActionItemError as e:
        assert "not found" in str(e).lower()

t23()


# ============================================================
# CLEANUP
# ============================================================

print()
print("--- CLEANUP: removing test action items ---")
import pyodbc
from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()
placeholders = ','.join(['?'] * len(created_item_ids)) if created_item_ids else "NULL"
if created_item_ids:
    cursor.execute(
        f"DELETE FROM dbo.APP_SupervisorActionItemAuditLog WHERE ActionItemID IN ({placeholders})",
        created_item_ids
    )
    cursor.execute(
        f"DELETE FROM dbo.APP_SupervisorActionItem WHERE ActionItemID IN ({placeholders})",
        created_item_ids
    )
    conn.commit()
cursor.execute("SELECT COUNT(*) FROM dbo.APP_SupervisorActionItem WHERE ActionItemID IN (" + placeholders + ")", created_item_ids) if created_item_ids else None
remaining = cursor.fetchone()[0] if created_item_ids else 0
cursor.close()
conn.close()
print(f"Cleaned up {len(created_item_ids)} test action item(s). Remaining: {remaining}")


# ============================================================
# TEST SUMMARY
# ============================================================

print()
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Total Tests:  {tests_run}")
print(f"✅ Passed:     {tests_passed}")
print(f"❌ Failed:     {tests_failed}")
print()

if tests_failed == 0:
    print("\U0001F389 ALL TESTS PASSED!")
    exit(0)
else:
    print(f"⚠️  {tests_failed} test(s) failed. Please review and fix.")
    exit(1)
