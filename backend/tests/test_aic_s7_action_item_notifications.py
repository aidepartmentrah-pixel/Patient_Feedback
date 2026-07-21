"""
Test Suite for Action Item Coordination — AIC-S7 Notification Foundation
Covers:
- supervisor_action_item_db.py: acknowledge_supervisor_action_item,
  get_unacknowledged_supervisor_action_items_for_user
- (more sections appended as later AIC-S7 tasks land)

Run: python backend/tests/test_aic_s7_action_item_notifications.py
"""

import sys
import os
sys.path.insert(0, 'backend')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from types import SimpleNamespace

from core.database import get_connection
from api_v2.db_layer import supervisor_action_item_db as supervisor_db
from api_v2.db_layer import action_item_subcase_db as subcase_item_db
from api_v2.db_layer import action_item_change_notice_db as notice_db
from api_v2.services import case_response_service

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


print("=" * 70)
print("AIC-S7: ACTION ITEM NOTIFICATION TEST SUITE")
print("=" * 70)
print()

CASE_ID = 1
VALID_UNIT = 1
SIBLING_UNIT = 2
SUPERVISOR_USER_ID = 1
RECIPIENT_USER_ID = 5  # AdminsrationUnit 5's ParentID = 1 (USER_UNDER_UNIT_1 in AIC-S2 tests)
SUBCASE_ID = 1600  # valid subcase row (used as FK target only, case mismatch irrelevant here)

created_supervisor_item_ids = []
created_subcase_item_ids = []
created_notice_ids = []

# ============================================================
# SECTION 1: supervisor_action_item_db acknowledgment
# ============================================================

@test("1. New supervisor action item starts unacknowledged")
def t1():
    item_id = supervisor_db.create_supervisor_action_item(
        incident_request_case_id=CASE_ID,
        target_org_unit_id=VALID_UNIT,
        target_user_id=RECIPIENT_USER_ID,
        created_by_user_id=SUPERVISOR_USER_ID,
        created_by_role_code="COMPLAINT_SUPERVISOR",
        description="AIC-S7 notification smoke test - direct target user",
    )
    assert item_id is not None
    created_supervisor_item_ids.append(item_id)

    item = supervisor_db.get_supervisor_action_item_by_id(item_id)
    assert item["acknowledged_at"] is None
    assert item["acknowledged_by_user_id"] is None

t1()


@test("2. get_unacknowledged_supervisor_action_items_for_user - direct target match")
def t2():
    items = supervisor_db.get_unacknowledged_supervisor_action_items_for_user(
        user_id=RECIPIENT_USER_ID, org_unit_ids=[]
    )
    assert any(i["action_item_id"] == created_supervisor_item_ids[0] for i in items), \
        "Directly targeted user should see the pending item even with no org_unit_ids"

t2()


@test("3. get_unacknowledged_supervisor_action_items_for_user - other user doesn't see it")
def t3():
    items = supervisor_db.get_unacknowledged_supervisor_action_items_for_user(
        user_id=999999, org_unit_ids=[SIBLING_UNIT]
    )
    assert not any(i["action_item_id"] == created_supervisor_item_ids[0] for i in items)

t3()


@test("4. acknowledge_supervisor_action_item - sets AcknowledgedAt/By, returns True")
def t4():
    ok = supervisor_db.acknowledge_supervisor_action_item(
        created_supervisor_item_ids[0], acknowledged_by_user_id=RECIPIENT_USER_ID
    )
    assert ok is True

    item = supervisor_db.get_supervisor_action_item_by_id(created_supervisor_item_ids[0])
    assert item["acknowledged_at"] is not None
    assert item["acknowledged_by_user_id"] == RECIPIENT_USER_ID

t4()


@test("5. acknowledge_supervisor_action_item - second call returns False (already acked)")
def t5():
    ok = supervisor_db.acknowledge_supervisor_action_item(
        created_supervisor_item_ids[0], acknowledged_by_user_id=RECIPIENT_USER_ID
    )
    assert ok is False

t5()


@test("6. Acknowledged item drops out of unacknowledged query")
def t6():
    items = supervisor_db.get_unacknowledged_supervisor_action_items_for_user(
        user_id=RECIPIENT_USER_ID, org_unit_ids=[]
    )
    assert not any(i["action_item_id"] == created_supervisor_item_ids[0] for i in items)

t6()


@test("7. Org-unit-only assignment (TargetUserID NULL) visible via org_unit_ids match")
def t7():
    item_id = supervisor_db.create_supervisor_action_item(
        incident_request_case_id=CASE_ID,
        target_org_unit_id=VALID_UNIT,
        created_by_user_id=SUPERVISOR_USER_ID,
        created_by_role_code="COMPLAINT_SUPERVISOR",
        description="AIC-S7 notification smoke test - org-unit-only target",
    )
    created_supervisor_item_ids.append(item_id)

    items_in_unit = supervisor_db.get_unacknowledged_supervisor_action_items_for_user(
        user_id=RECIPIENT_USER_ID, org_unit_ids=[VALID_UNIT]
    )
    assert any(i["action_item_id"] == item_id for i in items_in_unit)

    items_other_unit = supervisor_db.get_unacknowledged_supervisor_action_items_for_user(
        user_id=999999, org_unit_ids=[SIBLING_UNIT]
    )
    assert not any(i["action_item_id"] == item_id for i in items_other_unit)

t7()


@test("8. acknowledge_supervisor_action_item - nonexistent ID returns False")
def t8():
    ok = supervisor_db.acknowledge_supervisor_action_item(-999, acknowledged_by_user_id=RECIPIENT_USER_ID)
    assert ok is False

t8()


# ============================================================
# SECTION 2: action_item_change_notice_db
# ============================================================

@test("9. upsert_change_notice - first call inserts a new pending notice")
def t9():
    item_id = subcase_item_db.create_action_item(
        subcase_id=SUBCASE_ID,
        title="Original title",
        description="Original description",
        due_date=None,
        created_by_user_id=RECIPIENT_USER_ID,
        initial_status="DRAFT",
    )
    assert item_id is not None
    created_subcase_item_ids.append(item_id)

    notice_id = notice_db.upsert_change_notice(
        action_item_id=item_id,
        recipient_user_id=RECIPIENT_USER_ID,
        old_title="Original title", new_title="Changed title",
        old_description="Original description", new_description="Changed description",
        old_due_date=None, new_due_date=None,
        changed_by_user_id=SUPERVISOR_USER_ID,
    )
    assert notice_id is not None
    created_notice_ids.append(notice_id)

    notice = notice_db.get_change_notice_by_id(notice_id)
    assert notice["old_title"] == "Original title"
    assert notice["new_title"] == "Changed title"
    assert notice["acknowledged_at"] is None

t9()


@test("10. upsert_change_notice - second pre-ack change UPDATEs same row, keeps original Old*")
def t10():
    item_id = created_subcase_item_ids[0]
    notice_id_again = notice_db.upsert_change_notice(
        action_item_id=item_id,
        recipient_user_id=RECIPIENT_USER_ID,
        old_title="Changed title", new_title="Changed again title",
        old_description="Changed description", new_description="Changed again description",
        old_due_date=None, new_due_date=None,
        changed_by_user_id=SUPERVISOR_USER_ID,
    )
    assert notice_id_again == created_notice_ids[0], "Should reuse the same pending NoticeID, not create a second row"

    notice = notice_db.get_change_notice_by_id(notice_id_again)
    assert notice["old_title"] == "Original title", "Old* baseline must stay the first pre-change value"
    assert notice["new_title"] == "Changed again title", "New* must reflect the latest change"

t10()


@test("11. get_unacknowledged_change_notices_for_user - recipient sees pending notice")
def t11():
    notices = notice_db.get_unacknowledged_change_notices_for_user(RECIPIENT_USER_ID)
    assert any(n["notice_id"] == created_notice_ids[0] for n in notices)

t11()


@test("12. get_unacknowledged_change_notices_for_user - other user doesn't see it")
def t12():
    notices = notice_db.get_unacknowledged_change_notices_for_user(999999)
    assert not any(n["notice_id"] == created_notice_ids[0] for n in notices)

t12()


@test("13. acknowledge_change_notice - sets ack fields, returns True, then False on retry")
def t13():
    ok = notice_db.acknowledge_change_notice(created_notice_ids[0], acknowledged_by_user_id=RECIPIENT_USER_ID)
    assert ok is True

    notice = notice_db.get_change_notice_by_id(created_notice_ids[0])
    assert notice["acknowledged_at"] is not None
    assert notice["acknowledged_by_user_id"] == RECIPIENT_USER_ID

    ok_again = notice_db.acknowledge_change_notice(created_notice_ids[0], acknowledged_by_user_id=RECIPIENT_USER_ID)
    assert ok_again is False

t13()


@test("14. After acknowledgment, a new change starts a fresh pending notice (not reused)")
def t14():
    item_id = created_subcase_item_ids[0]
    new_notice_id = notice_db.upsert_change_notice(
        action_item_id=item_id,
        recipient_user_id=RECIPIENT_USER_ID,
        old_title="Changed again title", new_title="Third title",
        old_description="Changed again description", new_description="Third description",
        old_due_date=None, new_due_date=None,
        changed_by_user_id=SUPERVISOR_USER_ID,
    )
    assert new_notice_id != created_notice_ids[0], "Acknowledged notice must not be reused"
    created_notice_ids.append(new_notice_id)

t14()


@test("15. Acknowledged notice no longer appears in unacknowledged query")
def t15():
    notices = notice_db.get_unacknowledged_change_notices_for_user(RECIPIENT_USER_ID)
    notice_ids_visible = {n["notice_id"] for n in notices}
    assert created_notice_ids[0] not in notice_ids_visible, "First (acknowledged) notice should be gone"
    assert created_notice_ids[1] in notice_ids_visible, "Second (fresh) notice should be visible"

t15()


# ============================================================
# SECTION 3: case_response_service._upsert_action_items trigger wiring
#
# _upsert_action_items deletes any existing item under the subcase NOT
# referenced in the call's payload -- exactly like a real caller, which
# always submits the subcase's FULL current item list. These tests do
# the same: fetch what's currently there, carry it through unchanged,
# and only modify the one item under test.
# ============================================================

def _full_payload_with_override(override_item_id, override_title, override_description):
    payload = []
    for existing in subcase_item_db.get_action_items_by_subcase(SUBCASE_ID):
        if existing["action_item_id"] == override_item_id:
            payload.append({
                "action_item_id": override_item_id,
                "title": override_title,
                "description": override_description,
            })
        else:
            payload.append({
                "action_item_id": existing["action_item_id"],
                "title": existing["title"],
                "description": existing["description"],
            })
    return payload


@test("16. Same-user edit does NOT create a change notice")
def t16():
    item_id = subcase_item_db.create_action_item(
        subcase_id=SUBCASE_ID,
        title="Self-edit item",
        description="desc",
        due_date=None,
        created_by_user_id=RECIPIENT_USER_ID,
        initial_status="DRAFT",
    )
    created_subcase_item_ids.append(item_id)

    creator_user = SimpleNamespace(user_id=RECIPIENT_USER_ID)
    case_response_service._upsert_action_items(
        SUBCASE_ID,
        _full_payload_with_override(item_id, "Self-edit item CHANGED", "desc"),
        creator_user
    )

    notices = notice_db.get_unacknowledged_change_notices_for_user(RECIPIENT_USER_ID)
    assert not any(n["action_item_id"] == item_id for n in notices), \
        "Editing your own item must not generate a notice"

t16()


@test("17. Cross-user edit with NO actual field change does NOT create a notice")
def t17():
    item_id = subcase_item_db.create_action_item(
        subcase_id=SUBCASE_ID,
        title="Untouched item",
        description="desc",
        due_date=None,
        created_by_user_id=RECIPIENT_USER_ID,
        initial_status="DRAFT",
    )
    created_subcase_item_ids.append(item_id)

    supervisor_user = SimpleNamespace(user_id=SUPERVISOR_USER_ID)
    case_response_service._upsert_action_items(
        SUBCASE_ID,
        _full_payload_with_override(item_id, "Untouched item", "desc"),
        supervisor_user
    )

    notices = notice_db.get_unacknowledged_change_notices_for_user(RECIPIENT_USER_ID)
    assert not any(n["action_item_id"] == item_id for n in notices), \
        "Re-submitting identical values must not generate a notice"

t17()


@test("18. Cross-user edit WITH a real change creates a notice with correct old/new values")
def t18():
    item_id = subcase_item_db.create_action_item(
        subcase_id=SUBCASE_ID,
        title="Section's suggestion",
        description="Section's description",
        due_date=None,
        created_by_user_id=RECIPIENT_USER_ID,
        initial_status="DRAFT",
    )
    created_subcase_item_ids.append(item_id)

    supervisor_user = SimpleNamespace(user_id=SUPERVISOR_USER_ID)
    case_response_service._upsert_action_items(
        SUBCASE_ID,
        _full_payload_with_override(item_id, "Department's revised title", "Section's description"),
        supervisor_user
    )

    notices = notice_db.get_unacknowledged_change_notices_for_user(RECIPIENT_USER_ID)
    match = next((n for n in notices if n["action_item_id"] == item_id), None)
    assert match is not None, "Cross-user edit with a real change must generate a notice"
    created_notice_ids.append(match["notice_id"])

    assert match["old_title"] == "Section's suggestion"
    assert match["new_title"] == "Department's revised title"
    assert match["changed_by_user_id"] == SUPERVISOR_USER_ID

    updated_item = next(i for i in subcase_item_db.get_action_items_by_subcase(SUBCASE_ID) if i["action_item_id"] == item_id)
    assert updated_item["title"] == "Department's revised title", "Underlying action item must still be updated as before"

t18()


@test("19. A second cross-user edit before acknowledgment reuses the same pending notice")
def t19():
    item_id = created_subcase_item_ids[-1]
    notice_before = notice_db.get_unacknowledged_change_notices_for_user(RECIPIENT_USER_ID)
    pending = next(n for n in notice_before if n["action_item_id"] == item_id)

    supervisor_user = SimpleNamespace(user_id=SUPERVISOR_USER_ID)
    case_response_service._upsert_action_items(
        SUBCASE_ID,
        _full_payload_with_override(item_id, "Department's second revision", "Section's description"),
        supervisor_user
    )

    notices_after = notice_db.get_unacknowledged_change_notices_for_user(RECIPIENT_USER_ID)
    match = next(n for n in notices_after if n["action_item_id"] == item_id)
    assert match["notice_id"] == pending["notice_id"], "Should reuse the same pending notice, not create a second one"
    assert match["old_title"] == "Section's suggestion", "Old* baseline must remain the original pre-change value"
    assert match["new_title"] == "Department's second revision"

t19()


# ============================================================
# CLEANUP
# ============================================================

conn = get_connection()
cursor = conn.cursor()

if created_notice_ids:
    placeholders = ','.join(['?'] * len(created_notice_ids))
    cursor.execute(
        f"DELETE FROM dbo.APP_SubcaseActionItemChangeNotice WHERE NoticeID IN ({placeholders})",
        created_notice_ids
    )
    conn.commit()

if created_subcase_item_ids:
    placeholders = ','.join(['?'] * len(created_subcase_item_ids))
    cursor.execute(
        f"DELETE FROM dbo.APP_SubcaseActionItem WHERE ActionItemID IN ({placeholders})",
        created_subcase_item_ids
    )
    conn.commit()

if created_supervisor_item_ids:
    placeholders = ','.join(['?'] * len(created_supervisor_item_ids))
    cursor.execute(
        f"DELETE FROM dbo.APP_SupervisorActionItem WHERE ActionItemID IN ({placeholders})",
        created_supervisor_item_ids
    )
    conn.commit()

cursor.close()
conn.close()
print(f"Cleaned up {len(created_notice_ids)} notice(s), {len(created_subcase_item_ids)} subcase action item(s), "
      f"{len(created_supervisor_item_ids)} supervisor action item(s).")

print()
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Total Tests:  {tests_run}")
print(f"Passed:       {tests_passed}")
print(f"Failed:       {tests_failed}")
print()

if tests_failed == 0:
    print("ALL TESTS PASSED")
    sys.exit(0)
else:
    print(f"{tests_failed} test(s) failed.")
    sys.exit(1)
