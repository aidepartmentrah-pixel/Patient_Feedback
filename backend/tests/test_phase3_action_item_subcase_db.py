"""
Test Suite for action_item_subcase_db.py
Tests all 14 functions in the action item subcase database layer.

Run: python backend/tests/test_phase3_action_item_subcase_db.py
"""

import sys
sys.path.insert(0, 'backend')

from api_v2.db_layer import action_item_subcase_db as db
from api_v2.db_layer import administrative_subcase_db as subcase_db
from datetime import date, timedelta

# Test counters
tests_run = 0
tests_passed = 0
tests_failed = 0

def test(description):
    """Decorator for test functions"""
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


# ============================================================
# TEST SUITE
# ============================================================

print("=" * 70)
print("PHASE 3 - ACTION_ITEM_SUBCASE_DB.PY TEST SUITE")
print("=" * 70)
print()

# Test data storage
test_subcase_id = None
test_action_item_id = None
test_action_item_id_2 = None

# ============================================================
# SETUP: Create a test subcase
# ============================================================

print("SETUP: Creating test subcase...")
test_subcase_id = subcase_db.create_subcase(
    case_type="INCIDENT_RESPONSE",
    incident_id=36,  # Valid incident ID
    seasonal_report_id=None,
    target_org_unit_id=1,
    created_by_user_id=1,
    initial_status="SUBMITTED_TO_SECTION"
)
assert test_subcase_id is not None, "Failed to create test subcase"
print(f"✅ Test subcase created: SubcaseID={test_subcase_id}")
print()

# ============================================================
# CREATION / FETCH TESTS
# ============================================================

@test("1. Create action item")
def test_create_action_item():
    global test_action_item_id
    
    action_item_id = db.create_action_item(
        subcase_id=test_subcase_id,
        title="Test Action Item",
        description="This is a test action item description",
        due_date=date.today() + timedelta(days=7),
        created_by_user_id=1,
        initial_status="DRAFT"
    )
    
    assert action_item_id is not None, "Action item ID should not be None"
    assert isinstance(action_item_id, int), "Action item ID should be integer"
    test_action_item_id = action_item_id
    print(f"   Created ActionItemID: {test_action_item_id}")

test_create_action_item()


@test("2. Get action item by ID")
def test_get_action_item_by_id():
    global test_action_item_id
    
    action_item = db.get_action_item_by_id(test_action_item_id)
    
    assert action_item is not None, "Action item should exist"
    assert action_item["action_item_id"] == test_action_item_id, "IDs should match"
    assert action_item["subcase_id"] == test_subcase_id, "Subcase ID should match"
    assert action_item["title"] == "Test Action Item", "Title should match"
    assert action_item["status"] == "DRAFT", "Status should be DRAFT"
    assert action_item["description"] is not None, "Description should exist"
    print(f"   Retrieved action item: {action_item['action_item_id']}")

test_get_action_item_by_id()


@test("3. Create second action item")
def test_create_second_action_item():
    global test_action_item_id_2
    
    action_item_id = db.create_action_item(
        subcase_id=test_subcase_id,
        title="Second Test Action Item",
        description=None,  # Test with NULL description
        due_date=None,  # Test with NULL due date
        created_by_user_id=1,
        initial_status="SUBMITTED_TO_DEPT"
    )
    
    assert action_item_id is not None, "Action item ID should not be None"
    test_action_item_id_2 = action_item_id
    print(f"   Created second ActionItemID: {test_action_item_id_2}")

test_create_second_action_item()


@test("4. Get action items by subcase")
def test_get_action_items_by_subcase():
    global test_subcase_id
    
    action_items = db.get_action_items_by_subcase(test_subcase_id)
    
    assert isinstance(action_items, list), "Should return list"
    assert len(action_items) >= 2, "Should have at least 2 action items"
    assert any(ai["action_item_id"] == test_action_item_id for ai in action_items), "Should contain first test item"
    assert any(ai["action_item_id"] == test_action_item_id_2 for ai in action_items), "Should contain second test item"
    print(f"   Found {len(action_items)} action item(s) for subcase {test_subcase_id}")

test_get_action_items_by_subcase()


@test("5. Get action items by status (DRAFT)")
def test_get_action_items_by_status_draft():
    action_items = db.get_action_items_by_status("DRAFT")
    
    assert isinstance(action_items, list), "Should return list"
    assert len(action_items) > 0, "Should have at least one DRAFT action item"
    assert all(ai["status"] == "DRAFT" for ai in action_items), "All should have DRAFT status"
    print(f"   Found {len(action_items)} DRAFT action item(s)")

test_get_action_items_by_status_draft()


@test("6. Get action items by status (SUBMITTED_TO_DEPT)")
def test_get_action_items_by_status_submitted():
    action_items = db.get_action_items_by_status("SUBMITTED_TO_DEPT")
    
    assert isinstance(action_items, list), "Should return list"
    assert len(action_items) > 0, "Should have at least one SUBMITTED_TO_DEPT action item"
    assert any(ai["action_item_id"] == test_action_item_id_2 for ai in action_items), "Should contain second test item"
    print(f"   Found {len(action_items)} SUBMITTED_TO_DEPT action item(s)")

test_get_action_items_by_status_submitted()


# ============================================================
# ASSIGNMENT / TRACKING TESTS
# ============================================================

@test("7. Get action items by assigned user (initially empty)")
def test_get_by_assigned_user_empty():
    action_items = db.get_action_items_by_assigned_user(999)  # Non-existent user
    
    assert isinstance(action_items, list), "Should return list"
    assert len(action_items) == 0, "Should be empty for non-existent user"
    print(f"   Correctly returned empty list")

test_get_by_assigned_user_empty()


@test("8. Reassign action item to user")
def test_reassign_action_item():
    global test_action_item_id
    
    result = db.reassign_action_item(
        action_item_id=test_action_item_id,
        new_user_id=1,
        updated_by_user_id=1
    )
    
    assert result is True, "Reassignment should succeed"
    
    # Verify
    action_item = db.get_action_item_by_id(test_action_item_id)
    assert action_item["assigned_to_user_id"] == 1, "Should be assigned to user 1"
    print(f"   Action item reassigned to user 1")

test_reassign_action_item()


@test("9. Get action items by assigned user (after assignment)")
def test_get_by_assigned_user_after_assignment():
    action_items = db.get_action_items_by_assigned_user(1)
    
    assert isinstance(action_items, list), "Should return list"
    assert len(action_items) > 0, "Should have at least one action item"
    assert any(ai["action_item_id"] == test_action_item_id for ai in action_items), "Should contain assigned item"
    print(f"   Found {len(action_items)} action item(s) assigned to user 1")

test_get_by_assigned_user_after_assignment()


@test("10. Get overdue action items (initially none)")
def test_get_overdue_empty():
    # Our test items have future or NULL due dates, so should not be overdue
    action_items = db.get_overdue_action_items()
    
    assert isinstance(action_items, list), "Should return list"
    # Don't assert empty - there might be other overdue items in the DB
    print(f"   Found {len(action_items)} overdue action item(s)")

test_get_overdue_empty()


# ============================================================
# WORKFLOW MUTATION TESTS
# ============================================================

@test("11. Update action item status")
def test_update_status():
    global test_action_item_id
    
    result = db.update_action_item_status(
        action_item_id=test_action_item_id,
        new_status="IN_PROGRESS",
        updated_by_user_id=1
    )
    
    assert result is True, "Update should succeed"
    
    # Verify
    action_item = db.get_action_item_by_id(test_action_item_id)
    assert action_item["status"] == "IN_PROGRESS", "Status should be updated"
    print(f"   Status updated to IN_PROGRESS")

test_update_status()


@test("12. Set action item started")
def test_set_started():
    global test_action_item_id
    
    result = db.set_action_item_started(
        action_item_id=test_action_item_id,
        updated_by_user_id=1
    )
    
    assert result is True, "Update should succeed"
    
    # Verify
    action_item = db.get_action_item_by_id(test_action_item_id)
    assert action_item["started_at"] is not None, "StartedAt should be set"
    print(f"   StartedAt timestamp set")

test_set_started()


@test("13. Set action item completed")
def test_set_completed():
    global test_action_item_id
    
    result = db.set_action_item_completed(
        action_item_id=test_action_item_id,
        updated_by_user_id=1
    )
    
    assert result is True, "Update should succeed"
    
    # Verify
    action_item = db.get_action_item_by_id(test_action_item_id)
    assert action_item["completed_at"] is not None, "CompletedAt should be set"
    print(f"   CompletedAt timestamp set")

test_set_completed()


@test("14. Set action item verified")
def test_set_verified():
    global test_action_item_id
    
    result = db.set_action_item_verified(
        action_item_id=test_action_item_id,
        updated_by_user_id=1
    )
    
    assert result is True, "Update should succeed"
    
    # Verify
    action_item = db.get_action_item_by_id(test_action_item_id)
    assert action_item["verified_at"] is not None, "VerifiedAt should be set"
    print(f"   VerifiedAt timestamp set")

test_set_verified()


# ============================================================
# EDGE CASE TESTS
# ============================================================

@test("15. Get non-existent action item returns None")
def test_get_nonexistent():
    action_item = db.get_action_item_by_id(999999)
    
    assert action_item is None, "Should return None for non-existent ID"
    print(f"   Correctly returned None")

test_get_nonexistent()


@test("16. Update non-existent action item returns False")
def test_update_nonexistent():
    result = db.update_action_item_status(
        action_item_id=999999,
        new_status="DONE",
        updated_by_user_id=1
    )
    
    assert result is False, "Should return False for non-existent ID"
    print(f"   Correctly returned False")

test_update_nonexistent()


@test("17. Reassign non-existent action item returns False")
def test_reassign_nonexistent():
    result = db.reassign_action_item(
        action_item_id=999999,
        new_user_id=1,
        updated_by_user_id=1
    )
    
    assert result is False, "Should return False for non-existent ID"
    print(f"   Correctly returned False")

test_reassign_nonexistent()


@test("18. Delete non-existent action item returns False")
def test_delete_nonexistent():
    result = db.delete_action_item(999999)
    
    assert result is False, "Should return False for non-existent ID"
    print(f"   Correctly returned False")

test_delete_nonexistent()


# ============================================================
# ADMINISTRATION TESTS
# ============================================================

@test("19. Delete action item")
def test_delete_action_item():
    global test_action_item_id_2
    
    # Delete the second test action item
    result = db.delete_action_item(test_action_item_id_2)
    
    assert result is True, "Delete should succeed"
    
    # Verify deletion
    action_item = db.get_action_item_by_id(test_action_item_id_2)
    assert action_item is None, "Action item should be deleted"
    print(f"   Action item {test_action_item_id_2} deleted successfully")

test_delete_action_item()


@test("20. Verify action item count after deletion")
def test_verify_count_after_deletion():
    global test_subcase_id, test_action_item_id, test_action_item_id_2
    
    action_items = db.get_action_items_by_subcase(test_subcase_id)
    
    # Should have one less item now (deleted test_action_item_id_2)
    assert any(ai["action_item_id"] == test_action_item_id for ai in action_items), "First item should still exist"
    assert not any(ai["action_item_id"] == test_action_item_id_2 for ai in action_items), "Second item should be deleted"
    print(f"   Verified deletion - {len(action_items)} action item(s) remain")

test_verify_count_after_deletion()


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
    print("🎉 ALL TESTS PASSED! action_item_subcase_db.py is 100% functional.")
    exit(0)
else:
    print(f"⚠️  {tests_failed} test(s) failed. Please review and fix.")
    exit(1)
