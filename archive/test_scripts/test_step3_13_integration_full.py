"""
STEP 3.13 FULL INTEGRATION TEST — Follow-Up Service

Comprehensive end-to-end integration test for the follow-up service.
Tests real database operations with proper scope filtering and permission enforcement.

Test Scenarios:
1. Happy Path - Assigned User (read, start, complete)
2. Privileged Role Override - Admin modifies non-assigned items
3. Scope Violation - User out of scope cannot access
4. Permission Violation - Non-assigned without role cannot modify
5. Workflow Lifecycle - Full action item lifecycle
6. Delay/Cancel - Delay function works correctly
"""

import sys
import os
from datetime import datetime, date

# Force UTF-8 encoding for emoji support
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add backend directory to Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.database import get_connection


def test(description):
    """Test decorator"""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*80}")
            print(f"TEST: {description}")
            print('='*80)
            try:
                func()
                print("\n✅ TEST PASSED")
            except AssertionError as e:
                print(f"\n❌ TEST FAILED: {str(e)}")
                import traceback
                traceback.print_exc()
            except Exception as e:
                print(f"\n❌ TEST ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
        return wrapper
    return decorator


# Mock user class for testing
class MockUser:
    def __init__(self, user_id, allowed_unit_ids, role=None):
        self.user_id = user_id
        self.allowed_unit_ids = allowed_unit_ids if isinstance(allowed_unit_ids, set) else set(allowed_unit_ids)
        self.role = role


def create_test_incident():
    """Helper to create a minimal test incident."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO dbo.APP_IncidentCase (
                ComplaintText, FeedbackRecievedDate, IssuingOrgUnitID,
                SourceID, ExplanationStatusID, ImmediateAction, TakenAction,
                CreatedByUserID, PatientName, isINPatient, ClinicalRiskTypeID,
                FeedbackIntentTypeID, BuildingID, DomainID, CategoryID,
                SubCategoryID, ClassificationID, SeverityID, StageID,
                HarmLevelID, CaseStatusID, RequiresExplanation
            )
            OUTPUT INSERTED.IncidentRequestCaseID
            VALUES (
                'Integration Test Incident', ?, 1, 1, 4, '', '', 1, 'Test Patient',
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0
            )
        """, (datetime.now(),))
        
        incident_id = cursor.fetchone()[0]
        conn.commit()
        return incident_id
    finally:
        cursor.close()
        conn.close()


def create_test_subcase(incident_id, target_org_unit_id):
    """Helper to create a test subcase."""
    from api_v2.db_layer import administrative_subcase_db
    
    subcase_id = administrative_subcase_db.create_subcase(
        case_type="INCIDENT",
        incident_id=incident_id,
        seasonal_report_id=None,
        target_org_unit_id=target_org_unit_id,
        created_by_user_id=1
    )
    return subcase_id


def create_test_action_item(subcase_id, assigned_to_user_id):
    """Helper to create a test action item."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO dbo.APP_SubcaseActionItem (
                SubcaseID, Title, Description, DueDate, Status,
                AssignedToUserID, CreatedAt, CreatedByUserID
            )
            OUTPUT INSERTED.ActionItemID
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            subcase_id,
            "Integration Test Action Item",
            "Testing follow-up service",
            date.today(),
            "DRAFT",
            assigned_to_user_id,
            datetime.now(),
            1
        ))
        
        action_item_id = cursor.fetchone()[0]
        conn.commit()
        return action_item_id
    finally:
        cursor.close()
        conn.close()


def cleanup_test_data(incident_id):
    """Helper to cleanup test data."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Delete in correct order
        cursor.execute("DELETE FROM dbo.APP_SubcaseActionItem WHERE SubcaseID IN (SELECT SubcaseID FROM dbo.APP_AdministrativeSubcase WHERE IncidentRequestCaseID = ?)", (incident_id,))
        cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE IncidentRequestCaseID = ?", (incident_id,))
        cursor.execute("DELETE FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (incident_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


@test("TEST 1: Happy Path - Assigned User Can Read, Start, Complete")
def test_happy_path_assigned_user():
    """
    Test that an assigned user within scope can:
    1. See their action item in get_action_items_for_user
    2. Start the action item
    3. Complete the action item
    """
    from api_v2.services.follow_up_service import (
        get_action_items_for_user,
        start_action_item,
        complete_action_item
    )
    from api_v2.db_layer import action_item_subcase_db
    
    print("\n[SETUP] Creating test data...")
    incident_id = None
    
    try:
        # Create incident → subcase → action item
        incident_id = create_test_incident()
        print(f"  Created incident: {incident_id}")
        
        subcase_id = create_test_subcase(incident_id, target_org_unit_id=1)
        print(f"  Created subcase: {subcase_id} (target org unit: 1)")
        
        action_item_id = create_test_action_item(subcase_id, assigned_to_user_id=1)
        print(f"  Created action item: {action_item_id} (assigned to user 1)")
        
        # Create user: assigned to action item, correct scope
        user = MockUser(user_id=1, allowed_unit_ids={1}, role=None)
        print(f"  Created user: ID=1, scope={{1}}, role=None")
        
        # TEST 1: Get action items
        print("\n[TEST 1.1] get_action_items_for_user...")
        action_items = get_action_items_for_user(user)
        print(f"  Returned {len(action_items)} action item(s)")
        
        assert len(action_items) > 0, "Should return at least one action item"
        found = any(item['action_item_id'] == action_item_id for item in action_items)
        assert found, f"Should find action item {action_item_id}"
        print(f"  ✓ User can see their assigned action item")
        
        # TEST 2: Start action item
        print("\n[TEST 1.2] start_action_item...")
        success = start_action_item(action_item_id, user)
        assert success, "start_action_item should return True"
        
        # Verify StartedAt is set
        action_item = action_item_subcase_db.get_action_item_by_id(action_item_id)
        assert action_item["started_at"] is not None, "StartedAt should be set"
        print(f"  ✓ StartedAt timestamp set: {action_item['started_at']}")
        
        # TEST 3: Complete action item
        print("\n[TEST 1.3] complete_action_item...")
        success = complete_action_item(action_item_id, user)
        assert success, "complete_action_item should return True"
        
        # Verify CompletedAt is set
        action_item = action_item_subcase_db.get_action_item_by_id(action_item_id)
        assert action_item["completed_at"] is not None, "CompletedAt should be set"
        print(f"  ✓ CompletedAt timestamp set: {action_item['completed_at']}")
        
        print("\n[RESULT] All operations successful for assigned user within scope!")
        
    finally:
        if incident_id:
            print("\n[CLEANUP] Removing test data...")
            cleanup_test_data(incident_id)
            print("  ✓ Cleanup complete")


@test("TEST 2: Privileged Role Override - Admin Can Modify Non-Assigned Items")
def test_privileged_role_override():
    """
    Test that an admin user can modify action items they're not assigned to,
    as long as they're within scope.
    """
    from api_v2.services.follow_up_service import (
        get_action_items_for_user,
        start_action_item,
        delay_action_item
    )
    from api_v2.db_layer import action_item_subcase_db
    
    print("\n[SETUP] Creating test data...")
    incident_id = None
    
    try:
        # Create incident → subcase → action item assigned to user 1
        incident_id = create_test_incident()
        print(f"  Created incident: {incident_id}")
        
        subcase_id = create_test_subcase(incident_id, target_org_unit_id=1)
        print(f"  Created subcase: {subcase_id} (target org unit: 1)")
        
        action_item_id = create_test_action_item(subcase_id, assigned_to_user_id=1)
        print(f"  Created action item: {action_item_id} (assigned to user 1)")
        
        # Create ADMIN user (NOT assigned, but has role, correct scope)
        admin_user = MockUser(user_id=2, allowed_unit_ids={1}, role="ADMIN")
        print(f"  Created user: ID=2, scope={{1}}, role=ADMIN")
        
        # TEST 1: Get action items (should NOT see it - not assigned)
        print("\n[TEST 2.1] get_action_items_for_user (should be empty)...")
        action_items = get_action_items_for_user(admin_user)
        print(f"  Returned {len(action_items)} action item(s)")
        assert len(action_items) == 0, "Admin should not see items they're not assigned to"
        print(f"  ✓ Admin correctly doesn't see non-assigned items")
        
        # TEST 2: Start action item (should SUCCEED - admin override)
        print("\n[TEST 2.2] start_action_item (admin override)...")
        success = start_action_item(action_item_id, admin_user)
        assert success, "Admin should be able to start non-assigned items within scope"
        
        # Verify StartedAt is set
        action_item = action_item_subcase_db.get_action_item_by_id(action_item_id)
        assert action_item["started_at"] is not None, "StartedAt should be set"
        print(f"  ✓ Admin successfully started non-assigned item")
        
        # TEST 3: Delay action item (should SUCCEED - admin override)
        print("\n[TEST 2.3] delay_action_item (admin override)...")
        success = delay_action_item(action_item_id, admin_user)
        assert success, "Admin should be able to cancel non-assigned items within scope"
        
        # Verify status changed to CANCELLED
        action_item = action_item_subcase_db.get_action_item_by_id(action_item_id)
        assert action_item["status"] == "CANCELLED", "Status should be CANCELLED"
        print(f"  ✓ Admin successfully cancelled non-assigned item")
        
        print("\n[RESULT] Admin role override works correctly within scope!")
        
    finally:
        if incident_id:
            print("\n[CLEANUP] Removing test data...")
            cleanup_test_data(incident_id)
            print("  ✓ Cleanup complete")


@test("TEST 3: Scope Violation - User Out Of Scope Cannot Access")
def test_scope_violation():
    """
    Test that even an assigned user cannot access items outside their scope.
    Scope is enforced FIRST, before ownership.
    """
    from api_v2.services.follow_up_service import (
        get_action_items_for_user,
        start_action_item,
        Forbidden
    )
    
    print("\n[SETUP] Creating test data...")
    incident_id = None
    
    try:
        # Create incident → subcase targeting org unit 1 → action item assigned to user 1
        incident_id = create_test_incident()
        print(f"  Created incident: {incident_id}")
        
        subcase_id = create_test_subcase(incident_id, target_org_unit_id=1)
        print(f"  Created subcase: {subcase_id} (target org unit: 1)")
        
        action_item_id = create_test_action_item(subcase_id, assigned_to_user_id=1)
        print(f"  Created action item: {action_item_id} (assigned to user 1)")
        
        # Create user: assigned BUT wrong scope
        user_wrong_scope = MockUser(user_id=1, allowed_unit_ids={2, 3}, role=None)
        print(f"  Created user: ID=1 (assigned), scope={{2,3}} (WRONG!), role=None")
        
        # TEST 1: Get action items (should be empty - scope filter)
        print("\n[TEST 3.1] get_action_items_for_user (should be empty due to scope)...")
        action_items = get_action_items_for_user(user_wrong_scope)
        print(f"  Returned {len(action_items)} action item(s)")
        assert len(action_items) == 0, "Should return empty list due to scope filter"
        print(f"  ✓ Scope filter correctly excluded out-of-scope item")
        
        # TEST 2: Start action item (should raise Forbidden - scope check first)
        print("\n[TEST 3.2] start_action_item (should raise Forbidden)...")
        try:
            start_action_item(action_item_id, user_wrong_scope)
            raise AssertionError("Should have raised Forbidden due to scope violation")
        except Forbidden as e:
            print(f"  ✓ Correctly raised Forbidden: {str(e)}")
        
        print("\n[RESULT] Scope is enforced FIRST, even for assigned users!")
        
    finally:
        if incident_id:
            print("\n[CLEANUP] Removing test data...")
            cleanup_test_data(incident_id)
            print("  ✓ Cleanup complete")


@test("TEST 4: Permission Violation - Non-Assigned Without Role Cannot Modify")
def test_permission_violation():
    """
    Test that a user who is NOT assigned and has no privileged role
    cannot modify items, even if they're within scope.
    """
    from api_v2.services.follow_up_service import (
        get_action_items_for_user,
        start_action_item,
        Forbidden
    )
    
    print("\n[SETUP] Creating test data...")
    incident_id = None
    
    try:
        # Create incident → subcase → action item assigned to user 1
        incident_id = create_test_incident()
        print(f"  Created incident: {incident_id}")
        
        subcase_id = create_test_subcase(incident_id, target_org_unit_id=1)
        print(f"  Created subcase: {subcase_id} (target org unit: 1)")
        
        action_item_id = create_test_action_item(subcase_id, assigned_to_user_id=1)
        print(f"  Created action item: {action_item_id} (assigned to user 1)")
        
        # Create user: NOT assigned, no privileged role, but correct scope
        regular_user = MockUser(user_id=2, allowed_unit_ids={1}, role="REGULAR_USER")
        print(f"  Created user: ID=2 (NOT assigned), scope={{1}}, role=REGULAR_USER")
        
        # TEST 1: Get action items (should be empty - not assigned)
        print("\n[TEST 4.1] get_action_items_for_user (should be empty - not assigned)...")
        action_items = get_action_items_for_user(regular_user)
        print(f"  Returned {len(action_items)} action item(s)")
        assert len(action_items) == 0, "Should not see items they're not assigned to"
        print(f"  ✓ User correctly doesn't see non-assigned items")
        
        # TEST 2: Start action item (should raise Forbidden - no permission)
        print("\n[TEST 4.2] start_action_item (should raise Forbidden)...")
        try:
            start_action_item(action_item_id, regular_user)
            raise AssertionError("Should have raised Forbidden due to lack of permission")
        except Forbidden as e:
            print(f"  ✓ Correctly raised Forbidden: {str(e)}")
        
        print("\n[RESULT] Permission check works: non-assigned without role cannot modify!")
        
    finally:
        if incident_id:
            print("\n[CLEANUP] Removing test data...")
            cleanup_test_data(incident_id)
            print("  ✓ Cleanup complete")


@test("TEST 5: Workflow Lifecycle - Full Action Item Lifecycle")
def test_workflow_lifecycle():
    """
    Test the full lifecycle of an action item:
    DRAFT → Started (StartedAt) → Completed (CompletedAt)
    Verify timestamps are set correctly and sequentially.
    """
    from api_v2.services.follow_up_service import start_action_item, complete_action_item
    from api_v2.db_layer import action_item_subcase_db
    
    print("\n[SETUP] Creating test data...")
    incident_id = None
    
    try:
        # Create incident → subcase → action item
        incident_id = create_test_incident()
        subcase_id = create_test_subcase(incident_id, target_org_unit_id=1)
        action_item_id = create_test_action_item(subcase_id, assigned_to_user_id=1)
        print(f"  Created action item: {action_item_id}")
        
        user = MockUser(user_id=1, allowed_unit_ids={1}, role=None)
        
        # Initial state
        print("\n[VERIFY] Initial state...")
        action_item = action_item_subcase_db.get_action_item_by_id(action_item_id)
        print(f"  Status: {action_item['status']}")
        print(f"  StartedAt: {action_item['started_at']}")
        print(f"  CompletedAt: {action_item['completed_at']}")
        assert action_item['status'] == 'DRAFT', "Initial status should be DRAFT"
        assert action_item['started_at'] is None, "StartedAt should be None initially"
        assert action_item['completed_at'] is None, "CompletedAt should be None initially"
        
        # Start
        print("\n[ACTION 1] Starting action item...")
        start_action_item(action_item_id, user)
        action_item = action_item_subcase_db.get_action_item_by_id(action_item_id)
        started_at = action_item['started_at']
        print(f"  StartedAt: {started_at}")
        assert started_at is not None, "StartedAt should be set"
        
        # Complete
        print("\n[ACTION 2] Completing action item...")
        complete_action_item(action_item_id, user)
        action_item = action_item_subcase_db.get_action_item_by_id(action_item_id)
        completed_at = action_item['completed_at']
        print(f"  CompletedAt: {completed_at}")
        assert completed_at is not None, "CompletedAt should be set"
        
        # Verify timestamps are sequential
        print("\n[VERIFY] Timestamp sequence...")
        assert completed_at >= started_at, "CompletedAt should be after StartedAt"
        print(f"  ✓ Timestamps are sequential")
        
        print("\n[RESULT] Full lifecycle works correctly!")
        
    finally:
        if incident_id:
            print("\n[CLEANUP] Removing test data...")
            cleanup_test_data(incident_id)
            print("  ✓ Cleanup complete")


@test("TEST 6: Delay/Cancel Action Item")
def test_delay_action_item():
    """
    Test the delay_action_item function:
    - Should set status to CANCELLED
    - Should work for assigned user
    - Should work for admin role
    """
    from api_v2.services.follow_up_service import delay_action_item
    from api_v2.db_layer import action_item_subcase_db
    
    print("\n[SETUP] Creating test data...")
    incident_id = None
    
    try:
        # Create incident → subcase → action item
        incident_id = create_test_incident()
        subcase_id = create_test_subcase(incident_id, target_org_unit_id=1)
        action_item_id = create_test_action_item(subcase_id, assigned_to_user_id=1)
        print(f"  Created action item: {action_item_id}")
        
        user = MockUser(user_id=1, allowed_unit_ids={1}, role=None)
        
        # Initial state
        print("\n[VERIFY] Initial state...")
        action_item = action_item_subcase_db.get_action_item_by_id(action_item_id)
        print(f"  Status: {action_item['status']}")
        assert action_item['status'] == 'DRAFT', "Initial status should be DRAFT"
        
        # Delay/Cancel
        print("\n[ACTION] Delaying (cancelling) action item...")
        success = delay_action_item(action_item_id, user)
        assert success, "delay_action_item should return True"
        
        # Verify status changed
        action_item = action_item_subcase_db.get_action_item_by_id(action_item_id)
        print(f"  New status: {action_item['status']}")
        assert action_item['status'] == 'CANCELLED', "Status should be CANCELLED"
        print(f"  ✓ Status correctly changed to CANCELLED")
        
        print("\n[RESULT] Delay/cancel function works correctly!")
        
    finally:
        if incident_id:
            print("\n[CLEANUP] Removing test data...")
            cleanup_test_data(incident_id)
            print("  ✓ Cleanup complete")


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STEP 3.13 — FULL INTEGRATION TEST SUITE")
    print("End-to-End Testing of Follow-Up Service")
    print("="*80)
    
    print("\n" + "="*80)
    print("RUNNING INTEGRATION TESTS...")
    print("="*80)
    
    test_happy_path_assigned_user()
    test_privileged_role_override()
    test_scope_violation()
    test_permission_violation()
    test_workflow_lifecycle()
    test_delay_action_item()
    
    print("\n" + "="*80)
    print("INTEGRATION TEST SUITE COMPLETE")
    print("="*80)
    print("\n🎉 If all tests passed, STEP 3.13 is fully validated!")
    print("✅ Read operations work with scope filtering")
    print("✅ Mutations work with permission enforcement")
    print("✅ Scope is enforced FIRST, then ownership/role")
    print("✅ Full lifecycle tested end-to-end")
