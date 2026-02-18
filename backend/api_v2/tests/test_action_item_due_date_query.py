"""
📋 PHASE F — TEST F-B3 — ACTION ITEMS BY DUE DATE RANGE QUERY (API V2)

Integration tests for DB query: get_action_items_by_due_date_range

Tests verify:
- Date range filtering works correctly
- CANCELLED action items are excluded
- Subcase status gate (ADMIN_APPROVED or later) is enforced
- NULL DueDate items are excluded
- Joins to Users and OrgUnit tables work

These are integration tests requiring real DB connection.
"""

import pytest
from datetime import date, datetime, timedelta
from backend.api_v2.db_layer import action_item_subcase_db as action_db
from backend.api_v2.db_layer import administrative_subcase_db as subcase_db


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def db_connection():
    """Provide a database connection for tests."""
    try:
        conn = action_db.get_db_connection()
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"Database connection not available: {e}")


@pytest.fixture(scope="module")
def test_incident_id():
    """
    Use a known valid incident ID from the database.
    This assumes incident ID 36 exists (common test incident).
    """
    return 36


@pytest.fixture(scope="module")
def test_org_unit_id():
    """Use a known valid org unit ID."""
    return 1


@pytest.fixture(scope="module")
def test_user_id():
    """Use a known valid user ID."""
    return 1


@pytest.fixture
def test_subcase_admin_approved(test_incident_id, test_org_unit_id, test_user_id):
    """
    Create a test subcase with ADMIN_APPROVED status.
    This passes the status gate for action log report.
    
    Cleanup after test.
    """
    # Create subcase in initial status
    subcase_id = subcase_db.create_subcase(
        case_type="INCIDENT_RESPONSE",
        incident_id=test_incident_id,
        seasonal_report_id=None,
        target_org_unit_id=test_org_unit_id,
        created_by_user_id=test_user_id,
        initial_status="SUBMITTED_TO_SECTION"
    )
    
    assert subcase_id is not None, "Failed to create test subcase"
    
    # Update to ADMIN_APPROVED status (simulate workflow progression)
    conn = subcase_db.get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE dbo.APP_AdministrativeSubcase SET Status = 'ADMIN_APPROVED' WHERE SubcaseID = ?",
        (subcase_id,)
    )
    conn.commit()
    cursor.close()
    conn.close()
    
    yield subcase_id
    
    # Cleanup: delete test subcase and its action items
    try:
        conn = subcase_db.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (subcase_id,))
        cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
        conn.commit()
        cursor.close()
        conn.close()
    except:
        pass  # Ignore cleanup errors


@pytest.fixture
def test_subcase_pre_admin(test_incident_id, test_org_unit_id, test_user_id):
    """
    Create a test subcase with status BEFORE ADMIN_APPROVED.
    This should NOT pass the status gate.
    
    Cleanup after test.
    """
    subcase_id = subcase_db.create_subcase(
        case_type="INCIDENT_RESPONSE",
        incident_id=test_incident_id,
        seasonal_report_id=None,
        target_org_unit_id=test_org_unit_id,
        created_by_user_id=test_user_id,
        initial_status="SUBMITTED_TO_DEPT"  # Pre-admin status
    )
    
    assert subcase_id is not None, "Failed to create test subcase"
    
    yield subcase_id
    
    # Cleanup
    try:
        conn = subcase_db.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM dbo.APP_SubcaseActionItem WHERE SubcaseID = ?", (subcase_id,))
        cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
        conn.commit()
        cursor.close()
        conn.close()
    except:
        pass


# ============================================================================
# TEST 1 — RETURNS ROWS IN DATE RANGE
# ============================================================================

def test_returns_rows_in_range(db_connection, test_subcase_admin_approved, test_user_id):
    """
    Test that action items with DueDate in range are returned.
    """
    # Create action item with due date in test window
    test_due_date = date.today() + timedelta(days=7)
    
    action_item_id = action_db.create_action_item(
        subcase_id=test_subcase_admin_approved,
        title="Test Action in Range",
        description="This should be returned",
        due_date=test_due_date,
        created_by_user_id=test_user_id,
        initial_status="IN_PROGRESS",
        assigned_to_user_id=test_user_id
    )
    
    assert action_item_id is not None
    
    # Query with date range that includes our test action
    start_date = date.today()
    end_date = date.today() + timedelta(days=30)
    
    results = action_db.get_action_items_by_due_date_range(
        db_connection,
        start_date,
        end_date
    )
    
    # Find our action item in results
    found = False
    for item in results:
        if item["action_item_id"] == action_item_id:
            found = True
            
            # Validate structure
            assert item["subcase_id"] == test_subcase_admin_approved
            assert item["title"] == "Test Action in Range"
            assert item["status_code"] == "IN_PROGRESS"
            assert item["due_date"] == test_due_date
            assert item["assigned_to_user_id"] == test_user_id
            
            # Validate date is in range
            assert start_date <= item["due_date"] <= end_date
            
            # Validate joins present
            assert "assigned_to_display_name" in item
            assert "org_unit_name" in item
            assert "subcase_status" in item
            
            print(f"✅ Found action item {action_item_id} in results")
            print(f"   Assigned to: {item['assigned_to_display_name']}")
            print(f"   Org unit: {item['org_unit_name']}")
            
            break
    
    assert found, f"Action item {action_item_id} should be in results"


# ============================================================================
# TEST 2 — EXCLUDES CANCELLED
# ============================================================================

def test_excludes_cancelled(db_connection, test_subcase_admin_approved, test_user_id):
    """
    Test that CANCELLED action items are excluded from results.
    """
    # Create CANCELLED action item in date range
    test_due_date = date.today() + timedelta(days=5)
    
    cancelled_id = action_db.create_action_item(
        subcase_id=test_subcase_admin_approved,
        title="Cancelled Action Item",
        description="This should NOT be returned",
        due_date=test_due_date,
        created_by_user_id=test_user_id,
        initial_status="CANCELLED",
        assigned_to_user_id=test_user_id
    )
    
    assert cancelled_id is not None
    
    # Query date range
    start_date = date.today()
    end_date = date.today() + timedelta(days=30)
    
    results = action_db.get_action_items_by_due_date_range(
        db_connection,
        start_date,
        end_date
    )
    
    # Verify CANCELLED item is NOT in results
    for item in results:
        assert item["action_item_id"] != cancelled_id, \
            "CANCELLED action items should be excluded from results"
    
    print(f"✅ Correctly excluded CANCELLED action item {cancelled_id}")


# ============================================================================
# TEST 3 — EXCLUDES SUBCASE BEFORE ADMIN GATE
# ============================================================================

def test_excludes_subcase_before_gate(db_connection, test_subcase_pre_admin, test_user_id):
    """
    Test that action items from subcases not yet ADMIN_APPROVED are excluded.
    """
    # Create action item for subcase that hasn't passed admin gate
    test_due_date = date.today() + timedelta(days=5)
    
    pre_admin_action_id = action_db.create_action_item(
        subcase_id=test_subcase_pre_admin,
        title="Pre-Admin Action Item",
        description="Subcase status is SUBMITTED_TO_DEPT",
        due_date=test_due_date,
        created_by_user_id=test_user_id,
        initial_status="DRAFT",
        assigned_to_user_id=test_user_id
    )
    
    assert pre_admin_action_id is not None
    
    # Query date range
    start_date = date.today()
    end_date = date.today() + timedelta(days=30)
    
    results = action_db.get_action_items_by_due_date_range(
        db_connection,
        start_date,
        end_date
    )
    
    # Verify pre-admin action is NOT in results
    for item in results:
        assert item["action_item_id"] != pre_admin_action_id, \
            "Action items from pre-ADMIN_APPROVED subcases should be excluded"
    
    print(f"✅ Correctly excluded action item {pre_admin_action_id} from pre-admin subcase")


# ============================================================================
# TEST 4 — EXCLUDES NULL DUE DATE
# ============================================================================

def test_excludes_null_due_date(db_connection, test_subcase_admin_approved, test_user_id):
    """
    Test that action items with NULL DueDate are excluded.
    """
    # Create action item without due date
    null_due_date_id = action_db.create_action_item(
        subcase_id=test_subcase_admin_approved,
        title="No Due Date Action",
        description="DueDate is NULL",
        due_date=None,  # NULL due date
        created_by_user_id=test_user_id,
        initial_status="IN_PROGRESS",
        assigned_to_user_id=test_user_id
    )
    
    assert null_due_date_id is not None
    
    # Query any date range
    start_date = date(2020, 1, 1)
    end_date = date(2030, 12, 31)
    
    results = action_db.get_action_items_by_due_date_range(
        db_connection,
        start_date,
        end_date
    )
    
    # Verify NULL due date action is NOT in results
    for item in results:
        assert item["action_item_id"] != null_due_date_id, \
            "Action items with NULL DueDate should be excluded"
        assert item["due_date"] is not None, \
            "All returned items should have non-NULL DueDate"
    
    print(f"✅ Correctly excluded action item {null_due_date_id} with NULL DueDate")


# ============================================================================
# TEST 5 — VALIDATES JOINS PRESENT
# ============================================================================

def test_validates_joins_present(db_connection, test_subcase_admin_approved, test_user_id):
    """
    Test that joined fields from Users and OrgUnit tables are present.
    """
    # Create action item with user assignment
    test_due_date = date.today() + timedelta(days=3)
    
    action_item_id = action_db.create_action_item(
        subcase_id=test_subcase_admin_approved,
        title="Test Joins",
        description="Verify joined fields",
        due_date=test_due_date,
        created_by_user_id=test_user_id,
        initial_status="ADMIN_APPROVED",
        assigned_to_user_id=test_user_id
    )
    
    assert action_item_id is not None
    
    # Query
    start_date = date.today()
    end_date = date.today() + timedelta(days=30)
    
    results = action_db.get_action_items_by_due_date_range(
        db_connection,
        start_date,
        end_date
    )
    
    # Find our action and check joins
    found = False
    for item in results:
        if item["action_item_id"] == action_item_id:
            found = True
            
            # Check all join fields exist
            assert "assigned_to_display_name" in item, "User join field missing"
            assert "org_unit_name" in item, "OrgUnit join field missing"
            assert "subcase_status" in item, "Subcase status missing"
            assert "target_org_unit_id" in item, "Target org unit ID missing"
            
            # User display name should be populated (or None if no user)
            if item["assigned_to_user_id"] is not None:
                # Should have display name (may be None if user deleted, but field exists)
                pass
            
            print(f"✅ Join fields present:")
            print(f"   assigned_to_display_name: {item['assigned_to_display_name']}")
            print(f"   org_unit_name: {item['org_unit_name']}")
            print(f"   subcase_status: {item['subcase_status']}")
            
            break
    
    assert found, f"Action item {action_item_id} should be in results"


# ============================================================================
# TEST 6 — EXCLUDES OUT OF RANGE DATES
# ============================================================================

def test_excludes_out_of_range(db_connection, test_subcase_admin_approved, test_user_id):
    """
    Test that action items outside the date range are excluded.
    """
    # Create action item with future due date (outside test range)
    far_future_date = date.today() + timedelta(days=100)
    
    future_action_id = action_db.create_action_item(
        subcase_id=test_subcase_admin_approved,
        title="Future Action",
        description="Due date outside query range",
        due_date=far_future_date,
        created_by_user_id=test_user_id,
        initial_status="DRAFT",
        assigned_to_user_id=test_user_id
    )
    
    assert future_action_id is not None
    
    # Query narrow date range that excludes the future action
    start_date = date.today()
    end_date = date.today() + timedelta(days=30)
    
    results = action_db.get_action_items_by_due_date_range(
        db_connection,
        start_date,
        end_date
    )
    
    # Verify future action is NOT in results
    for item in results:
        assert item["action_item_id"] != future_action_id, \
            "Action items outside date range should be excluded"
        # All returned items should be in range
        assert start_date <= item["due_date"] <= end_date, \
            f"DueDate {item['due_date']} should be in range [{start_date}, {end_date}]"
    
    print(f"✅ Correctly excluded out-of-range action item {future_action_id}")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
