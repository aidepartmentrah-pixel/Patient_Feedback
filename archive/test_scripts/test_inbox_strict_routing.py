"""
INBOX STRICT ROUTING TEST SUITE (MODEL A)

Tests the refactored inbox routing to ensure:
- Each role only receives items for statuses they are responsible for
- Unified inbox routing has been removed
- Non-responsible roles receive empty inboxes
- No crashes or exceptions for unsupported roles
"""

import sys
import os

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
from datetime import datetime
import json


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_test(test_name, passed, message=""):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{status}: {test_name}")
    if message:
        print(f"   {message}")


class MockUser:
    """Mock user for testing with proper scopes and allowed_unit_ids"""
    def __init__(self, user_id, username, role_code, allowed_unit_ids=None):
        self.user_id = user_id
        self.username = username
        self.role_code = role_code
        self.scopes = [type('obj', (object,), {'role_code': role_code})]
        # allowed_unit_ids is used by scope filter - use set for proper filtering
        self.allowed_unit_ids = set(allowed_unit_ids) if allowed_unit_ids else set()


# =============================================================================
# TEST DATA SETUP
# =============================================================================

def setup_test_data():
    """
    Create test subcases in different statuses for testing inbox routing.
    Returns dict mapping status codes to subcase_ids.
    """
    print_section("Setting Up Test Data")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Find 2 valid org units (any non-frozen units)
        cursor.execute("""
            SELECT TOP 2 UniqueID, Name
            FROM AdminsrationUnit
            WHERE Frozen = 0
            ORDER BY UniqueID
        """)
        org_units = cursor.fetchall()
        
        if len(org_units) < 2:
            raise Exception("Need at least 2 org units for testing")
        
        section_id = org_units[0][0]
        dept_id = org_units[1][0]
        print(f"Using Org Unit 1: {org_units[0][1]} (ID: {section_id})")
        print(f"Using Org Unit 2: {org_units[1][1]} (ID: {dept_id})")
        
        # Find a test user
        cursor.execute("SELECT TOP 1 UserID FROM APP_Users")
        user_row = cursor.fetchone()
        if not user_row:
            raise Exception("No users found in database")
        test_user_id = user_row[0]
        print(f"Using test user ID: {test_user_id}")
        
        # Create a parent incident case
        cursor.execute("""
            INSERT INTO dbo.APP_IncidentCase (
                ComplaintText, ImmediateAction, TakenAction, 
                FeedbackRecievedDate, PatientName,
                IssuingOrgUnitID, CreatedByUserID,
                isINPatient, ClinicalRiskTypeID, FeedbackIntentTypeID,
                BuildingID, DomainID, CategoryID, SubCategoryID,
                ClassificationID, SeverityID, StageID, HarmLevelID,
                CaseStatusID, SourceID, ExplanationStatusID, RequiresExplanation
            )
            OUTPUT INSERTED.IncidentRequestCaseID
            VALUES (?, ?, ?, GETDATE(), ?,
                    ?, ?, 1, 1, 1,
                    1, 1, 6, 19,
                    132, 1, 1, 1,
                    1, 1, 1, 0)
        """, 'Test incident for inbox routing', 'Test action', 'Test action taken',
             'Test Patient', section_id, test_user_id)
        
        case_id = cursor.fetchone()[0]
        print(f"Created test incident case ID: {case_id}")
        
        # Create subcases in different workflow statuses
        test_subcases = {}
        
        statuses = [
            ('SUBMITTED_TO_SECTION', section_id, 'SECTION_ADMIN'),
            ('SECTION_ACCEPTED_PENDING_DEPT', dept_id, 'DEPARTMENT_ADMIN'),
            ('DEPT_ACCEPTED_PENDING_ADMIN', dept_id, 'ADMINISTRATION_ADMIN'),
        ]
        
        for status, target_org_unit_id, responsible_role in statuses:
            cursor.execute("""
                INSERT INTO dbo.APP_AdministrativeSubCase (
                    IncidentRequestCaseID,
                    CaseType,
                    TargetOrgUnitID,
                    Status,
                    CreatedAt,
                    CreatedByUserID
                )
                VALUES (?, 'INCIDENT_RESPONSE', ?, ?, GETDATE(), ?)
            """, case_id, target_org_unit_id, status, test_user_id)
            conn.commit()
            
            cursor.execute("SELECT @@IDENTITY as SubcaseID")
            subcase_id = cursor.fetchone()[0]
            test_subcases[status] = {
                'subcase_id': subcase_id,
                'target_org_unit_id': target_org_unit_id,
                'responsible_role': responsible_role
            }
            print(f"Created subcase {subcase_id} with status: {status}")
        
        print(f"\n✓ Test data created successfully")
        return {
            'case_id': case_id,
            'section_id': section_id,
            'dept_id': dept_id,
            'subcases': test_subcases
        }
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error setting up test data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def cleanup_test_data(test_data):
    """Clean up test data after tests complete"""
    print_section("Cleaning Up Test Data")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        case_id = test_data['case_id']
        
        # Delete subcases
        cursor.execute("DELETE FROM dbo.APP_AdministrativeSubCase WHERE IncidentRequestCaseID = ?", case_id)
        deleted_subcases = cursor.rowcount
        
        # Delete incident case
        cursor.execute("DELETE FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", case_id)
        deleted_cases = cursor.rowcount
        
        conn.commit()
        print(f"✓ Deleted {deleted_subcases} subcases and {deleted_cases} case(s)")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error cleaning up: {str(e)}")


# =============================================================================
# INBOX ROUTING TESTS
# =============================================================================

def test_section_admin_inbox_strict(test_data):
    """
    TEST: SECTION_ADMIN inbox returns only SUBMITTED_TO_SECTION items
    EXPECT: Only items in SUBMITTED_TO_SECTION status
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    section_id = test_data['section_id']
    user = MockUser(1, 'section_admin', 'SECTION_ADMIN', allowed_unit_ids=[section_id])
    
    inbox = get_inbox(user)
    
    # Check that inbox contains items
    has_items = len(inbox) > 0
    print_test(
        "SECTION_ADMIN receives inbox items",
        has_items,
        f"Received {len(inbox)} item(s)"
    )
    
    # Check that all items are SUBMITTED_TO_SECTION
    all_correct_status = all(item['status'] == 'SUBMITTED_TO_SECTION' for item in inbox)
    print_test(
        "All items are SUBMITTED_TO_SECTION status",
        all_correct_status,
        f"Statuses: {[item['status'] for item in inbox]}"
    )
    
    # Verify no items from other statuses leaked in
    other_statuses = [item['status'] for item in inbox if item['status'] != 'SUBMITTED_TO_SECTION']
    no_leakage = len(other_statuses) == 0
    print_test(
        "No cross-status leakage",
        no_leakage,
        f"Other statuses found: {other_statuses}" if other_statuses else "Clean"
    )
    
    return has_items and all_correct_status and no_leakage


def test_department_admin_inbox_strict(test_data):
    """
    TEST: DEPARTMENT_ADMIN inbox returns only SECTION_ACCEPTED_PENDING_DEPT items
    EXPECT: Only items in SECTION_ACCEPTED_PENDING_DEPT status
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    dept_id = test_data['dept_id']
    user = MockUser(2, 'dept_admin', 'DEPARTMENT_ADMIN', allowed_unit_ids=[dept_id])
    
    inbox = get_inbox(user)
    
    # Check that inbox contains items
    has_items = len(inbox) > 0
    print_test(
        "DEPARTMENT_ADMIN receives inbox items",
        has_items,
        f"Received {len(inbox)} item(s)"
    )
    
    # Check that all items are SECTION_ACCEPTED_PENDING_DEPT
    all_correct_status = all(item['status'] == 'SECTION_ACCEPTED_PENDING_DEPT' for item in inbox)
    print_test(
        "All items are SECTION_ACCEPTED_PENDING_DEPT status",
        all_correct_status,
        f"Statuses: {[item['status'] for item in inbox]}"
    )
    
    # Verify no items from other statuses leaked in
    other_statuses = [item['status'] for item in inbox if item['status'] != 'SECTION_ACCEPTED_PENDING_DEPT']
    no_leakage = len(other_statuses) == 0
    print_test(
        "No cross-status leakage",
        no_leakage,
        f"Other statuses found: {other_statuses}" if other_statuses else "Clean"
    )
    
    return has_items and all_correct_status and no_leakage


def test_administration_admin_inbox_strict(test_data):
    """
    TEST: ADMINISTRATION_ADMIN inbox returns only DEPT_ACCEPTED_PENDING_ADMIN items
    EXPECT: Only items in DEPT_ACCEPTED_PENDING_ADMIN status
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    dept_id = test_data['dept_id']
    # Administration sees all units, so we give broad access
    user = MockUser(3, 'admin_admin', 'ADMINISTRATION_ADMIN', allowed_unit_ids=[dept_id, test_data['section_id']])
    
    inbox = get_inbox(user)
    
    # Check that inbox contains items
    has_items = len(inbox) > 0
    print_test(
        "ADMINISTRATION_ADMIN receives inbox items",
        has_items,
        f"Received {len(inbox)} item(s)"
    )
    
    # Check that all items are DEPT_ACCEPTED_PENDING_ADMIN
    all_correct_status = all(item['status'] == 'DEPT_ACCEPTED_PENDING_ADMIN' for item in inbox)
    print_test(
        "All items are DEPT_ACCEPTED_PENDING_ADMIN status",
        all_correct_status,
        f"Statuses: {[item['status'] for item in inbox]}"
    )
    
    # Verify no items from other statuses leaked in
    other_statuses = [item['status'] for item in inbox if item['status'] != 'DEPT_ACCEPTED_PENDING_ADMIN']
    no_leakage = len(other_statuses) == 0
    print_test(
        "No cross-status leakage",
        no_leakage,
        f"Other statuses found: {other_statuses}" if other_statuses else "Clean"
    )
    
    return has_items and all_correct_status and no_leakage


def test_worker_inbox_empty(test_data):
    """
    TEST: WORKER inbox returns empty list
    EXPECT: Empty inbox, no exception
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    dept_id = test_data['dept_id']
    user = MockUser(4, 'worker', 'WORKER', allowed_unit_ids=[dept_id])
    
    try:
        inbox = get_inbox(user)
        is_empty = len(inbox) == 0
        is_list = isinstance(inbox, list)
        
        print_test(
            "WORKER receives empty inbox (no exception)",
            is_empty and is_list,
            f"Received: {type(inbox).__name__} with {len(inbox)} items"
        )
        
        return is_empty and is_list
        
    except Exception as e:
        print_test(
            "WORKER receives empty inbox (no exception)",
            False,
            f"Raised exception: {type(e).__name__}: {str(e)}"
        )
        return False


def test_complaint_supervisor_inbox_empty(test_data):
    """
    TEST: COMPLAINT_SUPERVISOR inbox returns empty list
    EXPECT: Empty inbox, no exception
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    dept_id = test_data['dept_id']
    user = MockUser(5, 'supervisor', 'COMPLAINT_SUPERVISOR', allowed_unit_ids=[dept_id])
    
    try:
        inbox = get_inbox(user)
        is_empty = len(inbox) == 0
        is_list = isinstance(inbox, list)
        
        print_test(
            "COMPLAINT_SUPERVISOR receives empty inbox (no exception)",
            is_empty and is_list,
            f"Received: {type(inbox).__name__} with {len(inbox)} items"
        )
        
        return is_empty and is_list
        
    except Exception as e:
        print_test(
            "COMPLAINT_SUPERVISOR receives empty inbox (no exception)",
            False,
            f"Raised exception: {type(e).__name__}: {str(e)}"
        )
        return False


def test_software_admin_inbox_empty(test_data):
    """
    TEST: SOFTWARE_ADMIN inbox returns empty list
    EXPECT: Empty inbox (no cross-stage override under Model A)
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    dept_id = test_data['dept_id']
    user = MockUser(6, 'software_admin', 'SOFTWARE_ADMIN', allowed_unit_ids=[dept_id, test_data['section_id']])
    
    try:
        inbox = get_inbox(user)
        is_empty = len(inbox) == 0
        is_list = isinstance(inbox, list)
        
        print_test(
            "SOFTWARE_ADMIN receives empty inbox (no exception)",
            is_empty and is_list,
            f"Received: {type(inbox).__name__} with {len(inbox)} items"
        )
        
        return is_empty and is_list
        
    except Exception as e:
        print_test(
            "SOFTWARE_ADMIN receives empty inbox (no exception)",
            False,
            f"Raised exception: {type(e).__name__}: {str(e)}"
        )
        return False


def test_no_unified_inbox_behavior(test_data):
    """
    TEST: Verify unified inbox behavior has been removed
    EXPECT: No role receives items from multiple workflow stages
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    print_section("Testing No Unified Inbox Behavior")
    
    roles_to_test = [
        ('SOFTWARE_ADMIN', MockUser(7, 'sw_admin', 'SOFTWARE_ADMIN', 
                                    allowed_unit_ids=[test_data['section_id'], test_data['dept_id']])),
        ('COMPLAINT_SUPERVISOR', MockUser(8, 'c_supervisor', 'COMPLAINT_SUPERVISOR',
                                          allowed_unit_ids=[test_data['section_id'], test_data['dept_id']])),
        ('WORKER', MockUser(9, 'worker', 'WORKER',
                           allowed_unit_ids=[test_data['section_id'], test_data['dept_id']])),
    ]
    
    all_passed = True
    
    for role_name, user in roles_to_test:
        inbox = get_inbox(user)
        
        # Get unique statuses in inbox
        statuses = set(item['status'] for item in inbox)
        
        # Should have 0 or 1 status (never multiple workflow stages)
        has_single_or_no_status = len(statuses) <= 1
        
        print_test(
            f"{role_name} does not have unified multi-stage inbox",
            has_single_or_no_status,
            f"Unique statuses in inbox: {len(statuses)} - {statuses if statuses else 'Empty'}"
        )
        
        all_passed = all_passed and has_single_or_no_status
    
    return all_passed


def test_response_schema_unchanged(test_data):
    """
    TEST: Verify response schema has not changed
    EXPECT: Inbox items still have expected fields
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    section_id = test_data['section_id']
    user = MockUser(10, 'section_admin', 'SECTION_ADMIN', allowed_unit_ids=[section_id])
    
    inbox = get_inbox(user)
    
    if len(inbox) == 0:
        print_test("Response schema check", False, "No items in inbox to verify")
        return False
    
    item = inbox[0]
    
    expected_fields = [
        'subcase_id',
        'case_type',
        'incident_id',
        'seasonal_report_id',
        'target_org_unit_id',
        'status',
        'created_at',
        'allowed_actions'
    ]
    
    has_all_fields = all(field in item for field in expected_fields)
    missing_fields = [field for field in expected_fields if field not in item]
    
    print_test(
        "Response schema unchanged",
        has_all_fields,
        f"Missing fields: {missing_fields}" if missing_fields else "All expected fields present"
    )
    
    return has_all_fields


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all inbox routing tests"""
    print("\n" + "="*80)
    print(" INBOX STRICT ROUTING TEST SUITE (MODEL A)")
    print("="*80)
    
    test_data = None
    
    try:
        # Setup test data
        test_data = setup_test_data()
        
        # Run tests
        results = {}
        
        print_section("STRICT ROLE ROUTING TESTS")
        results['section_admin'] = test_section_admin_inbox_strict(test_data)
        results['department_admin'] = test_department_admin_inbox_strict(test_data)
        results['administration_admin'] = test_administration_admin_inbox_strict(test_data)
        
        print_section("NON-RESPONSIBLE ROLE TESTS (Empty Inbox)")
        results['worker'] = test_worker_inbox_empty(test_data)
        results['complaint_supervisor'] = test_complaint_supervisor_inbox_empty(test_data)
        results['software_admin'] = test_software_admin_inbox_empty(test_data)
        
        print_section("BEHAVIORAL VERIFICATION TESTS")
        results['no_unified'] = test_no_unified_inbox_behavior(test_data)
        results['schema_unchanged'] = test_response_schema_unchanged(test_data)
        
        # Summary
        print_section("TEST SUMMARY")
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"\nPassed: {passed}/{total} tests")
        
        for test_name, result in results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {test_name}")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED - Strict routing verified!")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed")
        
        return passed == total
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if test_data:
            cleanup_test_data(test_data)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
