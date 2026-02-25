"""
STATUS ROLE MAP ENFORCEMENT TEST SUITE

Tests that STATUS_ROLE_MAP correctly drives inbox visibility:
- Each status appears in exactly one role's inbox
- No status leaks to incorrect roles
- Terminal statuses never appear in any inbox
- Map is correctly enforced by DB layer queries
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

def setup_status_test_data():
    """
    Create test subcases for EACH workflow status to verify mapping.
    Returns dict with subcase_id for each status.
    """
    print_section("Setting Up Status Test Data")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Find 2 valid org units
        cursor.execute("""
            SELECT TOP 2 UniqueID, Name
            FROM AdminsrationUnit
            WHERE Frozen = 0
            ORDER BY UniqueID
        """)
        org_units = cursor.fetchall()
        
        if len(org_units) < 2:
            raise Exception("Need at least 2 org units for testing")
        
        org_unit_1 = org_units[0][0]
        org_unit_2 = org_units[1][0]
        print(f"Using Org Unit 1: {org_units[0][1]} (ID: {org_unit_1})")
        print(f"Using Org Unit 2: {org_units[1][1]} (ID: {org_unit_2})")
        
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
        """, 'Test incident for status mapping', 'Test action', 'Test action taken',
             'Test Patient', org_unit_1, test_user_id)
        
        case_id = cursor.fetchone()[0]
        print(f"Created test incident case ID: {case_id}")
        conn.commit()
        
        # Define all statuses to test
        # Format: (status, target_org_unit, expected_role, is_terminal)
        all_statuses = [
            # Section-level statuses
            ('SUBMITTED_TO_SECTION', org_unit_1, 'SECTION_ADMIN', False),
            ('RETURNED_TO_SECTION_FOR_REVISION', org_unit_1, 'SECTION_ADMIN', False),
            
            # Department-level statuses
            ('SECTION_ACCEPTED_PENDING_DEPT', org_unit_2, 'DEPARTMENT_ADMIN', False),
            ('RETURNED_TO_DEPT_FOR_REVISION', org_unit_2, 'DEPARTMENT_ADMIN', False),
            
            # Administration-level statuses
            ('DEPT_ACCEPTED_PENDING_ADMIN', org_unit_2, 'ADMINISTRATION_ADMIN', False),
            
            # Terminal statuses (should NEVER appear in inbox)
            ('ADMIN_APPROVED', org_unit_2, None, True),
            ('SECTION_DENIED', org_unit_1, None, True),
            ('FORCE_CLOSED', org_unit_1, None, True),
        ]
        
        # Create subcases for each status
        test_data = {
            'case_id': case_id,
            'org_unit_1': org_unit_1,
            'org_unit_2': org_unit_2,
            'subcases': {}
        }
        
        for status, target_org_unit_id, expected_role, is_terminal in all_statuses:
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
            
            test_data['subcases'][status] = {
                'subcase_id': subcase_id,
                'target_org_unit_id': target_org_unit_id,
                'expected_role': expected_role,
                'is_terminal': is_terminal
            }
            
            terminal_flag = " (TERMINAL)" if is_terminal else ""
            print(f"Created subcase {subcase_id} with status: {status}{terminal_flag}")
        
        print(f"\n✓ Test data created successfully")
        return test_data
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error setting up test data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def cleanup_status_test_data(test_data):
    """Clean up test data after tests complete"""
    print_section("Cleaning Up Status Test Data")
    
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
# STATUS MAPPING TESTS
# =============================================================================

def test_status_role_map_exists():
    """
    TEST: STATUS_ROLE_MAP constant exists and has expected structure
    """
    from backend.api_v2.services.inbox_service import STATUS_ROLE_MAP
    
    has_map = STATUS_ROLE_MAP is not None
    print_test(
        "STATUS_ROLE_MAP constant exists",
        has_map,
        f"Found: {STATUS_ROLE_MAP}"
    )
    
    # Verify expected roles exist
    expected_roles = ['SECTION_ADMIN', 'DEPARTMENT_ADMIN', 'ADMINISTRATION_ADMIN']
    has_all_roles = all(role in STATUS_ROLE_MAP for role in expected_roles)
    print_test(
        "STATUS_ROLE_MAP has all expected roles",
        has_all_roles,
        f"Roles: {list(STATUS_ROLE_MAP.keys())}"
    )
    
    # Verify section admin statuses
    section_statuses = STATUS_ROLE_MAP.get('SECTION_ADMIN', [])
    has_section_statuses = (
        'SUBMITTED_TO_SECTION' in section_statuses and
        'RETURNED_TO_SECTION_FOR_REVISION' in section_statuses
    )
    print_test(
        "SECTION_ADMIN has correct statuses",
        has_section_statuses,
        f"Statuses: {section_statuses}"
    )
    
    # Verify department admin statuses
    dept_statuses = STATUS_ROLE_MAP.get('DEPARTMENT_ADMIN', [])
    has_dept_statuses = (
        'SECTION_ACCEPTED_PENDING_DEPT' in dept_statuses and
        'RETURNED_TO_DEPT_FOR_REVISION' in dept_statuses
    )
    print_test(
        "DEPARTMENT_ADMIN has correct statuses",
        has_dept_statuses,
        f"Statuses: {dept_statuses}"
    )
    
    # Verify admin statuses
    admin_statuses = STATUS_ROLE_MAP.get('ADMINISTRATION_ADMIN', [])
    has_admin_statuses = 'DEPT_ACCEPTED_PENDING_ADMIN' in admin_statuses
    print_test(
        "ADMINISTRATION_ADMIN has correct statuses",
        has_admin_statuses,
        f"Statuses: {admin_statuses}"
    )
    
    # Verify terminal statuses NOT in map
    terminal_statuses = ['ADMIN_APPROVED', 'SECTION_DENIED', 'FORCE_CLOSED', 'CLOSED']
    all_mapped_statuses = []
    for statuses in STATUS_ROLE_MAP.values():
        all_mapped_statuses.extend(statuses)
    
    no_terminal = not any(status in all_mapped_statuses for status in terminal_statuses)
    print_test(
        "Terminal statuses NOT in STATUS_ROLE_MAP",
        no_terminal,
        f"All mapped: {all_mapped_statuses}"
    )
    
    return has_map and has_all_roles and has_section_statuses and has_dept_statuses and has_admin_statuses and no_terminal


def test_status_visibility_matrix(test_data):
    """
    TEST: Each status appears in exactly one role's inbox (or none for terminal)
    
    This is a comprehensive matrix test:
    - For each status, check all 3 role inboxes
    - Verify only the expected role sees it
    - Verify terminal statuses never appear
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    print_section("Status Visibility Matrix Test")
    
    org_unit_1 = test_data['org_unit_1']
    org_unit_2 = test_data['org_unit_2']
    
    # Create users for all 3 roles
    roles = {
        'SECTION_ADMIN': MockUser(1, 'section_admin', 'SECTION_ADMIN', 
                                   allowed_unit_ids=[org_unit_1, org_unit_2]),
        'DEPARTMENT_ADMIN': MockUser(2, 'dept_admin', 'DEPARTMENT_ADMIN', 
                                      allowed_unit_ids=[org_unit_1, org_unit_2]),
        'ADMINISTRATION_ADMIN': MockUser(3, 'admin_admin', 'ADMINISTRATION_ADMIN', 
                                         allowed_unit_ids=[org_unit_1, org_unit_2])
    }
    
    all_passed = True
    
    # Test each status
    for status, subcase_info in test_data['subcases'].items():
        subcase_id = subcase_info['subcase_id']
        expected_role = subcase_info['expected_role']
        is_terminal = subcase_info['is_terminal']
        
        print(f"\n  Testing status: {status}")
        
        # Check each role's inbox
        visibility_results = {}
        for role_name, user in roles.items():
            inbox = get_inbox(user)
            subcase_ids = [item['subcase_id'] for item in inbox]
            is_visible = subcase_id in subcase_ids
            visibility_results[role_name] = is_visible
            
            if is_visible:
                print(f"    ✓ Visible to {role_name}")
        
        # Verify expectations
        if is_terminal:
            # Terminal statuses should be invisible to ALL roles
            any_visible = any(visibility_results.values())
            passed = not any_visible
            
            print_test(
                f"{status} not visible to any role (terminal)",
                passed,
                f"Visibility: {visibility_results}"
            )
            all_passed = all_passed and passed
            
        else:
            # Non-terminal: exactly one role should see it
            visible_to = [role for role, visible in visibility_results.items() if visible]
            
            exactly_one = len(visible_to) == 1
            correct_role = visible_to == [expected_role] if exactly_one else False
            
            passed = exactly_one and correct_role
            
            print_test(
                f"{status} visible only to {expected_role}",
                passed,
                f"Actually visible to: {visible_to if visible_to else 'none'}"
            )
            all_passed = all_passed and passed
    
    return all_passed


def test_returned_for_revision_statuses(test_data):
    """
    TEST: RETURNED_TO_*_FOR_REVISION statuses appear in correct role inboxes
    
    These are special revision statuses that should be treated the same
    as initial submission statuses for inbox routing.
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    print_section("Returned For Revision Status Test")
    
    org_unit_1 = test_data['org_unit_1']
    org_unit_2 = test_data['org_unit_2']
    
    # Test RETURNED_TO_SECTION_FOR_REVISION → SECTION_ADMIN
    section_user = MockUser(1, 'section_admin', 'SECTION_ADMIN', 
                            allowed_unit_ids=[org_unit_1, org_unit_2])
    section_inbox = get_inbox(section_user)
    section_statuses = set(item['status'] for item in section_inbox)
    
    has_returned_section = 'RETURNED_TO_SECTION_FOR_REVISION' in section_statuses
    print_test(
        "SECTION_ADMIN sees RETURNED_TO_SECTION_FOR_REVISION",
        has_returned_section,
        f"Statuses in section inbox: {section_statuses}"
    )
    
    # Test RETURNED_TO_DEPT_FOR_REVISION → DEPARTMENT_ADMIN
    dept_user = MockUser(2, 'dept_admin', 'DEPARTMENT_ADMIN', 
                         allowed_unit_ids=[org_unit_1, org_unit_2])
    dept_inbox = get_inbox(dept_user)
    dept_statuses = set(item['status'] for item in dept_inbox)
    
    has_returned_dept = 'RETURNED_TO_DEPT_FOR_REVISION' in dept_statuses
    print_test(
        "DEPARTMENT_ADMIN sees RETURNED_TO_DEPT_FOR_REVISION",
        has_returned_dept,
        f"Statuses in dept inbox: {dept_statuses}"
    )
    
    # Verify revision statuses don't leak to wrong roles
    admin_user = MockUser(3, 'admin_admin', 'ADMINISTRATION_ADMIN', 
                          allowed_unit_ids=[org_unit_1, org_unit_2])
    admin_inbox = get_inbox(admin_user)
    admin_statuses = set(item['status'] for item in admin_inbox)
    
    no_revision_leakage = (
        'RETURNED_TO_SECTION_FOR_REVISION' not in admin_statuses and
        'RETURNED_TO_DEPT_FOR_REVISION' not in admin_statuses
    )
    print_test(
        "ADMINISTRATION_ADMIN does not see revision statuses",
        no_revision_leakage,
        f"Statuses in admin inbox: {admin_statuses}"
    )
    
    return has_returned_section and has_returned_dept and no_revision_leakage


def test_terminal_statuses_never_in_inbox(test_data):
    """
    TEST: Terminal statuses (ADMIN_APPROVED, SECTION_DENIED, FORCE_CLOSED)
    never appear in ANY inbox
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    print_section("Terminal Status Exclusion Test")
    
    org_unit_1 = test_data['org_unit_1']
    org_unit_2 = test_data['org_unit_2']
    
    # Get all terminal subcase IDs
    terminal_subcase_ids = set()
    for status, info in test_data['subcases'].items():
        if info['is_terminal']:
            terminal_subcase_ids.add(info['subcase_id'])
    
    print(f"Terminal subcase IDs: {terminal_subcase_ids}")
    
    # Check all role inboxes
    roles = [
        ('SECTION_ADMIN', MockUser(1, 'sa', 'SECTION_ADMIN', allowed_unit_ids=[org_unit_1, org_unit_2])),
        ('DEPARTMENT_ADMIN', MockUser(2, 'da', 'DEPARTMENT_ADMIN', allowed_unit_ids=[org_unit_1, org_unit_2])),
        ('ADMINISTRATION_ADMIN', MockUser(3, 'aa', 'ADMINISTRATION_ADMIN', allowed_unit_ids=[org_unit_1, org_unit_2]))
    ]
    
    all_clean = True
    
    for role_name, user in roles:
        inbox = get_inbox(user)
        inbox_subcase_ids = set(item['subcase_id'] for item in inbox)
        
        # Check if any terminal subcases leaked in
        leaked = terminal_subcase_ids & inbox_subcase_ids
        is_clean = len(leaked) == 0
        
        print_test(
            f"{role_name} inbox contains no terminal statuses",
            is_clean,
            f"Leaked IDs: {leaked}" if leaked else "Clean"
        )
        
        all_clean = all_clean and is_clean
    
    return all_clean


def test_db_layer_respects_mapping(test_data):
    """
    TEST: DB layer functions return correct statuses per STATUS_ROLE_MAP
    """
    from backend.api_v2.db_layer import administrative_subcase_db
    from backend.api_v2.services.inbox_service import STATUS_ROLE_MAP
    
    print_section("DB Layer Mapping Compliance Test")
    
    # Test section queries
    section_subcases = administrative_subcase_db.get_subcases_pending_for_section()
    section_statuses = set(sc['status'] for sc in section_subcases)
    expected_section = set(STATUS_ROLE_MAP['SECTION_ADMIN'])
    
    section_correct = section_statuses.issubset(expected_section)
    print_test(
        "get_subcases_pending_for_section returns only SECTION_ADMIN statuses",
        section_correct,
        f"Expected: {expected_section}, Got: {section_statuses}"
    )
    
    # Test department queries
    dept_subcases = administrative_subcase_db.get_subcases_pending_for_department()
    dept_statuses = set(sc['status'] for sc in dept_subcases)
    expected_dept = set(STATUS_ROLE_MAP['DEPARTMENT_ADMIN'])
    
    dept_correct = dept_statuses.issubset(expected_dept)
    print_test(
        "get_subcases_pending_for_department returns only DEPARTMENT_ADMIN statuses",
        dept_correct,
        f"Expected: {expected_dept}, Got: {dept_statuses}"
    )
    
    # Test administration queries
    admin_subcases = administrative_subcase_db.get_subcases_pending_for_administration()
    admin_statuses = set(sc['status'] for sc in admin_subcases)
    expected_admin = set(STATUS_ROLE_MAP['ADMINISTRATION_ADMIN'])
    
    admin_correct = admin_statuses.issubset(expected_admin)
    print_test(
        "get_subcases_pending_for_administration returns only ADMINISTRATION_ADMIN statuses",
        admin_correct,
        f"Expected: {expected_admin}, Got: {admin_statuses}"
    )
    
    return section_correct and dept_correct and admin_correct


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all status-role mapping tests"""
    print("\n" + "="*80)
    print(" STATUS → ROLE MAP ENFORCEMENT TEST SUITE")
    print("="*80)
    
    test_data = None
    
    try:
        # Test 1: Map structure (no test data needed)
        print_section("MAP STRUCTURE TESTS")
        results = {}
        results['map_structure'] = test_status_role_map_exists()
        
        # Setup test data for remaining tests
        test_data = setup_status_test_data()
        
        # Test 2: Status visibility matrix
        results['visibility_matrix'] = test_status_visibility_matrix(test_data)
        
        # Test 3: Returned for revision statuses
        results['revision_statuses'] = test_returned_for_revision_statuses(test_data)
        
        # Test 4: Terminal status exclusion
        results['terminal_exclusion'] = test_terminal_statuses_never_in_inbox(test_data)
        
        # Test 5: DB layer compliance
        results['db_layer_compliance'] = test_db_layer_respects_mapping(test_data)
        
        # Summary
        print_section("TEST SUMMARY")
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"\nPassed: {passed}/{total} test groups")
        
        for test_name, result in results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {test_name}")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED - Status-role mapping verified!")
        else:
            print(f"\n⚠️  {total - passed} test group(s) failed")
        
        return passed == total
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if test_data:
            cleanup_status_test_data(test_data)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
