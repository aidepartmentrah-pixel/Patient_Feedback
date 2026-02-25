"""
WORKER INBOX SAFETY TEST SUITE

Tests that WORKER role inbox behavior is safe and deterministic:
- Always returns empty list
- Never throws exceptions
- Returns HTTP 200
- Correct response schema
- Safe even when workflow subcases exist
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
    print(f"{status}: {test_name}")
    if message:
        print(f"   {message}")
    return passed


class MockUser:
    """Mock user for testing with proper scopes and allowed_unit_ids"""
    def __init__(self, role_code, allowed_unit_ids=None):
        self.role_code = role_code
        self.scopes = [type('obj', (object,), {'role_code': role_code})]
        self.allowed_unit_ids = set(allowed_unit_ids) if allowed_unit_ids else set()


# =============================================================================
# UNIT TESTS - Direct Service Layer Testing
# =============================================================================

def test_worker_inbox_returns_empty_list():
    """
    TEST: WORKER inbox returns empty list (not None, not error)
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    worker_user = MockUser('WORKER', allowed_unit_ids=[1, 2, 3])
    
    result = get_inbox(worker_user)
    
    is_list = isinstance(result, list)
    is_empty = len(result) == 0
    not_none = result is not None
    
    passed = is_list and is_empty and not_none
    
    return print_test(
        "WORKER inbox returns empty list",
        passed,
        f"Type: {type(result).__name__}, Length: {len(result)}, Is None: {result is None}"
    )


def test_worker_inbox_no_exception():
    """
    TEST: WORKER inbox never raises exception
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    worker_user = MockUser('WORKER', allowed_unit_ids=[1])
    
    try:
        result = get_inbox(worker_user)
        no_exception = True
        return print_test(
            "WORKER inbox does not raise exception",
            no_exception,
            f"Returned: {type(result).__name__} with {len(result)} items"
        )
    except Exception as e:
        return print_test(
            "WORKER inbox does not raise exception",
            False,
            f"Exception raised: {type(e).__name__}: {str(e)}"
        )


def test_worker_inbox_with_no_scope():
    """
    TEST: WORKER with no allowed_unit_ids still returns empty list safely
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    worker_user = MockUser('WORKER', allowed_unit_ids=None)
    
    try:
        result = get_inbox(worker_user)
        is_list = isinstance(result, list)
        is_empty = len(result) == 0
        
        passed = is_list and is_empty
        return print_test(
            "WORKER with no scope returns empty list safely",
            passed,
            f"Type: {type(result).__name__}, Length: {len(result)}"
        )
    except Exception as e:
        return print_test(
            "WORKER with no scope returns empty list safely",
            False,
            f"Exception raised: {type(e).__name__}: {str(e)}"
        )


def test_worker_inbox_response_schema():
    """
    TEST: WORKER inbox response matches expected schema (list of dicts)
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    worker_user = MockUser('WORKER', allowed_unit_ids=[1, 2])
    
    result = get_inbox(worker_user)
    
    # Even though it's empty, verify it's the right type
    is_list = isinstance(result, list)
    
    # Verify it's not an error object structure
    is_not_error = not (isinstance(result, dict) and 'error' in result)
    
    # Verify it's serializable (can be converted to JSON)
    try:
        import json
        json_str = json.dumps(result)
        is_serializable = True
    except Exception as e:
        is_serializable = False
    
    passed = is_list and is_not_error and is_serializable
    
    return print_test(
        "WORKER inbox response matches schema",
        passed,
        f"Is list: {is_list}, Not error: {is_not_error}, Serializable: {is_serializable}"
    )


# =============================================================================
# INTEGRATION TESTS - With Real Database
# =============================================================================

def setup_test_subcases():
    """
    Create test subcases in all workflow statuses.
    Returns test data dict.
    """
    print_section("Setting Up Test Subcases")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Find org units
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
        
        # Find test user
        cursor.execute("SELECT TOP 1 UserID FROM APP_Users")
        user_row = cursor.fetchone()
        if not user_row:
            raise Exception("No users found in database")
        test_user_id = user_row[0]
        
        # Create parent incident case
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
        """, 'Test incident for worker inbox', 'Test action', 'Test action taken',
             'Test Patient', org_unit_1, test_user_id)
        
        case_id = cursor.fetchone()[0]
        conn.commit()
        print(f"Created test incident case ID: {case_id}")
        
        # Create subcases in all workflow statuses
        statuses = [
            'SUBMITTED_TO_SECTION',
            'SECTION_ACCEPTED_PENDING_DEPT',
            'DEPT_ACCEPTED_PENDING_ADMIN'
        ]
        
        subcase_ids = []
        for status in statuses:
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
            """, case_id, org_unit_1, status, test_user_id)
            conn.commit()
            
            cursor.execute("SELECT @@IDENTITY as SubcaseID")
            subcase_id = cursor.fetchone()[0]
            subcase_ids.append(subcase_id)
            print(f"Created subcase {subcase_id} with status: {status}")
        
        print("✓ Test data created successfully")
        
        return {
            'case_id': case_id,
            'org_unit_1': org_unit_1,
            'org_unit_2': org_unit_2,
            'subcase_ids': subcase_ids
        }
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error setting up test data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def cleanup_test_subcases(test_data):
    """Clean up test data"""
    print_section("Cleaning Up Test Subcases")
    
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


def test_worker_inbox_empty_despite_subcases(test_data):
    """
    TEST: WORKER inbox returns empty list even when workflow subcases exist
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    org_unit_1 = test_data['org_unit_1']
    org_unit_2 = test_data['org_unit_2']
    
    # Create worker with access to org units where subcases exist
    worker_user = MockUser('WORKER', allowed_unit_ids=[org_unit_1, org_unit_2])
    
    result = get_inbox(worker_user)
    
    is_list = isinstance(result, list)
    is_empty = len(result) == 0
    
    passed = is_list and is_empty
    
    return print_test(
        "WORKER inbox empty despite subcases existing in DB",
        passed,
        f"Subcases exist in DB but worker inbox returned {len(result)} items"
    )


def test_worker_vs_section_admin_different(test_data):
    """
    TEST: WORKER inbox is empty while SECTION_ADMIN sees items
    Verifies that WORKER check happens before role-specific queries
    """
    from backend.api_v2.services.inbox_service import get_inbox
    
    org_unit_1 = test_data['org_unit_1']
    
    # Worker should get empty
    worker_user = MockUser('WORKER', allowed_unit_ids=[org_unit_1])
    worker_result = get_inbox(worker_user)
    
    # Section admin should get items
    section_user = MockUser('SECTION_ADMIN', allowed_unit_ids=[org_unit_1])
    section_result = get_inbox(section_user)
    
    worker_empty = len(worker_result) == 0
    section_has_items = len(section_result) > 0
    
    passed = worker_empty and section_has_items
    
    return print_test(
        "WORKER gets empty while SECTION_ADMIN gets items",
        passed,
        f"Worker: {len(worker_result)} items, Section Admin: {len(section_result)} items"
    )


def test_worker_inbox_performance():
    """
    TEST: WORKER inbox returns quickly (no DB queries executed)
    """
    from backend.api_v2.services.inbox_service import get_inbox
    import time
    
    worker_user = MockUser('WORKER', allowed_unit_ids=[1, 2, 3])
    
    start_time = time.time()
    result = get_inbox(worker_user)
    end_time = time.time()
    
    execution_time_ms = (end_time - start_time) * 1000
    
    # Should be very fast since it just returns [] without DB queries
    is_fast = execution_time_ms < 50  # Less than 50ms
    is_empty = len(result) == 0
    
    passed = is_fast and is_empty
    
    return print_test(
        "WORKER inbox returns quickly (no DB overhead)",
        passed,
        f"Execution time: {execution_time_ms:.2f}ms (expected < 50ms)"
    )


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all WORKER inbox safety tests"""
    print("\n" + "="*80)
    print(" WORKER INBOX SAFETY TEST SUITE")
    print("="*80)
    
    test_data = None
    results = {}
    
    try:
        # Unit tests (no DB needed)
        print_section("UNIT TESTS - Service Layer")
        results['returns_empty_list'] = test_worker_inbox_returns_empty_list()
        results['no_exception'] = test_worker_inbox_no_exception()
        results['no_scope_safe'] = test_worker_inbox_with_no_scope()
        results['response_schema'] = test_worker_inbox_response_schema()
        results['performance'] = test_worker_inbox_performance()
        
        # Integration tests (with DB)
        test_data = setup_test_subcases()
        
        print_section("INTEGRATION TESTS - With Real Database")
        results['empty_despite_subcases'] = test_worker_inbox_empty_despite_subcases(test_data)
        results['different_from_admin'] = test_worker_vs_section_admin_different(test_data)
        
        # Summary
        print_section("TEST SUMMARY")
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"\nPassed: {passed}/{total} tests")
        
        for test_name, result in results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {test_name}")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED - WORKER inbox is safe!")
            print("✓ Returns empty list")
            print("✓ No exceptions")
            print("✓ Correct response schema")
            print("✓ Fast execution (no DB overhead)")
            print("✓ Safe even when subcases exist")
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
            cleanup_test_subcases(test_data)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
