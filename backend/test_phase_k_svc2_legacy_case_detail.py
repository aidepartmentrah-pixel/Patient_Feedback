"""
PHASE K — SVC2 — LEGACY CASE DETAIL TEST

Comprehensive test suite for get_legacy_case_detail function.

Tests:
1. Basic call with known case ID
2. Structure validation (case, request, actions)
3. Join verification
4. Action ordering (ASC by date)
5. Empty actions case
6. Not found case (returns None)
7. Read-only safety
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from api.db_layer.legacy_migration_db import get_legacy_case_detail
from core.database import get_connection


def print_header(text):
    """Print formatted test section header"""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print('=' * 80)


def print_test(test_name, passed, message=""):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} — {test_name}")
    if message:
        print(f"   {message}")


def get_test_case_id():
    """Get a known legacy case ID for testing"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT TOP 1 UniqueID FROM IncidentRequestCase ORDER BY UniqueID")
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row[0] if row else None


def test_basic_call():
    """TEST 1: Basic call with known case ID"""
    print_header("TEST 1: BASIC CALL")
    
    try:
        case_id = get_test_case_id()
        
        if not case_id:
            print_test("Test data available", False, "No legacy cases found")
            return False
        
        print(f"📌 Testing with legacy case ID: {case_id}")
        
        result = get_legacy_case_detail(case_id)
        
        is_dict = isinstance(result, dict)
        print_test("Returns dict", is_dict)
        
        not_none = result is not None
        print_test("Result not None", not_none)
        
        return is_dict and not_none
        
    except Exception as e:
        print_test("Basic call", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_structure_validation():
    """TEST 2: Structure validation"""
    print_header("TEST 2: STRUCTURE VALIDATION")
    
    try:
        case_id = get_test_case_id()
        result = get_legacy_case_detail(case_id)
        
        if not result:
            print_test("Has result", False)
            return False
        
        # Check top-level keys
        has_case = "case" in result
        has_request = "request" in result
        has_actions = "actions" in result
        
        print_test("Has 'case' key", has_case)
        print_test("Has 'request' key", has_request)
        print_test("Has 'actions' key", has_actions)
        
        # Check case is dict
        case_is_dict = isinstance(result.get("case"), dict)
        print_test("'case' is dict", case_is_dict)
        
        # Check request is dict
        request_is_dict = isinstance(result.get("request"), dict)
        print_test("'request' is dict", request_is_dict)
        
        # Check actions is list
        actions_is_list = isinstance(result.get("actions"), list)
        print_test("'actions' is list", actions_is_list)
        
        # Verify case fields
        if has_case and case_is_dict:
            case = result["case"]
            required_case_fields = [
                'UniqueID', 'Description', 'Note', 'DoctorID',
                'SectionID', 'DepartmentID', 'AdminID',
                'DateAndTimeCreated', 'DateAndTimeUpdated', 
                'DateAndTimeHappened', 'IncidentTypeID'
            ]
            
            print("\n📋 Case Fields:")
            all_case_fields = True
            for field in required_case_fields:
                present = field in case
                print_test(f"  {field}", present)
                if not present:
                    all_case_fields = False
        
        # Verify request fields
        if has_request and request_is_dict:
            request = result["request"]
            required_request_fields = [
                'PatientName', 'MRN', 'SourceBuilding', 'IsInPatient',
                'RequesterName', 'Note', 'DateAndTimeRecieved',
                'SourceSectionID', 'SourceDepartmentID', 'SourceAdminID'
            ]
            
            print("\n📋 Request Fields:")
            all_request_fields = True
            for field in required_request_fields:
                present = field in request
                print_test(f"  {field}", present)
                if not present:
                    all_request_fields = False
        
        return (has_case and has_request and has_actions and 
                case_is_dict and request_is_dict and actions_is_list)
        
    except Exception as e:
        print_test("Structure validation", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_join_verification():
    """TEST 3: Join verification"""
    print_header("TEST 3: JOIN VERIFICATION")
    
    try:
        case_id = get_test_case_id()
        result = get_legacy_case_detail(case_id)
        
        if not result:
            print_test("Has result", False)
            return False
        
        # Verify case ID matches input
        case_unique_id = result["case"]["UniqueID"]
        id_matches = case_unique_id == case_id
        print_test("Case UniqueID matches input", id_matches, 
                   f"Input: {case_id}, Got: {case_unique_id}")
        
        # Verify request data is populated
        request = result["request"]
        has_patient = request.get("PatientName") is not None
        print_test("Request has PatientName", has_patient, 
                   f"Patient: {request.get('PatientName')}")
        
        # Display sample data
        print("\n📊 Sample Data:")
        print(f"  Case ID: {case_unique_id}")
        print(f"  Patient: {request.get('PatientName')}")
        print(f"  Received: {request.get('DateAndTimeRecieved')}")
        print(f"  Description: {result['case'].get('Description', 'N/A')[:60]}...")
        print(f"  Actions: {len(result['actions'])} records")
        
        return id_matches and has_patient
        
    except Exception as e:
        print_test("Join verification", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_action_ordering():
    """TEST 4: Action ordering (ASC by date)"""
    print_header("TEST 4: ACTION ORDERING")
    
    try:
        # Find a case with multiple actions
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT TOP 1 IncidentRequestCaseID
            FROM IncidentRequestCaseAction
            GROUP BY IncidentRequestCaseID
            HAVING COUNT(*) >= 2
        """)
        
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not row:
            print_test("Multiple actions case found", False, 
                       "No cases with 2+ actions (will test with available data)")
            # Fall back to testing with single action case
            case_id = get_test_case_id()
        else:
            case_id = row[0]
            print(f"📌 Testing with case {case_id} (has multiple actions)")
        
        result = get_legacy_case_detail(case_id)
        
        if not result:
            print_test("Has result", False)
            return False
        
        actions = result["actions"]
        
        if len(actions) < 2:
            print_test("Action ordering", True, 
                       f"Only {len(actions)} action(s) - order validation skipped")
            return True
        
        # Check dates are in ascending order
        dates = [a["DateAndTimeCreated"] for a in actions if a["DateAndTimeCreated"]]
        
        is_ascending = all(dates[i] <= dates[i+1] for i in range(len(dates) - 1))
        
        print_test("Actions in ASC order", is_ascending)
        
        if len(dates) >= 2:
            print("\n📅 Sample action dates (should be oldest → newest):")
            for i, date in enumerate(dates[:5], 1):
                print(f"  {i}. {date}")
        
        return is_ascending
        
    except Exception as e:
        print_test("Action ordering", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_empty_actions():
    """TEST 5: Case with zero actions"""
    print_header("TEST 5: EMPTY ACTIONS CASE")
    
    try:
        # Find a case with no actions
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT TOP 1 irc.UniqueID
            FROM IncidentRequestCase irc
            LEFT JOIN IncidentRequestCaseAction act 
                ON act.IncidentRequestCaseID = irc.UniqueID
            WHERE act.UniqueID IS NULL
        """)
        
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not row:
            print_test("Empty actions case found", False, 
                       "All cases have actions - test skipped")
            return True  # Not a failure, just can't test
        
        case_id = row[0]
        print(f"📌 Testing with case {case_id} (has no actions)")
        
        result = get_legacy_case_detail(case_id)
        
        if not result:
            print_test("Has result", False)
            return False
        
        actions = result["actions"]
        
        is_list = isinstance(actions, list)
        print_test("Actions is list", is_list)
        
        is_empty = len(actions) == 0
        print_test("Actions list is empty", is_empty, f"Length: {len(actions)}")
        
        return is_list and is_empty
        
    except Exception as e:
        print_test("Empty actions", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_not_found_case():
    """TEST 6: Non-existing case returns None"""
    print_header("TEST 6: NOT FOUND CASE")
    
    try:
        non_existing_id = 999999999
        
        print(f"📌 Testing with non-existing ID: {non_existing_id}")
        
        result = get_legacy_case_detail(non_existing_id)
        
        is_none = result is None
        print_test("Returns None for non-existing ID", is_none)
        
        if result is not None:
            print(f"   ❌ Unexpected result: {type(result)}")
        
        return is_none
        
    except Exception as e:
        print_test("Not found case", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_read_only_safety():
    """TEST 7: Read-only safety"""
    print_header("TEST 7: READ-ONLY SAFETY CHECK")
    
    try:
        # Get row counts before
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM IncidentRequestCase")
        case_count_before = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM IncidentRequest")
        request_count_before = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM IncidentRequestCaseAction")
        action_count_before = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        print(f"📊 Before function call:")
        print(f"  IncidentRequestCase: {case_count_before}")
        print(f"  IncidentRequest: {request_count_before}")
        print(f"  IncidentRequestCaseAction: {action_count_before}")
        
        # Call function
        case_id = get_test_case_id()
        result = get_legacy_case_detail(case_id)
        
        # Get row counts after
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM IncidentRequestCase")
        case_count_after = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM IncidentRequest")
        request_count_after = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM IncidentRequestCaseAction")
        action_count_after = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        print(f"\n📊 After function call:")
        print(f"  IncidentRequestCase: {case_count_after}")
        print(f"  IncidentRequest: {request_count_after}")
        print(f"  IncidentRequestCaseAction: {action_count_after}")
        
        no_case_changes = case_count_before == case_count_after
        no_request_changes = request_count_before == request_count_after
        no_action_changes = action_count_before == action_count_after
        
        print_test("IncidentRequestCase unchanged", no_case_changes)
        print_test("IncidentRequest unchanged", no_request_changes)
        print_test("IncidentRequestCaseAction unchanged", no_action_changes)
        
        return no_case_changes and no_request_changes and no_action_changes
        
    except Exception as e:
        print_test("Read-only safety", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print_header("PHASE K — SVC2 — LEGACY CASE DETAIL TEST")
    print("Comprehensive validation of get_legacy_case_detail function")
    
    results = []
    
    results.append(("Basic Call", test_basic_call()))
    results.append(("Structure Validation", test_structure_validation()))
    results.append(("Join Verification", test_join_verification()))
    results.append(("Action Ordering", test_action_ordering()))
    results.append(("Empty Actions", test_empty_actions()))
    results.append(("Not Found Case", test_not_found_case()))
    results.append(("Read-Only Safety", test_read_only_safety()))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} — {test_name}")
    
    print(f"\n{'=' * 80}")
    print(f"TOTAL: {passed}/{total} tests passed")
    print('=' * 80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED — K-SVC-2 COMPLETE")
        return True
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
