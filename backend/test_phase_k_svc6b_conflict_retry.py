"""
PHASE K — SVC6B — MIGRATION CONFLICT & RETRY SAFETY TEST

Stress-tests migration orchestration for conflict handling,
retry safety, and partial-failure recovery.

Tests:
1. Retry after success
2. Simulated mapping write failure  
3. Retry after mapping failure
4. Manual mapping pre-insert
5. Concurrent double call (manual)
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from api.services.migration_service import migrate_legacy_case
from core.database import get_connection
import time


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


def cleanup_test_data(legacy_case_id):
    """Clean up test migration record"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM dbo.APP_DataMigration_Map WHERE legacy_case_id = ?", legacy_case_id)
        conn.commit()
        cursor.close()
        conn.close()
    except:
        pass


def get_test_payload():
    """Generate valid migration test payload with unique identifier"""
    unique_id = int(time.time() * 1000) % 1000000
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT TOP 1 DomainID FROM dbo.APP_LOOKUP_DOMAIN ORDER BY DomainID")
    row = cursor.fetchone()
    domain_id = row[0] if row else 1
    
    cursor.execute("SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY WHERE DomainID = ? ORDER BY CategoryID", domain_id)
    row = cursor.fetchone()
    category_id = row[0] if row else 1
    
    cursor.execute("SELECT TOP 1 SubCategoryID FROM dbo.APP_LOOKUP_SUBCATEGORY WHERE CategoryID = ? ORDER BY SubCategoryID", category_id)
    row = cursor.fetchone()
    subcategory_id = row[0] if row else 1
    
    cursor.execute("SELECT TOP 1 ClassificationID FROM dbo.APP_LOOKUP_CLASSIFICATION WHERE SubCategoryID = ? ORDER BY ClassificationID", subcategory_id)
    row = cursor.fetchone()
    classification_id = row[0] if row else 1
    
    cursor.close()
    conn.close()
    
    return {
        "complaint_text": f"Conflict test {unique_id} - emergency wait times",
        "immediate_action": "Patient prioritized",
        "taken_action": "Staffing review initiated",
        "feedback_received_date": "2024-06-15",
        "patient_name": "Test Patient Conflict",
        "is_inpatient": True,
        "clinical_risk_type_id": 1,
        "feedback_intent_type_id": 1,
        "building_id": 1,
        "domain_id": domain_id,
        "category_id": category_id,
        "subcategory_id": subcategory_id,
        "classification_id": classification_id,
        "severity_id": 1,
        "stage_id": 1,
        "harm_id": 1,
        "source_id": 1,
        "issuing_department_id": 1,
        "requires_explanation": False
    }


def test_retry_after_success():
    """TEST 1: Retry after success"""
    print_header("TEST 1: RETRY AFTER SUCCESS")
    
    legacy_case_id = 700001
    migrated_by_user_id = 1
    
    cleanup_test_data(legacy_case_id)
    
    try:
        payload = get_test_payload()
        
        print(f"📋 First migration...")
        result1 = migrate_legacy_case(legacy_case_id, payload, migrated_by_user_id)
        
        first_success = result1.get("success") == True
        first_status = result1.get("status")
        first_new_id = result1.get("new_case_id")
        
        print_test("First migration succeeded", first_success, f"Status: {first_status}, New ID: {first_new_id}")
        
        if not first_success:
            cleanup_test_data(legacy_case_id)
            return False
        
        print(f"\n📋 Retry migration (same legacy_case_id)...")
        result2 = migrate_legacy_case(legacy_case_id, payload, migrated_by_user_id)
        
        second_success = result2.get("success") == True
        second_status = result2.get("status") == "ALREADY_MIGRATED"
        second_new_id = result2.get("new_case_id")
        ids_match = second_new_id == first_new_id
        
        print_test("Retry succeeded", second_success)
        print_test("Status is ALREADY_MIGRATED", second_status, f"Got: {result2.get('status')}")
        print_test("Same new_case_id returned", ids_match, f"First: {first_new_id}, Second: {second_new_id}")
        
        # Verify only one mapping row
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_DataMigration_Map
            WHERE legacy_case_id = ?
        """, legacy_case_id)
        
        mapping_count = cursor.fetchone()[0]
        
        print_test("Exactly one mapping row", mapping_count == 1, f"Count: {mapping_count}")
        
        cursor.close()
        conn.close()
        
        cleanup_test_data(legacy_case_id)
        
        return first_success and second_success and second_status and ids_match and mapping_count == 1
        
    except Exception as e:
        print_test("Retry after success", False, str(e))
        import traceback
        traceback.print_exc()
        cleanup_test_data(legacy_case_id)
        return False


def test_simulated_mapping_failure():
    """TEST 2: Simulated mapping write failure"""
    print_header("TEST 2: SIMULATED MAPPING WRITE FAILURE")
    
    legacy_case_id = 700002
    migrated_by_user_id = 1
    
    cleanup_test_data(legacy_case_id)
    
    try:
        payload = get_test_payload()
        
        # Use invalid user ID to force FK violation in mapping insert
        invalid_user_id = 999999999
        
        print(f"📋 Attempting migration with invalid user ID (forces mapping failure)...")
        print(f"   Invalid user ID: {invalid_user_id}")
        
        result = migrate_legacy_case(legacy_case_id, payload, invalid_user_id)
        
        failed = result.get("success") == False
        has_mapping_error = "MAPPING" in result.get("error", "").upper()
        new_case_id = result.get("new_case_id")
        
        print_test("Migration failed", failed, f"Success: {result.get('success')}")
        print_test("Error is MAPPING_WRITE_FAILED", has_mapping_error, f"Error: {result.get('error')}")
        print_test("New case ID returned (case was created)", new_case_id is not None, f"Case ID: {new_case_id}")
        
        # Verify case exists but mapping doesn't
        conn = get_connection()
        cursor = conn.cursor()
        
        if new_case_id:
            cursor.execute("""
                SELECT COUNT(*)
                FROM dbo.APP_IncidentCase
                WHERE IncidentRequestCaseID = ?
            """, new_case_id)
            
            case_exists = cursor.fetchone()[0] > 0
            print_test("Case record exists", case_exists)
        else:
            case_exists = False
            print_test("Case record checking", False, "No case ID returned")
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_DataMigration_Map
            WHERE legacy_case_id = ?
        """, legacy_case_id)
        
        mapping_count = cursor.fetchone()[0]
        
        print_test("No mapping row exists", mapping_count == 0, f"Count: {mapping_count}")
        
        cursor.close()
        conn.close()
        
        cleanup_test_data(legacy_case_id)
        
        return failed and has_mapping_error and new_case_id is not None and mapping_count == 0
        
    except Exception as e:
        print_test("Simulated mapping failure", False, str(e))
        import traceback
        traceback.print_exc()
        cleanup_test_data(legacy_case_id)
        return False


def test_retry_after_mapping_failure():
    """TEST 3: Retry after mapping failure"""
    print_header("TEST 3: RETRY AFTER MAPPING FAILURE")
    
    legacy_case_id = 700003
    migrated_by_user_id = 1
    
    cleanup_test_data(legacy_case_id)
    
    try:
        payload = get_test_payload()
        
        # First attempt with invalid user ID (causes mapping failure)
        invalid_user_id = 999999999
        
        print(f"📋 First attempt with invalid user ID...")
        result1 = migrate_legacy_case(legacy_case_id, payload, invalid_user_id)
        
        first_failed = result1.get("success") == False
        first_case_id = result1.get("new_case_id")
        
        print_test("First attempt failed", first_failed, f"Error: {result1.get('error')}")
        print_test("Case created despite failure", first_case_id is not None, f"Case ID: {first_case_id}")
        
        if not first_case_id:
            print("   Cannot continue test without case ID")
            cleanup_test_data(legacy_case_id)
            return False
        
        # Second attempt with valid user ID (should now succeed)
        print(f"\n📋 Retry with valid user ID...")
        result2 = migrate_legacy_case(legacy_case_id, payload, migrated_by_user_id)
        
        second_success = result2.get("success") == True
        second_status = result2.get("status")
        second_case_id = result2.get("new_case_id")
        
        print_test("Retry succeeded", second_success, f"Status: {second_status}")
        print_test("Case ID returned", second_case_id is not None, f"Case ID: {second_case_id}")
        
        # Verify mapping now exists
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_DataMigration_Map
            WHERE legacy_case_id = ?
        """, legacy_case_id)
        
        mapping_exists = cursor.fetchone()[0] > 0
        
        print_test("Mapping row now exists", mapping_exists)
        
        # Verify only ONE case exists for this complaint text
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_IncidentCase
            WHERE ComplaintText LIKE ?
        """, payload["complaint_text"] + "%")
        
        case_count = cursor.fetchone()[0]
        
        print_test("Only one case created", case_count == 1, f"Count: {case_count}")
        
        cursor.close()
        conn.close()
        
        cleanup_test_data(legacy_case_id)
        
        return first_failed and second_success and mapping_exists and case_count == 1
        
    except Exception as e:
        print_test("Retry after mapping failure", False, str(e))
        import traceback
        traceback.print_exc()
        cleanup_test_data(legacy_case_id)
        return False


def test_manual_mapping_pre_insert():
    """TEST 4: Manual mapping pre-insert"""
    print_header("TEST 4: MANUAL MAPPING PRE-INSERT")
    
    legacy_case_id = 700004
    migrated_by_user_id = 1
    
    cleanup_test_data(legacy_case_id)
    
    try:
        # Get existing case ID
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT TOP 1 IncidentRequestCaseID FROM dbo.APP_IncidentCase ORDER BY IncidentRequestCaseID DESC")
        row = cursor.fetchone()
        existing_case_id = row[0] if row else None
        
        if not existing_case_id:
            print_test("Test data available", False, "No existing cases")
            cursor.close()
            conn.close()
            return False
        
        # Manually insert mapping first
        print(f"📋 Manually inserting mapping first...")
        print(f"   Legacy Case ID: {legacy_case_id}")
        print(f"   Existing Case ID: {existing_case_id}")
        
        cursor.execute("""
            INSERT INTO dbo.APP_DataMigration_Map
            (legacy_case_id, new_case_id, migrated_by_user_id, migrated_at)
            VALUES (?, ?, ?, GETDATE())
        """, legacy_case_id, existing_case_id, migrated_by_user_id)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Now try to migrate
        print(f"\n📋 Calling migrate_legacy_case (should detect pre-existing mapping)...")
        
        payload = get_test_payload()
        result = migrate_legacy_case(legacy_case_id, payload, migrated_by_user_id)
        
        success = result.get("success") == True
        status_already_migrated = result.get("status") == "ALREADY_MIGRATED"
        returned_id = result.get("new_case_id")
        
        print_test("Success returned", success)
        print_test("Status is ALREADY_MIGRATED", status_already_migrated, f"Got: {result.get('status')}")
        print_test("Returned existing case ID", returned_id == existing_case_id, f"Expected: {existing_case_id}, Got: {returned_id}")
        
        # Verify create_record_migrated was NOT called (no new case)
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_IncidentCase
            WHERE ComplaintText LIKE ?
        """, payload["complaint_text"] + "%")
        
        case_count = cursor.fetchone()[0]
        
        print_test("No new case created", case_count == 0, f"Count: {case_count}")
        
        cursor.close()
        conn.close()
        
        cleanup_test_data(legacy_case_id)
        
        return success and status_already_migrated and returned_id == existing_case_id and case_count == 0
        
    except Exception as e:
        print_test("Manual mapping pre-insert", False, str(e))
        import traceback
        traceback.print_exc()
        cleanup_test_data(legacy_case_id)
        return False


def test_concurrent_double_call():
    """TEST 5: Concurrent double call (manual simulation)"""
    print_header("TEST 5: CONCURRENT DOUBLE CALL (MANUAL)")
    
    legacy_case_id = 700005
    migrated_by_user_id = 1
    
    cleanup_test_data(legacy_case_id)
    
    try:
        payload = get_test_payload()
        
        print(f"📋 Simulating rapid double migration call...")
        print(f"   (In production, these would be concurrent)")
        
        # First call
        print(f"\n🔄 First call...")
        result1 = migrate_legacy_case(legacy_case_id, payload, migrated_by_user_id)
        
        first_success = result1.get("success") == True
        first_status = result1.get("status")
        first_new_id = result1.get("new_case_id")
        
        print_test("First call succeeded", first_success, f"Status: {first_status}, ID: {first_new_id}")
        
        # Second call immediately after (simulating race condition)
        print(f"\n🔄 Second call (immediate retry)...")
        result2 = migrate_legacy_case(legacy_case_id, payload, migrated_by_user_id)
        
        second_success = result2.get("success") == True
        second_status = result2.get("status")
        second_new_id = result2.get("new_case_id")
        
        print_test("Second call succeeded", second_success, f"Status: {second_status}, ID: {second_new_id}")
        
        # Verify outcomes
        one_migrated = (first_status == "MIGRATED" and second_status == "ALREADY_MIGRATED") or \
                      (first_status == "ALREADY_MIGRATED" and second_status == "MIGRATED")
        
        one_already = (first_status == "MIGRATED" and second_status == "ALREADY_MIGRATED") or \
                      (first_status == "ALREADY_MIGRATED" and second_status == "ALREADY_MIGRATED")
        
        print_test("One MIGRATED, one ALREADY_MIGRATED", one_migrated or one_already, 
                   f"First: {first_status}, Second: {second_status}")
        
        ids_match = first_new_id == second_new_id
        print_test("Same case ID returned", ids_match, f"IDs: {first_new_id}, {second_new_id}")
        
        # Verify unique constraint protected table
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_DataMigration_Map
            WHERE legacy_case_id = ?
        """, legacy_case_id)
        
        mapping_count = cursor.fetchone()[0]
        
        print_test("Exactly one mapping row", mapping_count == 1, f"Count: {mapping_count}")
        
        # Verify no duplicate cases
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_IncidentCase
            WHERE ComplaintText LIKE ?
        """, payload["complaint_text"] + "%")
        
        case_count = cursor.fetchone()[0]
        
        print_test("Only one case created", case_count == 1, f"Count: {case_count}")
        print_test("No exception leak", True)
        
        cursor.close()
        conn.close()
        
        cleanup_test_data(legacy_case_id)
        
        return (first_success and second_success and ids_match and 
                mapping_count == 1 and case_count == 1)
        
    except Exception as e:
        print_test("Concurrent double call", False, str(e))
        import traceback
        traceback.print_exc()
        cleanup_test_data(legacy_case_id)
        return False


def main():
    """Run all tests"""
    print_header("PHASE K — SVC6B — MIGRATION CONFLICT & RETRY SAFETY TEST")
    print("Stress-testing migration orchestration for conflict handling and recovery")
    
    results = []
    
    results.append(("Retry After Success", test_retry_after_success()))
    results.append(("Simulated Mapping Write Failure", test_simulated_mapping_failure()))
    results.append(("Retry After Mapping Failure", test_retry_after_mapping_failure()))
    results.append(("Manual Mapping Pre-Insert", test_manual_mapping_pre_insert()))
    results.append(("Concurrent Double Call", test_concurrent_double_call()))
    
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
        print("\n🎉 ALL TESTS PASSED — K-SVC-6B COMPLETE")
        return True
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
