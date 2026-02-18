"""
PHASE K — SVC6A — MIGRATION ORCHESTRATOR SERVICE TEST

Comprehensive tests for migrate_legacy_case orchestration function.

Tests:
1. Normal migration
2. Duplicate migration call
3. Mapping unique constraint safety
4. Insert failure propagation
5. ML hook non-blocking
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from api.services.migration_service import migrate_legacy_case
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
    import time
    unique_id = int(time.time() * 1000) % 1000000  # Microsecond-based unique ID
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get valid FK IDs
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
        "complaint_text": f"Legacy case orchestration test {unique_id} - patient wait time complaint",
        "immediate_action": "Expedited patient care",
        "taken_action": "Reviewed staffing levels",
        "feedback_received_date": "2024-05-10",
        "patient_name": "Test Patient Orchestration",
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


def test_normal_migration():
    """TEST 1: Normal migration"""
    print_header("TEST 1: NORMAL MIGRATION")
    
    legacy_case_id = 800001
    migrated_by_user_id = 1
    
    # Clean up first
    cleanup_test_data(legacy_case_id)
    
    try:
        payload = get_test_payload()
        
        print(f"📋 Test data:")
        print(f"   Legacy Case ID: {legacy_case_id}")
        print(f"   Migrated By User: {migrated_by_user_id}")
        
        # Call orchestrator
        result = migrate_legacy_case(legacy_case_id, payload, migrated_by_user_id)
        
        # Verify result structure
        success = result.get("success") == True
        status_correct = result.get("status") == "MIGRATED"
        has_legacy_id = result.get("legacy_case_id") == legacy_case_id
        has_new_id = result.get("new_case_id") is not None
        
        print_test("Success returned", success)
        print_test("Status is MIGRATED", status_correct, f"Got: {result.get('status')}")
        print_test("Legacy case ID returned", has_legacy_id)
        print_test("New case ID returned", has_new_id, f"New ID: {result.get('new_case_id')}")
        
        if not success or not has_new_id:
            print(f"   Full result: {result}")
            cleanup_test_data(legacy_case_id)
            return False
        
        new_case_id = result.get("new_case_id")
        
        # Verify mapping row exists
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MapID, legacy_case_id, new_case_id, migrated_by_user_id
            FROM dbo.APP_DataMigration_Map
            WHERE legacy_case_id = ?
        """, legacy_case_id)
        
        mapping_row = cursor.fetchone()
        
        if not mapping_row:
            print_test("Mapping row exists", False)
            cursor.close()
            conn.close()
            cleanup_test_data(legacy_case_id)
            return False
        
        print_test("Mapping row exists", True)
        
        mapping_legacy_correct = mapping_row[1] == legacy_case_id
        mapping_new_correct = mapping_row[2] == new_case_id
        mapping_user_correct = mapping_row[3] == migrated_by_user_id
        
        print_test("Mapping legacy_case_id matches", mapping_legacy_correct)
        print_test("Mapping new_case_id matches", mapping_new_correct)
        print_test("Mapping user ID matches", mapping_user_correct)
        
        cursor.close()
        conn.close()
        
        # Clean up
        cleanup_test_data(legacy_case_id)
        
        return (success and status_correct and has_legacy_id and has_new_id and
                mapping_legacy_correct and mapping_new_correct and mapping_user_correct)
        
    except Exception as e:
        print_test("Normal migration", False, str(e))
        import traceback
        traceback.print_exc()
        cleanup_test_data(legacy_case_id)
        return False


def test_duplicate_migration_call():
    """TEST 2: Duplicate migration call"""
    print_header("TEST 2: DUPLICATE MIGRATION CALL")
    
    legacy_case_id = 800002
    migrated_by_user_id = 1
    
    # Clean up first
    cleanup_test_data(legacy_case_id)
    
    try:
        payload = get_test_payload()
        
        # First migration
        print(f"📋 First migration attempt...")
        result1 = migrate_legacy_case(legacy_case_id, payload, migrated_by_user_id)
        
        first_success = result1.get("success") == True
        first_status = result1.get("status")
        
        print_test("First migration succeeded", first_success, f"Status: {first_status}")
        
        if not first_success:
            print(f"   Error: {result1.get('error')}")
            cleanup_test_data(legacy_case_id)
            return False
        
        first_new_id = result1.get("new_case_id")
        print(f"   First new_case_id: {first_new_id}")
        
        # Second migration (duplicate)
        print(f"\n📋 Second migration attempt (should be idempotent)...")
        result2 = migrate_legacy_case(legacy_case_id, payload, migrated_by_user_id)
        
        second_success = result2.get("success") == True
        second_status = result2.get("status") == "ALREADY_MIGRATED"
        second_new_id = result2.get("new_case_id")
        
        print_test("Second migration succeeded", second_success)
        print_test("Status is ALREADY_MIGRATED", second_status, f"Got: {result2.get('status')}")
        print_test("New case ID matches first", second_new_id == first_new_id, f"Second ID: {second_new_id}")
        
        # Verify only ONE case created
        conn = get_connection()
        cursor = conn.cursor()
        
        # Count cases with this complaint text
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_IncidentCase
            WHERE ComplaintText LIKE ?
        """, payload["complaint_text"] + "%")
        
        case_count = cursor.fetchone()[0]
        
        print_test("Only one case created", case_count == 1, f"Count: {case_count}")
        
        # Verify only ONE mapping row
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_DataMigration_Map
            WHERE legacy_case_id = ?
        """, legacy_case_id)
        
        mapping_count = cursor.fetchone()[0]
        
        print_test("Only one mapping row", mapping_count == 1, f"Count: {mapping_count}")
        
        cursor.close()
        conn.close()
        
        # Clean up
        cleanup_test_data(legacy_case_id)
        
        return (first_success and second_success and second_status and
                second_new_id == first_new_id and case_count == 1 and mapping_count == 1)
        
    except Exception as e:
        print_test("Duplicate migration call", False, str(e))
        import traceback
        traceback.print_exc()
        cleanup_test_data(legacy_case_id)
        return False


def test_mapping_unique_constraint_safety():
    """TEST 3: Mapping unique constraint safety"""
    print_header("TEST 3: MAPPING UNIQUE CONSTRAINT SAFETY")
    
    legacy_case_id = 800003
    migrated_by_user_id = 1
    
    # Clean up first
    cleanup_test_data(legacy_case_id)
    
    try:
        # Get a valid new_case_id
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
        
        # Manually insert mapping row first
        print(f"📋 Manually inserting mapping row first...")
        print(f"   Legacy Case ID: {legacy_case_id}")
        print(f"   New Case ID: {existing_case_id}")
        
        cursor.execute("""
            INSERT INTO dbo.APP_DataMigration_Map
            (legacy_case_id, new_case_id, migrated_by_user_id, migrated_at)
            VALUES (?, ?, ?, GETDATE())
        """, legacy_case_id, existing_case_id, migrated_by_user_id)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Now try to migrate (should detect existing mapping)
        print(f"\n🔄 Calling migrate_legacy_case (should detect pre-existing mapping)...")
        
        payload = get_test_payload()
        result = migrate_legacy_case(legacy_case_id, payload, migrated_by_user_id)
        
        success = result.get("success") == True
        status_already_migrated = result.get("status") == "ALREADY_MIGRATED"
        returned_id = result.get("new_case_id")
        
        print_test("Success returned", success)
        print_test("Status is ALREADY_MIGRATED", status_already_migrated, f"Got: {result.get('status')}")
        print_test("Returned existing case ID", returned_id == existing_case_id, f"Got: {returned_id}")
        print_test("No exception raised", True)
        
        # Clean up
        cleanup_test_data(legacy_case_id)
        
        return success and status_already_migrated and returned_id == existing_case_id
        
    except Exception as e:
        print_test("Mapping unique constraint safety", False, str(e))
        import traceback
        traceback.print_exc()
        cleanup_test_data(legacy_case_id)
        return False


def test_insert_failure_propagation():
    """TEST 4: Insert failure propagation"""
    print_header("TEST 4: INSERT FAILURE PROPAGATION")
    
    legacy_case_id = 800004
    migrated_by_user_id = 1
    
    # Clean up first
    cleanup_test_data(legacy_case_id)
    
    try:
        # Create invalid payload (missing required field)
        print(f"📋 Creating invalid payload (missing complaint_text)...")
        
        invalid_payload = {
            "immediate_action": "Test",
            "taken_action": "Test",
            "feedback_received_date": "2024-05-10",
            "patient_name": "Test",
            "is_inpatient": True,
            "clinical_risk_type_id": 1,
            "feedback_intent_type_id": 1,
            "building_id": 1,
            "domain_id": 1,
            "category_id": 1,
            "subcategory_id": 1,
            "classification_id": 1,
            "severity_id": 1,
            "stage_id": 1,
            "harm_id": 1,
            "source_id": 1,
            "issuing_department_id": 1,
            "requires_explanation": False
            # NOTE: Missing complaint_text
        }
        
        # Call orchestrator with invalid payload
        result = migrate_legacy_case(legacy_case_id, invalid_payload, migrated_by_user_id)
        
        failed = result.get("success") == False
        has_error = "error" in result or "message" in result
        
        print_test("Migration failed as expected", failed, f"Success: {result.get('success')}")
        print_test("Error returned", has_error, f"Error: {result.get('error', result.get('message'))}")
        
        # Verify no mapping row written
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_DataMigration_Map
            WHERE legacy_case_id = ?
        """, legacy_case_id)
        
        mapping_count = cursor.fetchone()[0]
        
        print_test("No mapping row written", mapping_count == 0, f"Count: {mapping_count}")
        
        cursor.close()
        conn.close()
        
        # Clean up
        cleanup_test_data(legacy_case_id)
        
        return failed and has_error and mapping_count == 0
        
    except Exception as e:
        print_test("Insert failure propagation", False, str(e))
        import traceback
        traceback.print_exc()
        cleanup_test_data(legacy_case_id)
        return False


def test_ml_hook_non_blocking():
    """TEST 5: ML hook non-blocking"""
    print_header("TEST 5: ML HOOK NON-BLOCKING")
    
    legacy_case_id = 800005
    migrated_by_user_id = 1
    
    # Clean up first
    cleanup_test_data(legacy_case_id)
    
    try:
        payload = get_test_payload()
        
        print(f"📋 Testing migration with ML hook...")
        print(f"   (ML hook may log warnings but should not block migration)")
        
        # Call orchestrator - ML hook may fail but shouldn't break migration
        result = migrate_legacy_case(legacy_case_id, payload, migrated_by_user_id)
        
        success = result.get("success") == True
        
        print_test("Migration succeeded despite ML warnings", success, f"Status: {result.get('status')}")
        
        # Verify mapping was still created
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_DataMigration_Map
            WHERE legacy_case_id = ?
        """, legacy_case_id)
        
        mapping_exists = cursor.fetchone()[0] > 0
        
        print_test("Mapping created despite ML issues", mapping_exists)
        
        cursor.close()
        conn.close()
        
        # Clean up
        cleanup_test_data(legacy_case_id)
        
        return success and mapping_exists
        
    except Exception as e:
        print_test("ML hook non-blocking", False, str(e))
        import traceback
        traceback.print_exc()
        cleanup_test_data(legacy_case_id)
        return False


def main():
    """Run all tests"""
    print_header("PHASE K — SVC6A — MIGRATION ORCHESTRATOR SERVICE TEST")
    print("Comprehensive validation of migrate_legacy_case orchestration")
    
    results = []
    
    results.append(("Normal Migration", test_normal_migration()))
    results.append(("Duplicate Migration Call", test_duplicate_migration_call()))
    results.append(("Mapping Unique Constraint Safety", test_mapping_unique_constraint_safety()))
    results.append(("Insert Failure Propagation", test_insert_failure_propagation()))
    results.append(("ML Hook Non-Blocking", test_ml_hook_non_blocking()))
    
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
        print("\n🎉 ALL TESTS PASSED — K-SVC-6A COMPLETE")
        return True
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
