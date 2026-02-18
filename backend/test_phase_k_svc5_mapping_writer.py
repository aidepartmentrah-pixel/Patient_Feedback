"""
PHASE K — SVC5 — MAPPING WRITER DB LAYER TEST

Comprehensive tests for insert_migration_mapping function.

Tests:
1. Insert success
2. Duplicate prevention
3. FK safety - new_case_id
4. FK safety - user
5. Rollback check
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from api.db_layer.migration_map_db import insert_migration_mapping
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


def cleanup_test_mapping(legacy_case_id):
    """Clean up test mapping record"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM dbo.APP_DataMigration_Map WHERE legacy_case_id = ?", legacy_case_id)
        conn.commit()
        cursor.close()
        conn.close()
    except:
        pass


def get_valid_test_ids():
    """Get valid FK IDs for testing"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get a valid case ID
    cursor.execute("SELECT TOP 1 IncidentRequestCaseID FROM dbo.APP_IncidentCase ORDER BY IncidentRequestCaseID DESC")
    row = cursor.fetchone()
    case_id = row[0] if row else None
    
    # Get a valid user ID
    cursor.execute("SELECT TOP 1 UserID FROM dbo.APP_Users ORDER BY UserID")
    row = cursor.fetchone()
    user_id = row[0] if row else None
    
    cursor.close()
    conn.close()
    
    return case_id, user_id


def test_insert_success():
    """TEST 1: Insert success"""
    print_header("TEST 1: INSERT SUCCESS")
    
    legacy_case_id = 900001
    
    # Clean up first
    cleanup_test_mapping(legacy_case_id)
    
    try:
        # Get valid FKs
        case_id, user_id = get_valid_test_ids()
        
        if not case_id or not user_id:
            print_test("Test data available", False, "Missing valid case or user IDs")
            return False
        
        print(f"📋 Test data:")
        print(f"   Legacy Case ID: {legacy_case_id}")
        print(f"   New Case ID: {case_id}")
        print(f"   Migrated By User: {user_id}")
        
        # Call function
        result = insert_migration_mapping(legacy_case_id, case_id, user_id)
        
        # Verify result
        success = result.get("success") == True
        legacy_matches = result.get("legacy_case_id") == legacy_case_id
        new_matches = result.get("new_case_id") == case_id
        
        print_test("Function returned success", success)
        print_test("Legacy case ID matches", legacy_matches, f"Got: {result.get('legacy_case_id')}")
        print_test("New case ID matches", new_matches, f"Got: {result.get('new_case_id')}")
        
        # Verify database row
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MapID, legacy_case_id, new_case_id, migrated_by_user_id, migrated_at
            FROM dbo.APP_DataMigration_Map
            WHERE legacy_case_id = ?
        """, legacy_case_id)
        
        row = cursor.fetchone()
        
        if not row:
            print_test("Row exists in database", False)
            cursor.close()
            conn.close()
            cleanup_test_mapping(legacy_case_id)
            return False
        
        print_test("Row exists in database", True)
        
        db_legacy_matches = row[1] == legacy_case_id
        db_new_matches = row[2] == case_id
        db_user_matches = row[3] == user_id
        has_timestamp = row[4] is not None
        
        print_test("DB legacy_case_id matches", db_legacy_matches, f"DB value: {row[1]}")
        print_test("DB new_case_id matches", db_new_matches, f"DB value: {row[2]}")
        print_test("DB migrated_by_user_id matches", db_user_matches, f"DB value: {row[3]}")
        print_test("DB has timestamp", has_timestamp)
        
        cursor.close()
        conn.close()
        
        # Clean up
        cleanup_test_mapping(legacy_case_id)
        
        return (success and legacy_matches and new_matches and 
                db_legacy_matches and db_new_matches and db_user_matches and has_timestamp)
        
    except Exception as e:
        print_test("Insert success", False, str(e))
        import traceback
        traceback.print_exc()
        cleanup_test_mapping(legacy_case_id)
        return False


def test_duplicate_prevention():
    """TEST 2: Duplicate prevention"""
    print_header("TEST 2: DUPLICATE PREVENTION")
    
    legacy_case_id = 900002
    
    # Clean up first
    cleanup_test_mapping(legacy_case_id)
    
    try:
        # Get valid FKs
        case_id, user_id = get_valid_test_ids()
        
        if not case_id or not user_id:
            print_test("Test data available", False, "Missing valid case or user IDs")
            return False
        
        # First insert - should succeed
        result1 = insert_migration_mapping(legacy_case_id, case_id, user_id)
        
        first_success = result1.get("success") == True
        print_test("First insert succeeded", first_success)
        
        if not first_success:
            cleanup_test_mapping(legacy_case_id)
            return False
        
        # Second insert - should raise ValueError
        second_failed = False
        error_message = ""
        
        try:
            result2 = insert_migration_mapping(legacy_case_id, case_id, user_id)
            print_test("Second insert raised exception", False, "No exception raised")
        except ValueError as ve:
            second_failed = True
            error_message = str(ve)
            print_test("Second insert raised ValueError", True, f"Message: {error_message}")
        except Exception as e:
            print_test("Second insert raised wrong exception type", False, f"Got {type(e).__name__}: {str(e)}")
        
        # Check error message
        message_correct = "already migrated" in error_message.lower()
        print_test("Error message correct", message_correct, f"Message: {error_message}")
        
        # Verify only one row exists
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_DataMigration_Map
            WHERE legacy_case_id = ?
        """, legacy_case_id)
        
        row_count = cursor.fetchone()[0]
        
        print_test("Only one row exists", row_count == 1, f"Count: {row_count}")
        
        cursor.close()
        conn.close()
        
        # Clean up
        cleanup_test_mapping(legacy_case_id)
        
        return first_success and second_failed and message_correct and row_count == 1
        
    except Exception as e:
        print_test("Duplicate prevention", False, str(e))
        import traceback
        traceback.print_exc()
        cleanup_test_mapping(legacy_case_id)
        return False


def test_fk_safety_case_id():
    """TEST 3: FK safety - new_case_id"""
    print_header("TEST 3: FK SAFETY — NEW_CASE_ID")
    
    legacy_case_id = 900003
    
    # Clean up first
    cleanup_test_mapping(legacy_case_id)
    
    try:
        # Get valid user ID
        _, user_id = get_valid_test_ids()
        
        if not user_id:
            print_test("Test data available", False, "Missing valid user ID")
            return False
        
        # Use invalid case ID
        invalid_case_id = 999999999
        
        print(f"📋 Test data:")
        print(f"   Invalid Case ID: {invalid_case_id}")
        print(f"   User ID: {user_id}")
        
        # Should raise generic Exception
        exception_raised = False
        exception_type = None
        
        try:
            result = insert_migration_mapping(legacy_case_id, invalid_case_id, user_id)
            print_test("Exception raised", False, "No exception raised")
        except ValueError as ve:
            exception_raised = True
            exception_type = "ValueError"
            print_test("Wrong exception type", False, f"Got ValueError instead of generic Exception")
        except Exception as e:
            exception_raised = True
            exception_type = "Exception"
            error_message = str(e)
            print_test("Generic Exception raised", True, f"Message: {error_message}")
            
            # Check error mentions "Failed to insert"
            message_correct = "failed to insert" in error_message.lower()
            print_test("Error message correct", message_correct)
        
        # Verify no row inserted
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_DataMigration_Map
            WHERE legacy_case_id = ?
        """, legacy_case_id)
        
        row_count = cursor.fetchone()[0]
        
        print_test("No row inserted", row_count == 0, f"Count: {row_count}")
        
        cursor.close()
        conn.close()
        
        # Clean up
        cleanup_test_mapping(legacy_case_id)
        
        return exception_raised and exception_type == "Exception" and row_count == 0
        
    except Exception as e:
        print_test("FK safety case ID", False, str(e))
        import traceback
        traceback.print_exc()
        cleanup_test_mapping(legacy_case_id)
        return False


def test_fk_safety_user_id():
    """TEST 4: FK safety - user"""
    print_header("TEST 4: FK SAFETY — USER")
    
    legacy_case_id = 900004
    
    # Clean up first
    cleanup_test_mapping(legacy_case_id)
    
    try:
        # Get valid case ID
        case_id, _ = get_valid_test_ids()
        
        if not case_id:
            print_test("Test data available", False, "Missing valid case ID")
            return False
        
        # Use invalid user ID
        invalid_user_id = 999999999
        
        print(f"📋 Test data:")
        print(f"   Case ID: {case_id}")
        print(f"   Invalid User ID: {invalid_user_id}")
        
        # Should raise generic Exception
        exception_raised = False
        
        try:
            result = insert_migration_mapping(legacy_case_id, case_id, invalid_user_id)
            print_test("Exception raised", False, "No exception raised")
        except ValueError as ve:
            exception_raised = True
            print_test("Wrong exception type", False, f"Got ValueError instead of generic Exception")
        except Exception as e:
            exception_raised = True
            error_message = str(e)
            print_test("Generic Exception raised", True, f"Message: {error_message}")
        
        # Verify no row inserted
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_DataMigration_Map
            WHERE legacy_case_id = ?
        """, legacy_case_id)
        
        row_count = cursor.fetchone()[0]
        
        print_test("No row inserted", row_count == 0, f"Count: {row_count}")
        
        cursor.close()
        conn.close()
        
        # Clean up
        cleanup_test_mapping(legacy_case_id)
        
        return exception_raised and row_count == 0
        
    except Exception as e:
        print_test("FK safety user", False, str(e))
        import traceback
        traceback.print_exc()
        cleanup_test_mapping(legacy_case_id)
        return False


def test_rollback_check():
    """TEST 5: Rollback check"""
    print_header("TEST 5: ROLLBACK CHECK")
    
    legacy_case_id = 900005
    
    # Clean up first
    cleanup_test_mapping(legacy_case_id)
    
    try:
        # Get valid user ID
        _, user_id = get_valid_test_ids()
        
        if not user_id:
            print_test("Test data available", False, "Missing valid user ID")
            return False
        
        # Use invalid case ID to force failure AFTER duplicate check
        invalid_case_id = 999999999
        
        print(f"📋 Test scenario:")
        print(f"   Legacy Case ID: {legacy_case_id}")
        print(f"   Invalid Case ID: {invalid_case_id} (forces FK failure)")
        print(f"   User ID: {user_id}")
        
        # Count rows before
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM dbo.APP_DataMigration_Map")
        count_before = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        print(f"\n📊 Before: {count_before} total mappings")
        
        # Attempt insert (should fail and rollback)
        try:
            result = insert_migration_mapping(legacy_case_id, invalid_case_id, user_id)
            print_test("Insert failed as expected", False, "No exception raised")
        except Exception as e:
            print_test("Insert failed as expected", True, f"Exception: {type(e).__name__}")
        
        # Count rows after
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM dbo.APP_DataMigration_Map")
        count_after = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        print(f"📊 After: {count_after} total mappings")
        
        # Verify no partial insert
        no_partial_insert = count_before == count_after
        print_test("No partial insert occurred", no_partial_insert, f"Before: {count_before}, After: {count_after}")
        
        # Verify specific row doesn't exist
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_DataMigration_Map
            WHERE legacy_case_id = ?
        """, legacy_case_id)
        
        specific_count = cursor.fetchone()[0]
        
        print_test("Test row doesn't exist", specific_count == 0, f"Count: {specific_count}")
        
        cursor.close()
        conn.close()
        
        # Clean up
        cleanup_test_mapping(legacy_case_id)
        
        return no_partial_insert and specific_count == 0
        
    except Exception as e:
        print_test("Rollback check", False, str(e))
        import traceback
        traceback.print_exc()
        cleanup_test_mapping(legacy_case_id)
        return False


def main():
    """Run all tests"""
    print_header("PHASE K — SVC5 — MAPPING WRITER DB LAYER TEST")
    print("Comprehensive validation of insert_migration_mapping function")
    
    results = []
    
    results.append(("Insert Success", test_insert_success()))
    results.append(("Duplicate Prevention", test_duplicate_prevention()))
    results.append(("FK Safety - Case ID", test_fk_safety_case_id()))
    results.append(("FK Safety - User ID", test_fk_safety_user_id()))
    results.append(("Rollback Check", test_rollback_check()))
    
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
        print("\n🎉 ALL TESTS PASSED — K-SVC-5 COMPLETE")
        return True
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
