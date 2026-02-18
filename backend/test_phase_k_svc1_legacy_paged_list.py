"""
PHASE K — SVC1 — LEGACY PAGED LIST TEST

Comprehensive test suite for list_legacy_cases_paged function.

Tests:
1. Basic call and return structure
2. Data structure validation
3. Pagination correctness
4. Mapping exclusion (migrated cases filtered out)
5. Order by date DESC
6. Preview length validation
7. Read-only safety
"""

import sys
from pathlib import Path
from typing import Set

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from api.db_layer.legacy_migration_db import list_legacy_cases_paged
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


def test_basic_call():
    """TEST 1: Basic call and return structure"""
    print_header("TEST 1: BASIC CALL")
    
    try:
        rows, total = list_legacy_cases_paged(page=1, page_size=50)
        
        # Check return types
        is_list = isinstance(rows, list)
        print_test("Returns list", is_list)
        
        is_int = isinstance(total, int)
        print_test("Returns int total", is_int)
        
        # Check page size constraint
        within_limit = len(rows) <= 50
        print_test("Respects page_size limit", within_limit, f"Got {len(rows)} rows")
        
        print(f"\nResult: {len(rows)} rows returned, {total} total unmigrated cases")
        
        return is_list and is_int and within_limit
        
    except Exception as e:
        print_test("Basic call", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_structure_check():
    """TEST 2: Data structure validation"""
    print_header("TEST 2: STRUCTURE CHECK")
    
    try:
        rows, total = list_legacy_cases_paged(page=1, page_size=10)
        
        if not rows:
            print_test("Has data", False, "No rows returned (empty legacy table?)")
            return False
        
        print_test("Has data", True, f"{len(rows)} rows available")
        
        first_row = rows[0]
        
        # Check required fields
        required_fields = [
            'legacy_case_id',
            'incident_request_id',
            'patient_name',
            'received_date',
            'preview_description',
            'source_section_id',
            'source_department_id',
            'source_admin_id'
        ]
        
        all_present = True
        for field in required_fields:
            present = field in first_row
            print_test(f"Field '{field}' exists", present)
            if not present:
                all_present = False
        
        # Display sample row
        if all_present:
            print("\n📋 Sample Row:")
            print(f"  Legacy Case ID: {first_row['legacy_case_id']}")
            print(f"  Request ID: {first_row['incident_request_id']}")
            print(f"  Patient: {first_row['patient_name']}")
            print(f"  Received: {first_row['received_date']}")
            print(f"  Preview: {first_row['preview_description'][:50]}...")
        
        return all_present
        
    except Exception as e:
        print_test("Structure check", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_pagination():
    """TEST 3: Pagination correctness"""
    print_header("TEST 3: PAGINATION CHECK")
    
    try:
        rows_page1, total = list_legacy_cases_paged(page=1, page_size=10)
        rows_page2, total2 = list_legacy_cases_paged(page=2, page_size=10)
        
        print_test("Total count consistent", total == total2, f"Page 1: {total}, Page 2: {total2}")
        
        # Check for duplicate IDs across pages
        ids_page1: Set[int] = {row['legacy_case_id'] for row in rows_page1}
        ids_page2: Set[int] = {row['legacy_case_id'] for row in rows_page2}
        
        overlap = ids_page1.intersection(ids_page2)
        
        no_duplicates = len(overlap) == 0
        print_test("No duplicate IDs between pages", no_duplicates, 
                   f"Overlap count: {len(overlap)}" if overlap else "Clean pagination")
        
        # Display page results
        print(f"\n📄 Pagination Results:")
        print(f"  Page 1: {len(rows_page1)} rows")
        print(f"  Page 2: {len(rows_page2)} rows")
        print(f"  Total available: {total}")
        
        return (total == total2) and no_duplicates
        
    except Exception as e:
        print_test("Pagination check", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_mapping_exclusion():
    """TEST 4: Mapping exclusion - migrated cases filtered out"""
    print_header("TEST 4: MAPPING EXCLUSION CHECK")
    
    conn = None
    cursor = None
    test_legacy_id = None
    
    try:
        # Step 1: Get a legacy case ID that exists
        rows_before, total_before = list_legacy_cases_paged(page=1, page_size=1)
        
        if not rows_before:
            print_test("Test data available", False, "No unmigrated cases found")
            return False
        
        test_legacy_id = rows_before[0]['legacy_case_id']
        print(f"📌 Test legacy case ID: {test_legacy_id}")
        
        # Step 2: Get valid IDs for mapping insert
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT TOP 1 IncidentRequestCaseID FROM APP_IncidentCase")
        case_row = cursor.fetchone()
        
        cursor.execute("SELECT TOP 1 UserID FROM APP_Users")
        user_row = cursor.fetchone()
        
        if not case_row or not user_row:
            print_test("Required test data", False, "Missing case or user records")
            return False
        
        new_case_id = case_row[0]
        user_id = user_row[0]
        
        # Step 3: Insert mapping record
        cursor.execute("""
            INSERT INTO APP_DataMigration_Map 
            (legacy_case_id, new_case_id, migrated_by_user_id, migrated_at)
            VALUES (?, ?, ?, GETDATE())
        """, test_legacy_id, new_case_id, user_id)
        conn.commit()
        
        print_test("Mapping record inserted", True, f"Mapped legacy {test_legacy_id} → new {new_case_id}")
        
        # Step 4: Query again - should not see this legacy case
        rows_after, total_after = list_legacy_cases_paged(page=1, page_size=100)
        
        legacy_ids_after = {row['legacy_case_id'] for row in rows_after}
        excluded = test_legacy_id not in legacy_ids_after
        
        print_test("Mapped case excluded from results", excluded)
        
        # Verify total count decreased
        count_decreased = total_after == (total_before - 1)
        print_test("Total count decreased by 1", count_decreased, 
                   f"Before: {total_before}, After: {total_after}")
        
        return excluded and count_decreased
        
    except Exception as e:
        print_test("Mapping exclusion", False, str(e))
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test data
        if conn and cursor and test_legacy_id:
            try:
                cursor.execute("DELETE FROM APP_DataMigration_Map WHERE legacy_case_id = ?", test_legacy_id)
                conn.commit()
                print("🧹 Cleanup: Test mapping record removed")
            except:
                pass
        
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def test_order_check():
    """TEST 5: Order by received_date DESC"""
    print_header("TEST 5: ORDER CHECK")
    
    try:
        rows, total = list_legacy_cases_paged(page=1, page_size=20)
        
        if len(rows) < 2:
            print_test("Sufficient data for order test", False, "Need at least 2 rows")
            return False
        
        print_test("Sufficient data", True, f"{len(rows)} rows")
        
        # Check dates are in descending order
        dates = [row['received_date'] for row in rows if row['received_date']]
        
        is_descending = all(dates[i] >= dates[i+1] for i in range(len(dates) - 1))
        
        print_test("Dates in DESC order", is_descending)
        
        if len(dates) >= 3:
            print("\n📅 Sample dates (should be newest → oldest):")
            for i, date in enumerate(dates[:5], 1):
                print(f"  {i}. {date}")
        
        return is_descending
        
    except Exception as e:
        print_test("Order check", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_preview_length():
    """TEST 6: Preview description length validation"""
    print_header("TEST 6: PREVIEW LENGTH CHECK")
    
    try:
        rows, total = list_legacy_cases_paged(page=1, page_size=50)
        
        if not rows:
            print_test("Has data", False)
            return False
        
        # Check all preview_description fields are <= 200 chars
        all_valid = True
        max_length = 0
        violations = []
        
        for row in rows:
            preview = row['preview_description']
            if preview:
                length = len(preview)
                max_length = max(max_length, length)
                if length > 200:
                    all_valid = False
                    violations.append((row['legacy_case_id'], length))
        
        print_test("All previews <= 200 chars", all_valid, 
                   f"Max length found: {max_length}")
        
        if violations:
            print(f"\n❌ Violations found: {len(violations)}")
            for legacy_id, length in violations[:3]:
                print(f"  Case {legacy_id}: {length} chars")
        
        return all_valid
        
    except Exception as e:
        print_test("Preview length check", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_read_only_safety():
    """TEST 7: Read-only safety check"""
    print_header("TEST 7: READ-ONLY SAFETY CHECK")
    
    try:
        # Before: get counts
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM IncidentRequestCase")
        case_count_before = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM IncidentRequest")
        request_count_before = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM APP_DataMigration_Map")
        map_count_before = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        print(f"📊 Before function call:")
        print(f"  IncidentRequestCase: {case_count_before}")
        print(f"  IncidentRequest: {request_count_before}")
        print(f"  APP_DataMigration_Map: {map_count_before}")
        
        # Call function
        rows, total = list_legacy_cases_paged(page=1, page_size=50)
        
        # After: verify counts unchanged
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM IncidentRequestCase")
        case_count_after = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM IncidentRequest")
        request_count_after = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM APP_DataMigration_Map")
        map_count_after = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        print(f"\n📊 After function call:")
        print(f"  IncidentRequestCase: {case_count_after}")
        print(f"  IncidentRequest: {request_count_after}")
        print(f"  APP_DataMigration_Map: {map_count_after}")
        
        no_case_changes = case_count_before == case_count_after
        no_request_changes = request_count_before == request_count_after
        no_map_changes = map_count_before == map_count_after
        
        print_test("IncidentRequestCase unchanged", no_case_changes)
        print_test("IncidentRequest unchanged", no_request_changes)
        print_test("APP_DataMigration_Map unchanged", no_map_changes)
        
        return no_case_changes and no_request_changes and no_map_changes
        
    except Exception as e:
        print_test("Read-only safety", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print_header("PHASE K — SVC1 — LEGACY PAGED LIST TEST")
    print("Comprehensive validation of list_legacy_cases_paged function")
    
    results = []
    
    results.append(("Basic Call", test_basic_call()))
    results.append(("Structure Check", test_structure_check()))
    results.append(("Pagination", test_pagination()))
    results.append(("Mapping Exclusion", test_mapping_exclusion()))
    results.append(("Order Check", test_order_check()))
    results.append(("Preview Length", test_preview_length()))
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
        print("\n🎉 ALL TESTS PASSED — K-SVC-1 COMPLETE")
        return True
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
