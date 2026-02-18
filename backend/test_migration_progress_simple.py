"""
SIMPLE MIGRATION PROGRESS TEST

Direct test of migration progress functions without API layer.

RUN:
    python test_migration_progress_simple.py
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from api.services.migration_progress_service import get_migration_progress
from api.db_layer.migration_progress_db import get_migration_progress_counts
from core.database import get_connection


def print_header(text):
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print('=' * 80)


def print_test(test_name, passed, message=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} — {test_name}")
    if message:
        print(f"   {message}")


def test_db_layer():
    """TEST 1: Database layer returns correct counts"""
    print_header("TEST 1: DATABASE LAYER")
    
    try:
        counts = get_migration_progress_counts()
        
        has_total = "total_cases" in counts
        has_migrated = "migrated_cases" in counts
        
        print_test("Has 'total_cases' field", has_total)
        print_test("Has 'migrated_cases' field", has_migrated)
        
        if has_total and has_migrated:
            is_total_int = isinstance(counts["total_cases"], int)
            is_migrated_int = isinstance(counts["migrated_cases"], int)
            
            print_test("total_cases is int", is_total_int)
            print_test("migrated_cases is int", is_migrated_int)
            
            print(f"\n📊 Database Counts:")
            print(f"   Total cases: {counts['total_cases']}")
            print(f"   Migrated cases: {counts['migrated_cases']}")
            
            return has_total and has_migrated and is_total_int and is_migrated_int
        
        return False
        
    except Exception as e:
        print_test("Database query", False, str(e))
        return False


def test_service_layer():
    """TEST 2: Service layer calculates progress correctly"""
    print_header("TEST 2: SERVICE LAYER")
    
    try:
        progress = get_migration_progress()
        
        has_success = "success" in progress
        has_total = "total_cases" in progress
        has_migrated = "migrated_cases" in progress
        has_remaining = "remaining_cases" in progress
        has_percent = "percent_complete" in progress
        
        print_test("Has 'success' field", has_success)
        print_test("Has 'total_cases' field", has_total)
        print_test("Has 'migrated_cases' field", has_migrated)
        print_test("Has 'remaining_cases' field", has_remaining)
        print_test("Has 'percent_complete' field", has_percent)
        
        if not all([has_success, has_total, has_migrated, has_remaining, has_percent]):
            return False
        
        print(f"\n📊 Service Progress:")
        print(f"   Success: {progress['success']}")
        print(f"   Total cases: {progress['total_cases']}")
        print(f"   Migrated cases: {progress['migrated_cases']}")
        print(f"   Remaining cases: {progress['remaining_cases']}")
        print(f"   Percent complete: {progress['percent_complete']}")
        
        # Verify calculations
        total = progress['total_cases']
        migrated = progress['migrated_cases']
        remaining = progress['remaining_cases']
        percent = progress['percent_complete']
        
        # Check remaining = total - migrated
        remaining_correct = remaining == (total - migrated)
        print_test("remaining_cases calculation correct", remaining_correct,
                  f"Expected {total - migrated}, got {remaining}")
        
        # Check percent calculation
        if total == 0:
            expected_percent = 0.0
        else:
            expected_percent = round((migrated * 100.0) / total, 1)
        
        percent_correct = percent == expected_percent
        print_test("percent_complete calculation correct", percent_correct,
                  f"Expected {expected_percent}, got {percent}")
        
        # Check percent has max 1 decimal place
        percent_str = str(percent)
        decimal_places = 0
        if "." in percent_str:
            decimal_places = len(percent_str.split(".")[1])
        
        has_one_decimal = decimal_places <= 1
        print_test("percent has ≤1 decimal place", has_one_decimal,
                  f"Decimal places: {decimal_places}")
        
        return remaining_correct and percent_correct and has_one_decimal
        
    except Exception as e:
        print_test("Service calculation", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_direct_database_query():
    """TEST 3: Verify database tables exist and are queryable"""
    print_header("TEST 3: DIRECT DATABASE QUERY")
    
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Test APP_IncidentCase table
        cursor.execute("SELECT COUNT(*) FROM dbo.APP_IncidentCase")
        total = cursor.fetchone()[0]
        print_test("APP_IncidentCase table accessible", True, f"Count: {total}")
        
        # Test APP_DataMigration_Map table
        cursor.execute("SELECT COUNT(*) FROM dbo.APP_DataMigration_Map")
        migrated = cursor.fetchone()[0]
        print_test("APP_DataMigration_Map table accessible", True, f"Count: {migrated}")
        
        # Calculate percent
        if total == 0:
            percent = 0.0
        else:
            percent = round((migrated * 100.0) / total, 1)
        
        print(f"\n📊 Direct Query Results:")
        print(f"   Total cases: {total}")
        print(f"   Migrated cases: {migrated}")
        print(f"   Percent: {percent}%")
        
        return True
        
    except Exception as e:
        print_test("Direct database query", False, str(e))
        return False
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def run_all_tests():
    """Run all migration progress tests"""
    print("\n" + "=" * 80)
    print("  MIGRATION PROGRESS SIMPLE TEST SUITE")
    print("=" * 80)
    
    results = []
    
    # Test 1: Database layer
    results.append(("Database Layer", test_db_layer()))
    
    # Test 2: Service layer
    results.append(("Service Layer", test_service_layer()))
    
    # Test 3: Direct database query
    results.append(("Direct Database Query", test_direct_database_query()))
    
    # Print summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} — {name}")
    
    print(f"\n{'=' * 80}")
    print(f"  TESTS PASSED: {passed}/{total}")
    print('=' * 80)
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
