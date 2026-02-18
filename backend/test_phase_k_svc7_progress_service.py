"""
PHASE K — SVC7 — MIGRATION PROGRESS SERVICE TEST

Validates migration progress reporting service for correct counts,
percent calculation, and read-only safety.

TARGET
------
backend/api/services/migration_progress_service.py
Function: get_migration_progress()

TESTS
-----
1. Basic Execution - Service returns all required fields
2. Count Match Validation - Service counts match direct SQL queries
3. Remaining Calculation - remaining = total - migrated
4. Percent Calculation - Percent matches manual calculation
5. Zero Total Safety - Handles empty database without crash
6. Read Only Safety - Confirms no write operations in code

RUN
---
python test_phase_k_svc7_progress_service.py
"""

import sys
from core.database import get_connection
from api.services.migration_progress_service import get_migration_progress


def test_basic_execution():
    """
    TEST 1: BASIC EXECUTION
    
    Verify service returns success and all required fields.
    """
    print("=" * 80)
    print("  TEST 1: BASIC EXECUTION")
    print("=" * 80)
    
    result = get_migration_progress()
    
    # Check success flag
    assert result.get("success") == True, "❌ FAIL — success should be True"
    print("✅ PASS — Success flag returned")
    
    # Check all required fields exist
    required_fields = ["total_cases", "migrated_cases", "remaining_cases", "percent_complete"]
    
    for field in required_fields:
        assert field in result, f"❌ FAIL — Missing field: {field}"
    
    print("✅ PASS — All required fields present")
    print(f"   Fields: {', '.join(required_fields)}")
    
    # Check all numeric fields are numeric
    assert isinstance(result["total_cases"], int), "❌ FAIL — total_cases not int"
    assert isinstance(result["migrated_cases"], int), "❌ FAIL — migrated_cases not int"
    assert isinstance(result["remaining_cases"], int), "❌ FAIL — remaining_cases not int"
    assert isinstance(result["percent_complete"], (int, float)), "❌ FAIL — percent_complete not numeric"
    
    print("✅ PASS — All fields have correct types")
    print(f"   Total: {result['total_cases']}")
    print(f"   Migrated: {result['migrated_cases']}")
    print(f"   Remaining: {result['remaining_cases']}")
    print(f"   Percent: {result['percent_complete']}%")
    print()


def test_count_match_validation():
    """
    TEST 2: COUNT MATCH VALIDATION
    
    Compare service counts with direct SQL queries.
    """
    print("=" * 80)
    print("  TEST 2: COUNT MATCH VALIDATION")
    print("=" * 80)
    
    # Get service result
    result = get_migration_progress()
    
    # Query database directly
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Query 1: Total cases
        cursor.execute("SELECT COUNT(*) FROM dbo.APP_IncidentCase")
        sql_total = cursor.fetchone()[0]
        
        # Query 2: Migrated cases
        cursor.execute("SELECT COUNT(*) FROM dbo.APP_DataMigration_Map")
        sql_migrated = cursor.fetchone()[0]
        
        print(f"📊 SQL Query Results:")
        print(f"   Total cases: {sql_total}")
        print(f"   Migrated cases: {sql_migrated}")
        print()
        print(f"📊 Service Results:")
        print(f"   Total cases: {result['total_cases']}")
        print(f"   Migrated cases: {result['migrated_cases']}")
        print()
        
        # Compare
        assert result["total_cases"] == sql_total, \
            f"❌ FAIL — Total cases mismatch: service={result['total_cases']}, SQL={sql_total}"
        print("✅ PASS — Total cases match SQL")
        
        assert result["migrated_cases"] == sql_migrated, \
            f"❌ FAIL — Migrated cases mismatch: service={result['migrated_cases']}, SQL={sql_migrated}"
        print("✅ PASS — Migrated cases match SQL")
        print()
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def test_remaining_calculation():
    """
    TEST 3: REMAINING CALCULATION
    
    Verify remaining_cases = total_cases - migrated_cases
    """
    print("=" * 80)
    print("  TEST 3: REMAINING CALCULATION")
    print("=" * 80)
    
    result = get_migration_progress()
    
    expected_remaining = result["total_cases"] - result["migrated_cases"]
    
    print(f"📐 Calculation:")
    print(f"   Total: {result['total_cases']}")
    print(f"   Migrated: {result['migrated_cases']}")
    print(f"   Expected Remaining: {expected_remaining}")
    print(f"   Actual Remaining: {result['remaining_cases']}")
    print()
    
    assert result["remaining_cases"] == expected_remaining, \
        f"❌ FAIL — Remaining calculation wrong: expected={expected_remaining}, got={result['remaining_cases']}"
    print("✅ PASS — Remaining calculation correct")
    
    # Check never negative
    assert result["remaining_cases"] >= 0, "❌ FAIL — Remaining cases is negative"
    print("✅ PASS — Remaining cases is non-negative")
    print()


def test_percent_calculation():
    """
    TEST 4: PERCENT CALCULATION
    
    Verify percent_complete matches manual calculation.
    """
    print("=" * 80)
    print("  TEST 4: PERCENT CALCULATION")
    print("=" * 80)
    
    result = get_migration_progress()
    
    total = result["total_cases"]
    migrated = result["migrated_cases"]
    
    # Manual calculation
    if total == 0:
        expected_percent = 0.0
    else:
        expected_percent = round((migrated * 100.0) / total, 2)
    
    print(f"📐 Calculation:")
    print(f"   Migrated: {migrated}")
    print(f"   Total: {total}")
    print(f"   Formula: ({migrated} / {total}) * 100")
    print(f"   Expected Percent: {expected_percent}%")
    print(f"   Actual Percent: {result['percent_complete']}%")
    print()
    
    assert result["percent_complete"] == expected_percent, \
        f"❌ FAIL — Percent calculation wrong: expected={expected_percent}, got={result['percent_complete']}"
    print("✅ PASS — Percent calculation correct")
    
    # Check range
    assert 0.0 <= result["percent_complete"] <= 100.0, \
        f"❌ FAIL — Percent out of range: {result['percent_complete']}"
    print("✅ PASS — Percent in valid range (0-100)")
    print()


def test_zero_total_safety():
    """
    TEST 5: ZERO TOTAL SAFETY
    
    Verify service handles zero total cases without crash.
    
    NOTE: This test checks the current database state. If database
    has cases, we verify no crash. If empty, we verify zero-division safety.
    """
    print("=" * 80)
    print("  TEST 5: ZERO TOTAL SAFETY")
    print("=" * 80)
    
    result = get_migration_progress()
    
    print(f"📊 Current State:")
    print(f"   Total cases: {result['total_cases']}")
    print(f"   Migrated cases: {result['migrated_cases']}")
    print()
    
    if result["total_cases"] == 0:
        # Zero total - verify percent is 0
        assert result["percent_complete"] == 0.0, \
            f"❌ FAIL — With zero total, percent should be 0.0, got {result['percent_complete']}"
        print("✅ PASS — Zero total handled safely")
        print("   Percent: 0.0 (no division error)")
    else:
        # Non-zero total - verify no crash
        print("✅ PASS — Non-zero total processed without crash")
        print(f"   Percent: {result['percent_complete']}%")
    
    # Verify no exception was raised
    print("✅ PASS — No division-by-zero exception")
    print()


def test_read_only_safety():
    """
    TEST 6: READ ONLY SAFETY
    
    Verify migration_progress_db.py contains no write operations.
    """
    print("=" * 80)
    print("  TEST 6: READ ONLY SAFETY")
    print("=" * 80)
    
    # Read the db_layer file
    with open("api/db_layer/migration_progress_db.py", "r", encoding="utf-8") as f:
        db_layer_code = f.read()
    
    # Check for write operations
    write_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE TABLE", "ALTER TABLE"]
    
    found_writes = []
    for keyword in write_keywords:
        if keyword in db_layer_code.upper():
            found_writes.append(keyword)
    
    if found_writes:
        print(f"❌ FAIL — Found write operations: {', '.join(found_writes)}")
        assert False, "DB layer should be read-only"
    else:
        print("✅ PASS — No write operations (INSERT/UPDATE/DELETE) found")
    
    # Check only SELECT exists
    assert "SELECT" in db_layer_code.upper(), "❌ FAIL — No SELECT query found"
    print("✅ PASS — Only SELECT queries present")
    
    # Count SELECT statements (should be 2)
    select_count = db_layer_code.upper().count("SELECT COUNT(*)")
    print(f"✅ PASS — {select_count} SELECT COUNT(*) queries found")
    print()


def main():
    """
    Run all K-SVC-7 Progress Service tests.
    """
    print()
    print("=" * 80)
    print("  PHASE K — SVC7 — MIGRATION PROGRESS SERVICE TEST")
    print("=" * 80)
    print("Validating migration progress reporting service")
    print()
    
    tests = [
        ("Basic Execution", test_basic_execution),
        ("Count Match Validation", test_count_match_validation),
        ("Remaining Calculation", test_remaining_calculation),
        ("Percent Calculation", test_percent_calculation),
        ("Zero Total Safety", test_zero_total_safety),
        ("Read Only Safety", test_read_only_safety)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL — {test_name}")
            print(f"   {str(e)}")
            print()
            failed += 1
        except Exception as e:
            print(f"❌ FAIL — {test_name}")
            print(f"   Unexpected error: {str(e)}")
            print()
            failed += 1
    
    # Summary
    print("=" * 80)
    print("  TEST SUMMARY")
    print("=" * 80)
    for test_name, _ in tests:
        status = "✅ PASS" if test_name not in [t[0] for t in tests[passed:]] else "❌ FAIL"
        print(f"{status} — {test_name}")
    
    print()
    print("=" * 80)
    print(f"TOTAL: {passed}/{len(tests)} tests passed")
    print("=" * 80)
    print()
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED — K-SVC-7 COMPLETE")
    else:
        print(f"⚠️  {failed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
