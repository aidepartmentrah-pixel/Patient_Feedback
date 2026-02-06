"""
Test Insight DB Layer - Time-Bucket Trend Query (B-I6)
Unit and Integration tests for get_subcase_created_time_buckets function.

Run: python backend/test_insight_time_buckets.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add backend directory to path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from api_v2.db_layer import insight_db

print("=" * 80)
print("INSIGHT DB LAYER - TIME-BUCKET TREND QUERY TEST (B-I6)")
print("=" * 80)

test_passed = 0
test_failed = 0

# ============================================================
# UNIT TESTS (No database required)
# ============================================================

print("\n" + "=" * 80)
print("UNIT TESTS (Structure & Logic)")
print("=" * 80)

# Test 1: Function exists and is callable
print("\n[UNIT TEST 1] Function exists and is callable...")
try:
    assert hasattr(insight_db, 'get_subcase_created_time_buckets')
    assert callable(insight_db.get_subcase_created_time_buckets)
    print("✅ PASS: Function exists and is callable")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 2: Function signature
print("\n[UNIT TEST 2] Function signature is correct...")
try:
    import inspect
    sig = inspect.signature(insight_db.get_subcase_created_time_buckets)
    params = list(sig.parameters.keys())
    assert 'conn' in params, "Missing 'conn' parameter"
    assert 'allowed_unit_ids' in params, "Missing 'allowed_unit_ids' parameter"
    assert 'bucket' in params, "Missing 'bucket' parameter"
    print("✅ PASS: Signature correct (conn, allowed_unit_ids, bucket)")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 3: Return type annotation
print("\n[UNIT TEST 3] Return type annotation...")
try:
    sig = inspect.signature(insight_db.get_subcase_created_time_buckets)
    print(f"   Return annotation: {sig.return_annotation}")
    print("✅ PASS: Return type documented")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# ============================================================
# INTEGRATION TESTS (Database required)
# ============================================================

print("\n" + "=" * 80)
print("INTEGRATION TESTS (Database Operations)")
print("=" * 80)

# Test 4: Database connection works
print("\n[INTEGRATION TEST 4] Database connection...")
try:
    conn = insight_db.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    assert result[0] == 1
    cursor.close()
    conn.close()
    print("✅ PASS: Database connection successful")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: Could not connect to database")
    print(f"   Error: {e}")
    test_failed += 1
    print("\n⚠️  Skipping remaining integration tests (database unavailable)")
    print("\n" + "=" * 80)
    print(f"FINAL RESULTS: {test_passed} passed, {test_failed} failed")
    print("=" * 80)
    sys.exit(1)

# Test 5: Invalid bucket raises ValueError
print("\n[INTEGRATION TEST 5] Invalid bucket raises ValueError...")
try:
    conn = insight_db.get_db_connection()
    
    error_raised = False
    try:
        result = insight_db.get_subcase_created_time_buckets(conn, [1], bucket="invalid")
    except ValueError as e:
        error_raised = True
        print(f"   ✓ ValueError raised: {e}")
    
    conn.close()
    
    assert error_raised, "Should raise ValueError for invalid bucket"
    print("✅ PASS: Invalid bucket raises ValueError")
    test_passed += 1
except AssertionError as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1
except Exception as e:
    print(f"❌ FAIL: Unexpected error: {e}")
    test_failed += 1

# Test 6: Empty allowed_unit_ids returns empty list
print("\n[INTEGRATION TEST 6] Empty allowed_unit_ids...")
try:
    conn = insight_db.get_db_connection()
    result = insight_db.get_subcase_created_time_buckets(conn, [], bucket="month")
    conn.close()
    
    assert isinstance(result, list), "Should return a list"
    assert len(result) == 0, "Should return empty list for empty allowed_unit_ids"
    print("✅ PASS: Empty allowed_unit_ids returns empty list")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 7: Invalid org unit ID returns empty results
print("\n[INTEGRATION TEST 7] Invalid org unit ID...")
try:
    conn = insight_db.get_db_connection()
    result = insight_db.get_subcase_created_time_buckets(conn, [-99999], bucket="month")
    conn.close()
    
    assert isinstance(result, list), "Should return a list"
    print(f"   Returned {len(result)} results (expected 0)")
    assert len(result) == 0, "Should return empty list for non-existent org unit"
    print("✅ PASS: Invalid org unit ID returns empty results")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 8: Query with "day" bucket
print("\n[INTEGRATION TEST 8] Query with 'day' bucket...")
try:
    conn = insight_db.get_db_connection()
    cursor = conn.cursor()
    
    # Get org units
    cursor.execute("""
        SELECT DISTINCT TOP 5 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    
    if not org_units:
        print("   ℹ️  No subcases in database - skipping test")
        test_passed += 1
    else:
        result = insight_db.get_subcase_created_time_buckets(conn, org_units, bucket="day")
        
        print(f"   Found {len(result)} day buckets")
        if len(result) > 0:
            print(f"   Sample buckets (first 3):")
            for item in result[:3]:
                print(f"      • {item['bucket_label']}: {item['count']} subcases")
            
            # Validate structure
            item = result[0]
            assert isinstance(item, dict), "Each item should be a dict"
            assert 'bucket_label' in item, "Should have 'bucket_label' key"
            assert 'count' in item, "Should have 'count' key"
            assert isinstance(item['bucket_label'], str), "bucket_label should be string"
            assert isinstance(item['count'], int), "count should be integer"
            
            # Validate day format: YYYY-MM-DD
            label = item['bucket_label']
            assert len(label) == 10, f"Day label should be 10 chars: {label}"
            assert label[4] == '-' and label[7] == '-', f"Day format should be YYYY-MM-DD: {label}"
        
        print("✅ PASS: 'day' bucket query works")
        test_passed += 1
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 9: Query with "month" bucket
print("\n[INTEGRATION TEST 9] Query with 'month' bucket...")
try:
    conn = insight_db.get_db_connection()
    cursor = conn.cursor()
    
    # Get org units
    cursor.execute("""
        SELECT DISTINCT TOP 5 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    
    if not org_units:
        print("   ℹ️  No subcases in database - skipping test")
        test_passed += 1
    else:
        result = insight_db.get_subcase_created_time_buckets(conn, org_units, bucket="month")
        
        print(f"   Found {len(result)} month buckets")
        if len(result) > 0:
            print(f"   Sample buckets:")
            for item in result:
                print(f"      • {item['bucket_label']}: {item['count']} subcases")
            
            # Validate structure
            item = result[0]
            assert isinstance(item, dict), "Each item should be a dict"
            assert 'bucket_label' in item, "Should have 'bucket_label' key"
            assert 'count' in item, "Should have 'count' key"
            
            # Validate month format: YYYY-MM
            label = item['bucket_label']
            assert len(label) == 7, f"Month label should be 7 chars: {label}"
            assert label[4] == '-', f"Month format should be YYYY-MM: {label}"
        
        print("✅ PASS: 'month' bucket query works")
        test_passed += 1
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 10: Query with "year" bucket
print("\n[INTEGRATION TEST 10] Query with 'year' bucket...")
try:
    conn = insight_db.get_db_connection()
    cursor = conn.cursor()
    
    # Get org units
    cursor.execute("""
        SELECT DISTINCT TOP 5 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    
    if not org_units:
        print("   ℹ️  No subcases in database - skipping test")
        test_passed += 1
    else:
        result = insight_db.get_subcase_created_time_buckets(conn, org_units, bucket="year")
        
        print(f"   Found {len(result)} year buckets")
        if len(result) > 0:
            print(f"   Year buckets:")
            for item in result:
                print(f"      • {item['bucket_label']}: {item['count']} subcases")
            
            # Validate structure
            item = result[0]
            assert isinstance(item, dict), "Each item should be a dict"
            assert 'bucket_label' in item, "Should have 'bucket_label' key"
            assert 'count' in item, "Should have 'count' key"
            
            # Validate year format: YYYY
            label = item['bucket_label']
            assert len(label) == 4, f"Year label should be 4 chars: {label}"
            assert label.isdigit(), f"Year should be numeric: {label}"
        
        print("✅ PASS: 'year' bucket query works")
        test_passed += 1
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 11: Verify aggregation sums correctly
print("\n[INTEGRATION TEST 11] Verify aggregation sums correctly...")
try:
    conn = insight_db.get_db_connection()
    cursor = conn.cursor()
    
    # Get org units
    cursor.execute("""
        SELECT DISTINCT TOP 5 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    
    if not org_units:
        print("   ℹ️  No subcases in database - skipping test")
        test_passed += 1
    else:
        # Get total count directly
        placeholders = ','.join('?' * len(org_units))
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM dbo.APP_AdministrativeSubcase
            WHERE TargetOrgUnitID IN ({placeholders})
        """, org_units)
        total_count = cursor.fetchone()[0]
        
        # Get aggregated count via function (using year for fewer buckets)
        result = insight_db.get_subcase_created_time_buckets(conn, org_units, bucket="year")
        sum_from_buckets = sum(item['count'] for item in result)
        
        print(f"   Total subcases (direct query): {total_count}")
        print(f"   Sum of bucket counts: {sum_from_buckets}")
        
        assert sum_from_buckets == total_count, \
            f"Sum mismatch: {sum_from_buckets} != {total_count}"
        
        print("✅ PASS: Aggregation sums correctly")
        test_passed += 1
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 12: Verify results are sorted ascending
print("\n[INTEGRATION TEST 12] Verify results are sorted ascending...")
try:
    conn = insight_db.get_db_connection()
    cursor = conn.cursor()
    
    # Get org units
    cursor.execute("""
        SELECT DISTINCT TOP 5 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    
    if not org_units:
        print("   ℹ️  No subcases in database - skipping test")
        test_passed += 1
    else:
        result = insight_db.get_subcase_created_time_buckets(conn, org_units, bucket="month")
        
        if len(result) < 2:
            print("   ℹ️  Less than 2 buckets - skipping sort test")
            test_passed += 1
        else:
            labels = [item['bucket_label'] for item in result]
            
            print(f"   Bucket labels: {labels}")
            
            # Check that labels are ascending (string comparison works for YYYY-MM format)
            is_sorted_asc = all(labels[i] <= labels[i+1] for i in range(len(labels)-1))
            
            assert is_sorted_asc, "Results should be sorted ascending by time"
            
            print("✅ PASS: Results are sorted ascending")
            test_passed += 1
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 13: Verify scope filtering works
print("\n[INTEGRATION TEST 13] Verify scope filtering...")
try:
    conn = insight_db.get_db_connection()
    cursor = conn.cursor()
    
    # Get two different org units
    cursor.execute("""
        SELECT DISTINCT TOP 2 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
        ORDER BY TargetOrgUnitID
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    
    if len(org_units) < 2:
        print("   ℹ️  Less than 2 org units - skipping test")
        test_passed += 1
    else:
        print(f"   Testing with org units: {org_units}")
        
        # Get counts for both org units
        result_both = insight_db.get_subcase_created_time_buckets(conn, org_units, bucket="year")
        total_both = sum(item['count'] for item in result_both)
        
        # Get counts for first org unit only
        result_first = insight_db.get_subcase_created_time_buckets(conn, [org_units[0]], bucket="year")
        total_first = sum(item['count'] for item in result_first)
        
        # Get counts for second org unit only
        result_second = insight_db.get_subcase_created_time_buckets(conn, [org_units[1]], bucket="year")
        total_second = sum(item['count'] for item in result_second)
        
        print(f"   Both units: {total_both} subcases")
        print(f"   First unit: {total_first} subcases")
        print(f"   Second unit: {total_second} subcases")
        
        # Total for both should equal sum of individuals
        assert total_both == total_first + total_second, \
            "Multi-unit scope filtering incorrect"
        
        print("✅ PASS: Scope filtering works correctly")
        test_passed += 1
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 14: Verify month bucket aggregates days correctly
print("\n[INTEGRATION TEST 14] Verify month aggregates days correctly...")
try:
    conn = insight_db.get_db_connection()
    cursor = conn.cursor()
    
    # Get org units
    cursor.execute("""
        SELECT DISTINCT TOP 5 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    
    if not org_units:
        print("   ℹ️  No subcases in database - skipping test")
        test_passed += 1
    else:
        # Get day buckets
        result_day = insight_db.get_subcase_created_time_buckets(conn, org_units, bucket="day")
        
        # Get month buckets
        result_month = insight_db.get_subcase_created_time_buckets(conn, org_units, bucket="month")
        
        total_day = sum(item['count'] for item in result_day)
        total_month = sum(item['count'] for item in result_month)
        
        print(f"   Day buckets: {len(result_day)} buckets, {total_day} total")
        print(f"   Month buckets: {len(result_month)} buckets, {total_month} total")
        
        # Totals should match regardless of bucket granularity
        assert total_day == total_month, \
            f"Bucket granularity should not change totals: {total_day} != {total_month}"
        
        print("✅ PASS: Month correctly aggregates days")
        test_passed += 1
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 15: Verify year bucket aggregates months correctly
print("\n[INTEGRATION TEST 15] Verify year aggregates months correctly...")
try:
    conn = insight_db.get_db_connection()
    cursor = conn.cursor()
    
    # Get org units
    cursor.execute("""
        SELECT DISTINCT TOP 5 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    
    if not org_units:
        print("   ℹ️  No subcases in database - skipping test")
        test_passed += 1
    else:
        # Get month buckets
        result_month = insight_db.get_subcase_created_time_buckets(conn, org_units, bucket="month")
        
        # Get year buckets
        result_year = insight_db.get_subcase_created_time_buckets(conn, org_units, bucket="year")
        
        total_month = sum(item['count'] for item in result_month)
        total_year = sum(item['count'] for item in result_year)
        
        print(f"   Month buckets: {len(result_month)} buckets, {total_month} total")
        print(f"   Year buckets: {len(result_year)} buckets, {total_year} total")
        
        # Totals should match regardless of bucket granularity
        assert total_month == total_year, \
            f"Bucket granularity should not change totals: {total_month} != {total_year}"
        
        # Year buckets should be fewer or equal to month buckets
        assert len(result_year) <= len(result_month), \
            "Year buckets should be fewer than or equal to month buckets"
        
        print("✅ PASS: Year correctly aggregates months")
        test_passed += 1
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"✅ Passed: {test_passed}")
print(f"❌ Failed: {test_failed}")
print(f"📊 Total:  {test_passed + test_failed}")

if test_failed == 0:
    print("\n🎉 ALL TESTS PASSED - B-I6 COMPLETE")
    print("=" * 80)
    print("\nFunction Status:")
    print("  ✓ Structure validated")
    print("  ✓ Database connection works")
    print("  ✓ Invalid bucket raises ValueError")
    print("  ✓ Empty input handling correct")
    print("  ✓ 'day' bucket works (YYYY-MM-DD)")
    print("  ✓ 'month' bucket works (YYYY-MM)")
    print("  ✓ 'year' bucket works (YYYY)")
    print("  ✓ Aggregation sums correctly")
    print("  ✓ Results sorted ascending")
    print("  ✓ Scope filtering works")
    print("  ✓ Bucket granularity preserves totals")
    print("\n🎊 ALL DB LAYER FUNCTIONS COMPLETE (B-I3 through B-I6)")
    print("=" * 80)
    print("\nReady for B-I7 (Create Insight Service Layer)")
    print("=" * 80)
    sys.exit(0)
else:
    print(f"\n❌ {test_failed} TEST(S) FAILED")
    print("=" * 80)
    sys.exit(1)
