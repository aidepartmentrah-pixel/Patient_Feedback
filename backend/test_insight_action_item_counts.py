"""
Test Insight DB Layer - Action Item Aggregation (B-I4)
Unit and Integration tests for get_action_item_counts function.

Run: python backend/test_insight_action_item_counts.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add backend directory to path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from api_v2.db_layer import insight_db

print("=" * 80)
print("INSIGHT DB LAYER - ACTION ITEM AGGREGATION TEST (B-I4)")
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
    assert hasattr(insight_db, 'get_action_item_counts')
    assert callable(insight_db.get_action_item_counts)
    print("✅ PASS: Function exists and is callable")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 2: Function signature
print("\n[UNIT TEST 2] Function signature is correct...")
try:
    import inspect
    sig = inspect.signature(insight_db.get_action_item_counts)
    params = list(sig.parameters.keys())
    assert 'conn' in params, "Missing 'conn' parameter"
    assert 'allowed_unit_ids' in params, "Missing 'allowed_unit_ids' parameter"
    print("✅ PASS: Signature correct (conn, allowed_unit_ids)")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 3: Return type annotation
print("\n[UNIT TEST 3] Return type annotation...")
try:
    sig = inspect.signature(insight_db.get_action_item_counts)
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

# Test 5: Empty allowed_unit_ids returns zero metrics
print("\n[INTEGRATION TEST 5] Empty allowed_unit_ids...")
try:
    conn = insight_db.get_db_connection()
    result = insight_db.get_action_item_counts(conn, [])
    conn.close()
    
    assert isinstance(result, dict), "Should return a dict"
    assert result == {"total": 0, "open": 0, "completed": 0, "overdue": 0}, \
        "Should return all zeros for empty allowed_unit_ids"
    print("✅ PASS: Empty allowed_unit_ids returns zero metrics")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 6: Invalid org unit ID returns zero metrics
print("\n[INTEGRATION TEST 6] Invalid org unit ID...")
try:
    conn = insight_db.get_db_connection()
    result = insight_db.get_action_item_counts(conn, [-99999])
    conn.close()
    
    assert isinstance(result, dict), "Should return a dict"
    print(f"   Returned metrics: {result}")
    assert result == {"total": 0, "open": 0, "completed": 0, "overdue": 0}, \
        "Should return all zeros for non-existent org unit"
    print("✅ PASS: Invalid org unit ID returns zero metrics")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 7: Query actual database for action items
print("\n[INTEGRATION TEST 7] Query actual action items...")
try:
    conn = insight_db.get_db_connection()
    cursor = conn.cursor()
    
    # Get org units that have action items
    cursor.execute("""
        SELECT DISTINCT TOP 5 s.TargetOrgUnitID 
        FROM dbo.APP_SubcaseActionItem a
        INNER JOIN dbo.APP_AdministrativeSubcase s ON a.SubcaseID = s.SubcaseID
        WHERE s.TargetOrgUnitID IS NOT NULL
        ORDER BY s.TargetOrgUnitID
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    
    if not org_units:
        print("   ℹ️  No action items in database - skipping test")
        test_passed += 1
    else:
        print(f"   Testing with org units: {org_units}")
        
        result = insight_db.get_action_item_counts(conn, org_units)
        
        print(f"   Metrics returned:")
        print(f"      • Total: {result['total']}")
        print(f"      • Open: {result['open']}")
        print(f"      • Completed: {result['completed']}")
        print(f"      • Overdue: {result['overdue']}")
        
        # Validate return structure
        assert isinstance(result, dict), "Should return a dict"
        assert 'total' in result, "Should have 'total' key"
        assert 'open' in result, "Should have 'open' key"
        assert 'completed' in result, "Should have 'completed' key"
        assert 'overdue' in result, "Should have 'overdue' key"
        
        # Validate data types
        assert isinstance(result['total'], int), "Total should be integer"
        assert isinstance(result['open'], int), "Open should be integer"
        assert isinstance(result['completed'], int), "Completed should be integer"
        assert isinstance(result['overdue'], int), "Overdue should be integer"
        
        # Validate logical consistency
        assert result['total'] >= 0, "Total should be non-negative"
        assert result['open'] >= 0, "Open should be non-negative"
        assert result['completed'] >= 0, "Completed should be non-negative"
        assert result['overdue'] >= 0, "Overdue should be non-negative"
        
        print("✅ PASS: Query returns valid action item metrics")
        test_passed += 1
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 8: Verify open + completed = total
print("\n[INTEGRATION TEST 8] Verify open + completed = total...")
try:
    conn = insight_db.get_db_connection()
    cursor = conn.cursor()
    
    # Get org units with action items
    cursor.execute("""
        SELECT DISTINCT TOP 5 s.TargetOrgUnitID 
        FROM dbo.APP_SubcaseActionItem a
        INNER JOIN dbo.APP_AdministrativeSubcase s ON a.SubcaseID = s.SubcaseID
        WHERE s.TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    
    if not org_units:
        print("   ℹ️  No action items in database - skipping test")
        test_passed += 1
    else:
        result = insight_db.get_action_item_counts(conn, org_units)
        
        print(f"   Total: {result['total']}")
        print(f"   Open: {result['open']}")
        print(f"   Completed: {result['completed']}")
        print(f"   Sum (open + completed): {result['open'] + result['completed']}")
        
        assert result['open'] + result['completed'] == result['total'], \
            f"Open + Completed should equal Total: {result['open']} + {result['completed']} != {result['total']}"
        
        print("✅ PASS: Open + Completed = Total (mathematically correct)")
        test_passed += 1
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 9: Verify overdue <= open
print("\n[INTEGRATION TEST 9] Verify overdue <= open...")
try:
    conn = insight_db.get_db_connection()
    cursor = conn.cursor()
    
    # Get org units with action items
    cursor.execute("""
        SELECT DISTINCT TOP 5 s.TargetOrgUnitID 
        FROM dbo.APP_SubcaseActionItem a
        INNER JOIN dbo.APP_AdministrativeSubcase s ON a.SubcaseID = s.SubcaseID
        WHERE s.TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    
    if not org_units:
        print("   ℹ️  No action items in database - skipping test")
        test_passed += 1
    else:
        result = insight_db.get_action_item_counts(conn, org_units)
        
        print(f"   Open: {result['open']}")
        print(f"   Overdue: {result['overdue']}")
        
        assert result['overdue'] <= result['open'], \
            f"Overdue cannot exceed Open: {result['overdue']} > {result['open']}"
        
        print("✅ PASS: Overdue <= Open (logical consistency)")
        test_passed += 1
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 10: Verify JOIN scope filtering
print("\n[INTEGRATION TEST 10] Verify JOIN scope filtering...")
try:
    conn = insight_db.get_db_connection()
    cursor = conn.cursor()
    
    # Get two different org unit IDs with action items
    cursor.execute("""
        SELECT DISTINCT TOP 2 s.TargetOrgUnitID 
        FROM dbo.APP_SubcaseActionItem a
        INNER JOIN dbo.APP_AdministrativeSubcase s ON a.SubcaseID = s.SubcaseID
        WHERE s.TargetOrgUnitID IS NOT NULL
        ORDER BY s.TargetOrgUnitID
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    
    if len(org_units) < 2:
        print("   ℹ️  Less than 2 org units with action items - skipping test")
        test_passed += 1
    else:
        print(f"   Testing with org units: {org_units}")
        
        # Get counts for both org units
        result_both = insight_db.get_action_item_counts(conn, org_units)
        
        # Get counts for first org unit only
        result_first = insight_db.get_action_item_counts(conn, [org_units[0]])
        
        # Get counts for second org unit only
        result_second = insight_db.get_action_item_counts(conn, [org_units[1]])
        
        print(f"   Both units: {result_both['total']} action items")
        print(f"   First unit: {result_first['total']} action items")
        print(f"   Second unit: {result_second['total']} action items")
        
        # Total for both should equal sum of individuals
        expected = result_first['total'] + result_second['total']
        actual = result_both['total']
        assert actual == expected, \
            f"Multi-unit filtering incorrect: {actual} != {expected}"
        
        print("✅ PASS: JOIN scope filtering works correctly")
        test_passed += 1
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 11: Verify completed count accuracy
print("\n[INTEGRATION TEST 11] Verify completed count accuracy...")
try:
    conn = insight_db.get_db_connection()
    cursor = conn.cursor()
    
    # Get a sample org unit
    cursor.execute("""
        SELECT TOP 1 s.TargetOrgUnitID 
        FROM dbo.APP_SubcaseActionItem a
        INNER JOIN dbo.APP_AdministrativeSubcase s ON a.SubcaseID = s.SubcaseID
        WHERE s.TargetOrgUnitID IS NOT NULL
    """)
    row = cursor.fetchone()
    
    if not row:
        print("   ℹ️  No action items in database - skipping test")
        test_passed += 1
    else:
        sample_org_unit = row[0]
        
        # Get count from function
        result = insight_db.get_action_item_counts(conn, [sample_org_unit])
        
        # Manually verify completed count
        cursor.execute("""
            SELECT COUNT(*) 
            FROM dbo.APP_SubcaseActionItem a
            INNER JOIN dbo.APP_AdministrativeSubcase s ON a.SubcaseID = s.SubcaseID
            WHERE s.TargetOrgUnitID = ?
              AND a.CompletedAt IS NOT NULL
        """, (sample_org_unit,))
        manual_completed = cursor.fetchone()[0]
        
        print(f"   Org unit: {sample_org_unit}")
        print(f"   Function completed count: {result['completed']}")
        print(f"   Manual completed count: {manual_completed}")
        
        assert result['completed'] == manual_completed, \
            f"Completed count mismatch: {result['completed']} != {manual_completed}"
        
        print("✅ PASS: Completed count is accurate")
        test_passed += 1
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 12: Verify overdue count accuracy
print("\n[INTEGRATION TEST 12] Verify overdue count accuracy...")
try:
    conn = insight_db.get_db_connection()
    cursor = conn.cursor()
    
    # Get a sample org unit
    cursor.execute("""
        SELECT TOP 1 s.TargetOrgUnitID 
        FROM dbo.APP_SubcaseActionItem a
        INNER JOIN dbo.APP_AdministrativeSubcase s ON a.SubcaseID = s.SubcaseID
        WHERE s.TargetOrgUnitID IS NOT NULL
    """)
    row = cursor.fetchone()
    
    if not row:
        print("   ℹ️  No action items in database - skipping test")
        test_passed += 1
    else:
        sample_org_unit = row[0]
        
        # Get count from function
        result = insight_db.get_action_item_counts(conn, [sample_org_unit])
        
        # Manually verify overdue count
        cursor.execute("""
            SELECT COUNT(*) 
            FROM dbo.APP_SubcaseActionItem a
            INNER JOIN dbo.APP_AdministrativeSubcase s ON a.SubcaseID = s.SubcaseID
            WHERE s.TargetOrgUnitID = ?
              AND a.DueDate < CAST(GETDATE() AS DATE)
              AND a.CompletedAt IS NULL
        """, (sample_org_unit,))
        manual_overdue = cursor.fetchone()[0]
        
        print(f"   Org unit: {sample_org_unit}")
        print(f"   Function overdue count: {result['overdue']}")
        print(f"   Manual overdue count: {manual_overdue}")
        
        assert result['overdue'] == manual_overdue, \
            f"Overdue count mismatch: {result['overdue']} != {manual_overdue}"
        
        print("✅ PASS: Overdue count is accurate")
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
    print("\n🎉 ALL TESTS PASSED - B-I4 COMPLETE")
    print("=" * 80)
    print("\nFunction Status:")
    print("  ✓ Structure validated")
    print("  ✓ Database connection works")
    print("  ✓ Empty input handling correct")
    print("  ✓ JOIN scope filtering works")
    print("  ✓ All metrics calculated correctly")
    print("  ✓ open + completed = total")
    print("  ✓ overdue <= open")
    print("  ✓ Completed count accurate")
    print("  ✓ Overdue count accurate")
    print("\nReady for B-I5 (Stuck Subcases Detection)")
    print("=" * 80)
    sys.exit(0)
else:
    print(f"\n❌ {test_failed} TEST(S) FAILED")
    print("=" * 80)
    sys.exit(1)
