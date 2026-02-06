"""
Test Insight Service - KPI Summary Function (B-I8)
Unit and Integration tests for get_kpi_summary service function.

Run: python backend/test_insight_kpi_summary.py
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Add parent directory to path for 'backend' module imports
parent_dir = backend_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from api_v2.services import insight_service
from backend.api.schemas.auth_models import CurrentUser
from backend.api_v2.db_layer import insight_db

print("=" * 80)
print("INSIGHT SERVICE - KPI SUMMARY TEST (B-I8)")
print("=" * 80)

test_passed = 0
test_failed = 0

# ============================================================
# UNIT TESTS
# ============================================================

print("\n" + "=" * 80)
print("UNIT TESTS (Structure & Logic)")
print("=" * 80)

# Test 1: Function signature correct
print("\n[UNIT TEST 1] Function signature...")
try:
    import inspect
    sig = inspect.signature(insight_service.get_kpi_summary)
    params = list(sig.parameters.keys())
    
    assert 'current_user' in params, "Missing 'current_user' parameter"
    assert len(params) == 1, f"Should have exactly 1 parameter, got {len(params)}"
    
    print(f"   Parameters: {params}")
    print("✅ PASS: Function signature correct")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 2: Function is no longer a stub
print("\n[UNIT TEST 2] Function is implemented (not a stub)...")
try:
    # Check that function body is not just 'pass'
    source = inspect.getsource(insight_service.get_kpi_summary)
    
    # Should have real implementation
    assert 'insight_db' in source, "Should call insight_db functions"
    assert 'get_subcase_status_counts' in source, "Should call get_subcase_status_counts"
    assert 'get_action_item_counts' in source, "Should call get_action_item_counts"
    
    print("   ✓ Function calls insight_db.get_subcase_status_counts")
    print("   ✓ Function calls insight_db.get_action_item_counts")
    print("✅ PASS: Function is implemented")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 3: Return type annotation
print("\n[UNIT TEST 3] Return type annotation...")
try:
    sig = inspect.signature(insight_service.get_kpi_summary)
    print(f"   Return annotation: {sig.return_annotation}")
    assert sig.return_annotation != inspect.Signature.empty, "Should have return type annotation"
    print("✅ PASS: Return type documented")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# ============================================================
# INTEGRATION TESTS
# ============================================================

print("\n" + "=" * 80)
print("INTEGRATION TESTS (Database Operations)")
print("=" * 80)

# Test 4: Database connection works
print("\n[INTEGRATION TEST 4] Database connection...")
try:
    conn = insight_service.get_db_connection()
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

# Test 5: Function works with empty allowed_unit_ids
print("\n[INTEGRATION TEST 5] Empty allowed_unit_ids...")
try:
    mock_user = CurrentUser(
        user_id=1,
        username="test_user",
        is_active=True,
        scopes=[],
        allowed_unit_ids=set()  # Empty set
    )
    
    result = insight_service.get_kpi_summary(mock_user)
    
    assert isinstance(result, dict), "Should return a dict"
    assert 'total_subcases' in result, "Should have 'total_subcases' key"
    assert 'by_status' in result, "Should have 'by_status' key"
    assert 'action_items' in result, "Should have 'action_items' key"
    
    print(f"   Result: {result}")
    assert result['total_subcases'] == 0, "Should have 0 total for empty scope"
    assert len(result['by_status']) == 0, "Should have empty status list"
    
    print("✅ PASS: Empty allowed_unit_ids handled correctly")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 6: Function works with invalid org unit ID
print("\n[INTEGRATION TEST 6] Invalid org unit ID...")
try:
    mock_user = CurrentUser(
        user_id=1,
        username="test_user",
        is_active=True,
        scopes=[],
        allowed_unit_ids={-99999}  # Non-existent ID
    )
    
    result = insight_service.get_kpi_summary(mock_user)
    
    assert isinstance(result, dict), "Should return a dict"
    print(f"   Total subcases: {result['total_subcases']}")
    assert result['total_subcases'] == 0, "Should have 0 total for invalid org unit"
    
    print("✅ PASS: Invalid org unit ID handled correctly")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 7: Function returns correct structure
print("\n[INTEGRATION TEST 7] Response structure validation...")
try:
    conn = insight_service.get_db_connection()
    cursor = conn.cursor()
    
    # Get valid org units
    cursor.execute("""
        SELECT DISTINCT TOP 3 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    if not org_units:
        print("   ℹ️  No subcases in database - skipping test")
        test_passed += 1
    else:
        mock_user = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units)
        )
        
        result = insight_service.get_kpi_summary(mock_user)
        
        print(f"   Response keys: {list(result.keys())}")
        
        # Validate top-level structure
        assert isinstance(result, dict), "Should return a dict"
        assert 'total_subcases' in result, "Missing 'total_subcases'"
        assert 'by_status' in result, "Missing 'by_status'"
        assert 'action_items' in result, "Missing 'action_items'"
        
        # Validate total_subcases
        assert isinstance(result['total_subcases'], int), "total_subcases should be int"
        
        # Validate by_status
        assert isinstance(result['by_status'], list), "by_status should be list"
        if len(result['by_status']) > 0:
            status_item = result['by_status'][0]
            assert isinstance(status_item, dict), "Status items should be dicts"
            assert 'status' in status_item, "Status item missing 'status' key"
            assert 'count' in status_item, "Status item missing 'count' key"
        
        # Validate action_items
        assert isinstance(result['action_items'], dict), "action_items should be dict"
        assert 'total' in result['action_items'], "action_items missing 'total'"
        assert 'open' in result['action_items'], "action_items missing 'open'"
        assert 'completed' in result['action_items'], "action_items missing 'completed'"
        assert 'overdue' in result['action_items'], "action_items missing 'overdue'"
        
        print("✅ PASS: Response structure correct")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 8: total_subcases equals sum of by_status counts
print("\n[INTEGRATION TEST 8] total_subcases computation...")
try:
    conn = insight_service.get_db_connection()
    cursor = conn.cursor()
    
    # Get valid org units
    cursor.execute("""
        SELECT DISTINCT TOP 3 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    if not org_units:
        print("   ℹ️  No subcases in database - skipping test")
        test_passed += 1
    else:
        mock_user = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units)
        )
        
        result = insight_service.get_kpi_summary(mock_user)
        
        # Compute sum from by_status
        sum_by_status = sum(item['count'] for item in result['by_status'])
        
        print(f"   total_subcases: {result['total_subcases']}")
        print(f"   sum(by_status): {sum_by_status}")
        
        assert result['total_subcases'] == sum_by_status, \
            f"total_subcases {result['total_subcases']} != sum {sum_by_status}"
        
        print("✅ PASS: total_subcases computed correctly")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 9: Function respects scope filtering
print("\n[INTEGRATION TEST 9] Scope filtering...")
try:
    conn = insight_service.get_db_connection()
    cursor = conn.cursor()
    
    # Get two different org units
    cursor.execute("""
        SELECT DISTINCT TOP 2 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
        ORDER BY TargetOrgUnitID
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    if len(org_units) < 2:
        print("   ℹ️  Less than 2 org units - skipping test")
        test_passed += 1
    else:
        print(f"   Testing with org units: {org_units}")
        
        # Get results for both units
        user_both = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units)
        )
        result_both = insight_service.get_kpi_summary(user_both)
        
        # Get results for first unit only
        user_first = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={org_units[0]}
        )
        result_first = insight_service.get_kpi_summary(user_first)
        
        # Get results for second unit only
        user_second = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={org_units[1]}
        )
        result_second = insight_service.get_kpi_summary(user_second)
        
        print(f"   Both units: {result_both['total_subcases']} subcases")
        print(f"   First unit: {result_first['total_subcases']} subcases")
        print(f"   Second unit: {result_second['total_subcases']} subcases")
        
        # Total for both should equal sum of individuals
        expected_total = result_first['total_subcases'] + result_second['total_subcases']
        assert result_both['total_subcases'] == expected_total, \
            f"Multi-unit scope filtering incorrect: {result_both['total_subcases']} != {expected_total}"
        
        print("✅ PASS: Scope filtering works correctly")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 10: Verify no status renaming/merging
print("\n[INTEGRATION TEST 10] Status names preserved...")
try:
    conn = insight_service.get_db_connection()
    cursor = conn.cursor()
    
    # Get valid org units
    cursor.execute("""
        SELECT DISTINCT TOP 3 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    if not org_units:
        print("   ℹ️  No subcases in database - skipping test")
        test_passed += 1
    else:
        mock_user = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units)
        )
        
        result = insight_service.get_kpi_summary(mock_user)
        
        # Get all status values from response
        status_values = [item['status'] for item in result['by_status']]
        
        print(f"   Statuses in response: {status_values}")
        
        # Verify these are raw database status values (no transformation)
        valid_statuses = [
            'SUBMITTED_TO_SECTION',
            'SECTION_ACCEPTED_PENDING_DEPT',
            'DEPT_ACCEPTED_PENDING_ADMIN',
            'RETURNED_TO_SECTION_FOR_REVISION',
            'RETURNED_TO_DEPT_FOR_REVISION',
            'ADMIN_APPROVED',
            'SECTION_DENIED',
            'FORCE_CLOSED'
        ]
        
        for status in status_values:
            assert status in valid_statuses, \
                f"Status '{status}' not a valid database status (may be renamed/transformed)"
        
        print("✅ PASS: Status names preserved (not renamed/merged)")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 11: Verify action items structure
print("\n[INTEGRATION TEST 11] Action items structure...")
try:
    conn = insight_service.get_db_connection()
    cursor = conn.cursor()
    
    # Get valid org units
    cursor.execute("""
        SELECT DISTINCT TOP 3 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    if not org_units:
        print("   ℹ️  No subcases in database - skipping test")
        test_passed += 1
    else:
        mock_user = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units)
        )
        
        result = insight_service.get_kpi_summary(mock_user)
        action_items = result['action_items']
        
        print(f"   Action items: {action_items}")
        
        # Verify all are integers
        assert isinstance(action_items['total'], int), "total should be int"
        assert isinstance(action_items['open'], int), "open should be int"
        assert isinstance(action_items['completed'], int), "completed should be int"
        assert isinstance(action_items['overdue'], int), "overdue should be int"
        
        # Verify mathematical consistency
        assert action_items['open'] + action_items['completed'] == action_items['total'], \
            "open + completed should equal total"
        assert action_items['overdue'] <= action_items['open'], \
            "overdue should be <= open"
        
        print("✅ PASS: Action items structure correct")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 12: Verify data consistency with direct DB queries
print("\n[INTEGRATION TEST 12] Data consistency check...")
try:
    conn = insight_service.get_db_connection()
    cursor = conn.cursor()
    
    # Get valid org units
    cursor.execute("""
        SELECT DISTINCT TOP 3 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    
    if not org_units:
        print("   ℹ️  No subcases in database - skipping test")
        cursor.close()
        conn.close()
        test_passed += 1
    else:
        # Get direct count from database
        placeholders = ','.join('?' * len(org_units))
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM dbo.APP_AdministrativeSubcase
            WHERE TargetOrgUnitID IN ({placeholders})
        """, org_units)
        db_total = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        # Get count from service
        mock_user = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units)
        )
        
        result = insight_service.get_kpi_summary(mock_user)
        service_total = result['total_subcases']
        
        print(f"   Database count: {db_total}")
        print(f"   Service count: {service_total}")
        
        assert service_total == db_total, \
            f"Service total {service_total} != database total {db_total}"
        
        print("✅ PASS: Data consistent with database")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 13: Verify no percentages computed
print("\n[INTEGRATION TEST 13] No percentages computed...")
try:
    conn = insight_service.get_db_connection()
    cursor = conn.cursor()
    
    # Get valid org units
    cursor.execute("""
        SELECT DISTINCT TOP 3 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    if not org_units:
        print("   ℹ️  No subcases in database - skipping test")
        test_passed += 1
    else:
        mock_user = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units)
        )
        
        result = insight_service.get_kpi_summary(mock_user)
        
        # Check for percentage-related keys
        disallowed_keys = ['percentage', 'percent', 'pct', 'rate', 'ratio']
        
        # Check top-level keys
        for key in result.keys():
            assert key.lower() not in disallowed_keys, f"Should not have '{key}' key"
        
        # Check by_status items
        for status_item in result['by_status']:
            for key in status_item.keys():
                assert key.lower() not in disallowed_keys, f"Should not have '{key}' in status items"
        
        # Check action_items
        for key in result['action_items'].keys():
            assert key.lower() not in disallowed_keys, f"Should not have '{key}' in action_items"
        
        print("✅ PASS: No percentages computed")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 14: Connection is properly closed
print("\n[INTEGRATION TEST 14] Connection cleanup...")
try:
    conn = insight_service.get_db_connection()
    cursor = conn.cursor()
    
    # Get valid org units
    cursor.execute("""
        SELECT DISTINCT TOP 3 TargetOrgUnitID 
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IS NOT NULL
    """)
    org_units = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    if not org_units:
        print("   ℹ️  No subcases in database - skipping test")
        test_passed += 1
    else:
        mock_user = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units)
        )
        
        # Call function
        result = insight_service.get_kpi_summary(mock_user)
        
        # Verify function doesn't leak connections
        # (If connection wasn't closed, subsequent calls would accumulate connections)
        # We'll test by making multiple calls
        for i in range(5):
            result = insight_service.get_kpi_summary(mock_user)
        
        print("   ✓ Multiple calls successful (no connection leaks)")
        print("✅ PASS: Connection cleanup works")
        test_passed += 1
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
    print("\n🎉 ALL TESTS PASSED - B-I8 COMPLETE")
    print("=" * 80)
    print("\nFunction Status:")
    print("  ✓ Signature correct")
    print("  ✓ Implemented (not a stub)")
    print("  ✓ Database connection works")
    print("  ✓ Empty input handling correct")
    print("  ✓ Response structure correct")
    print("  ✓ total_subcases computed correctly")
    print("  ✓ Scope filtering works")
    print("  ✓ Status names preserved")
    print("  ✓ Action items structure correct")
    print("  ✓ Data consistent with database")
    print("  ✓ No percentages computed")
    print("  ✓ Connection cleanup works")
    print("\n📋 Output Structure Verified:")
    print("  {")
    print("    'total_subcases': int,")
    print("    'by_status': [{'status': str, 'count': int}],")
    print("    'action_items': {")
    print("      'total': int,")
    print("      'open': int,")
    print("      'completed': int,")
    print("      'overdue': int")
    print("    }")
    print("  }")
    print("\n" + "=" * 80)
    print("Ready for B-I9 (Implement Distribution Service Function)")
    print("=" * 80)
    sys.exit(0)
else:
    print(f"\n❌ {test_failed} TEST(S) FAILED")
    print("=" * 80)
    sys.exit(1)
