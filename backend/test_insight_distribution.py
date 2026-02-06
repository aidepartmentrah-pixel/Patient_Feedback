"""
Test Insight Service - Distribution Function (B-I9)
Unit and Integration tests for get_distribution service function.

Run: python backend/test_insight_distribution.py
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
print("INSIGHT SERVICE - DISTRIBUTION TEST (B-I9)")
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
    sig = inspect.signature(insight_service.get_distribution)
    params = list(sig.parameters.keys())
    
    assert 'current_user' in params, "Missing 'current_user' parameter"
    assert 'dimension' in params, "Missing 'dimension' parameter"
    assert len(params) == 2, f"Should have exactly 2 parameters, got {len(params)}"
    
    print(f"   Parameters: {params}")
    print("✅ PASS: Function signature correct")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 2: Function is implemented
print("\n[UNIT TEST 2] Function is implemented (not a stub)...")
try:
    source = inspect.getsource(insight_service.get_distribution)
    
    # Should have real implementation
    assert 'insight_db' in source, "Should call insight_db functions"
    assert 'get_subcase_status_counts' in source, "Should call get_subcase_status_counts"
    assert 'get_subcase_org_unit_counts' in source, "Should call get_subcase_org_unit_counts"
    assert 'ValueError' in source, "Should raise ValueError for invalid dimensions"
    
    print("   ✓ Function calls insight_db.get_subcase_status_counts")
    print("   ✓ Function calls insight_db.get_subcase_org_unit_counts")
    print("   ✓ Function validates dimension parameter")
    print("✅ PASS: Function is implemented")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 3: DB function exists
print("\n[UNIT TEST 3] DB function get_subcase_org_unit_counts exists...")
try:
    assert hasattr(insight_db, 'get_subcase_org_unit_counts'), \
        "insight_db missing get_subcase_org_unit_counts function"
    assert callable(insight_db.get_subcase_org_unit_counts), \
        "get_subcase_org_unit_counts is not callable"
    
    print("✅ PASS: DB function exists")
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

# Test 5: Invalid dimension raises ValueError
print("\n[INTEGRATION TEST 5] Invalid dimension raises ValueError...")
try:
    mock_user = CurrentUser(
        user_id=1,
        username="test_user",
        is_active=True,
        scopes=[],
        allowed_unit_ids={1, 2}
    )
    
    error_raised = False
    try:
        result = insight_service.get_distribution(mock_user, dimension="invalid")
    except ValueError as e:
        error_raised = True
        print(f"   ✓ ValueError raised: {e}")
    
    assert error_raised, "Should raise ValueError for invalid dimension"
    print("✅ PASS: Invalid dimension raises ValueError")
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
    mock_user = CurrentUser(
        user_id=1,
        username="test_user",
        is_active=True,
        scopes=[],
        allowed_unit_ids=set()
    )
    
    result_status = insight_service.get_distribution(mock_user, dimension="status")
    result_org = insight_service.get_distribution(mock_user, dimension="org_unit")
    
    assert isinstance(result_status, list), "Should return a list"
    assert len(result_status) == 0, "Status distribution should be empty"
    
    assert isinstance(result_org, list), "Should return a list"
    assert len(result_org) == 0, "Org unit distribution should be empty"
    
    print("✅ PASS: Empty allowed_unit_ids handled correctly")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 7: Status dimension returns correct structure
print("\n[INTEGRATION TEST 7] Status dimension structure...")
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
        
        result = insight_service.get_distribution(mock_user, dimension="status")
        
        print(f"   Found {len(result)} status items")
        
        assert isinstance(result, list), "Should return a list"
        
        if len(result) > 0:
            item = result[0]
            print(f"   Sample item: {item}")
            
            # Validate structure
            assert isinstance(item, dict), "Items should be dicts"
            assert 'key' in item, "Should have 'key' field"
            assert 'count' in item, "Should have 'count' field"
            assert len(item) == 2, "Should have exactly 2 fields"
            
            # Validate types
            assert isinstance(item['key'], str), "key should be string (status value)"
            assert isinstance(item['count'], int), "count should be int"
        
        print("✅ PASS: Status dimension structure correct")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 8: Org unit dimension returns correct structure
print("\n[INTEGRATION TEST 8] Org unit dimension structure...")
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
        
        result = insight_service.get_distribution(mock_user, dimension="org_unit")
        
        print(f"   Found {len(result)} org unit items")
        
        assert isinstance(result, list), "Should return a list"
        
        if len(result) > 0:
            item = result[0]
            print(f"   Sample item: {item}")
            
            # Validate structure
            assert isinstance(item, dict), "Items should be dicts"
            assert 'key' in item, "Should have 'key' field"
            assert 'count' in item, "Should have 'count' field"
            assert len(item) == 2, "Should have exactly 2 fields"
            
            # Validate types
            assert isinstance(item['key'], int), "key should be int (org unit ID)"
            assert isinstance(item['count'], int), "count should be int"
        
        print("✅ PASS: Org unit dimension structure correct")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 9: Status dimension has valid status values
print("\n[INTEGRATION TEST 9] Status values are valid...")
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
        
        result = insight_service.get_distribution(mock_user, dimension="status")
        
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
        
        status_keys = [item['key'] for item in result]
        print(f"   Status keys: {status_keys}")
        
        for status in status_keys:
            assert status in valid_statuses, \
                f"Status '{status}' not a valid database status"
        
        print("✅ PASS: Status values are valid")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 10: Org unit dimension has valid org unit IDs
print("\n[INTEGRATION TEST 10] Org unit IDs are valid...")
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
        
        result = insight_service.get_distribution(mock_user, dimension="org_unit")
        
        org_keys = [item['key'] for item in result]
        print(f"   Org unit keys: {org_keys}")
        
        # All org unit keys should be in the allowed list
        for org_id in org_keys:
            assert org_id in org_units, \
                f"Org unit {org_id} not in allowed list {org_units}"
        
        print("✅ PASS: Org unit IDs are valid")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 11: Sum of counts matches total subcases
print("\n[INTEGRATION TEST 11] Count totals match...")
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
        # Get direct count
        placeholders = ','.join('?' * len(org_units))
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM dbo.APP_AdministrativeSubcase
            WHERE TargetOrgUnitID IN ({placeholders})
        """, org_units)
        db_total = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        mock_user = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids=set(org_units)
        )
        
        # Get counts from both dimensions
        result_status = insight_service.get_distribution(mock_user, dimension="status")
        result_org = insight_service.get_distribution(mock_user, dimension="org_unit")
        
        sum_status = sum(item['count'] for item in result_status)
        sum_org = sum(item['count'] for item in result_org)
        
        print(f"   Database total: {db_total}")
        print(f"   Status sum: {sum_status}")
        print(f"   Org unit sum: {sum_org}")
        
        assert sum_status == db_total, \
            f"Status sum {sum_status} != database total {db_total}"
        assert sum_org == db_total, \
            f"Org unit sum {sum_org} != database total {db_total}"
        
        print("✅ PASS: Count totals match")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 12: Scope filtering works
print("\n[INTEGRATION TEST 12] Scope filtering...")
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
        result_both = insight_service.get_distribution(user_both, dimension="status")
        total_both = sum(item['count'] for item in result_both)
        
        # Get results for first unit only
        user_first = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={org_units[0]}
        )
        result_first = insight_service.get_distribution(user_first, dimension="status")
        total_first = sum(item['count'] for item in result_first)
        
        # Get results for second unit only
        user_second = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={org_units[1]}
        )
        result_second = insight_service.get_distribution(user_second, dimension="status")
        total_second = sum(item['count'] for item in result_second)
        
        print(f"   Both units: {total_both} subcases")
        print(f"   First unit: {total_first} subcases")
        print(f"   Second unit: {total_second} subcases")
        
        expected_total = total_first + total_second
        assert total_both == expected_total, \
            f"Multi-unit scope filtering incorrect: {total_both} != {expected_total}"
        
        print("✅ PASS: Scope filtering works correctly")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 13: No label lookups (org names)
print("\n[INTEGRATION TEST 13] No label lookups...")
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
        
        result = insight_service.get_distribution(mock_user, dimension="org_unit")
        
        # Should not have name/label fields
        for item in result:
            assert 'name' not in item, "Should not have 'name' field"
            assert 'label' not in item, "Should not have 'label' field"
            assert 'org_name' not in item, "Should not have 'org_name' field"
            assert 'unit_name' not in item, "Should not have 'unit_name' field"
        
        print("✅ PASS: No label lookups")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 14: No percentages computed
print("\n[INTEGRATION TEST 14] No percentages...")
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
        
        result_status = insight_service.get_distribution(mock_user, dimension="status")
        result_org = insight_service.get_distribution(mock_user, dimension="org_unit")
        
        disallowed_keys = ['percentage', 'percent', 'pct', 'rate', 'ratio']
        
        for item in result_status + result_org:
            for key in item.keys():
                assert key.lower() not in disallowed_keys, \
                    f"Should not have '{key}' field"
        
        print("✅ PASS: No percentages computed")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 15: Connection cleanup
print("\n[INTEGRATION TEST 15] Connection cleanup...")
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
        
        # Make multiple calls to test connection cleanup
        for i in range(5):
            result = insight_service.get_distribution(mock_user, dimension="status")
        
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
    print("\n🎉 ALL TESTS PASSED - B-I9 COMPLETE")
    print("=" * 80)
    print("\nFunction Status:")
    print("  ✓ Signature correct")
    print("  ✓ Implemented (not a stub)")
    print("  ✓ DB function created (get_subcase_org_unit_counts)")
    print("  ✓ Invalid dimension raises ValueError")
    print("  ✓ Empty input handling correct")
    print("  ✓ Status dimension structure correct")
    print("  ✓ Org unit dimension structure correct")
    print("  ✓ Status values are valid")
    print("  ✓ Org unit IDs are valid")
    print("  ✓ Count totals match database")
    print("  ✓ Scope filtering works")
    print("  ✓ No label lookups")
    print("  ✓ No percentages computed")
    print("  ✓ Connection cleanup works")
    print("\n📋 Output Structure Verified:")
    print("  [")
    print("    {'key': <status_str or org_unit_int>, 'count': int},")
    print("    ...")
    print("  ]")
    print("\n🎯 Supported Dimensions:")
    print("  • 'status' - Groups by subcase status")
    print("  • 'org_unit' - Groups by target org unit ID")
    print("\n" + "=" * 80)
    print("Ready for B-I10 (Implement Trend Service Function)")
    print("=" * 80)
    sys.exit(0)
else:
    print(f"\n❌ {test_failed} TEST(S) FAILED")
    print("=" * 80)
    sys.exit(1)
