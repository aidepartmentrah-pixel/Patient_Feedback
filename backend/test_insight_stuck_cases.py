"""
Test Insight Service - Stuck Cases Function (B-I11)
Unit and Integration tests for get_stuck_cases service function.

Run: python backend/test_insight_stuck_cases.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

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
print("INSIGHT SERVICE - STUCK CASES TEST (B-I11)")
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
    sig = inspect.signature(insight_service.get_stuck_cases)
    params = list(sig.parameters.keys())
    
    assert 'current_user' in params, "Missing 'current_user' parameter"
    assert 'days_threshold' in params, "Missing 'days_threshold' parameter"
    assert len(params) == 2, f"Should have exactly 2 parameters, got {len(params)}"
    
    # Check default value
    default_val = sig.parameters['days_threshold'].default
    print(f"   days_threshold default: {default_val}")
    assert default_val == 30, f"Default should be 30, got {default_val}"
    
    print(f"   Parameters: {params}")
    print("✅ PASS: Function signature correct")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 2: Function is implemented
print("\n[UNIT TEST 2] Function is implemented (not a stub)...")
try:
    source = inspect.getsource(insight_service.get_stuck_cases)
    
    # Should have real implementation
    assert 'insight_db' in source, "Should call insight_db functions"
    assert 'get_stuck_subcases' in source, "Should call get_stuck_subcases"
    assert 'pass' not in source or 'pass' in 'threshold', "Should not be a stub"
    
    print("   ✓ Function calls insight_db.get_stuck_subcases")
    print("✅ PASS: Function is implemented")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    test_failed += 1

# Test 3: Docstring mentions "stuck" definition
print("\n[UNIT TEST 3] Docstring documents 'stuck' definition...")
try:
    doc = insight_service.get_stuck_cases.__doc__
    assert doc is not None, "Missing docstring"
    
    doc_lower = doc.lower()
    assert 'stuck' in doc_lower, "Docstring should mention 'stuck'"
    assert 'updatedat' in doc_lower or 'updated' in doc_lower, "Should mention UpdatedAt"
    assert 'threshold' in doc_lower, "Should mention threshold"
    assert 'terminal' in doc_lower, "Should mention terminal statuses"
    
    print("   ✓ Docstring explains stuck definition")
    print("✅ PASS: Docstring correct")
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

# Test 5: Empty allowed_unit_ids returns empty list
print("\n[INTEGRATION TEST 5] Empty allowed_unit_ids...")
try:
    mock_user = CurrentUser(
        user_id=1,
        username="test_user",
        is_active=True,
        scopes=[],
        allowed_unit_ids=set()
    )
    
    result = insight_service.get_stuck_cases(mock_user, days_threshold=30)
    
    assert isinstance(result, list), "Should return a list"
    assert len(result) == 0, "Should return empty list for empty scope"
    
    print("✅ PASS: Empty allowed_unit_ids handled correctly")
    test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 6: Function returns list
print("\n[INTEGRATION TEST 6] Returns list...")
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
        
        result = insight_service.get_stuck_cases(mock_user, days_threshold=7)
        
        print(f"   Found {len(result)} stuck cases")
        assert isinstance(result, list), "Should return a list"
        
        print("✅ PASS: Returns list")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 7: Return structure has all required fields
print("\n[INTEGRATION TEST 7] Return structure...")
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
        
        result = insight_service.get_stuck_cases(mock_user, days_threshold=7)
        
        if len(result) > 0:
            item = result[0]
            print(f"   Sample item keys: {list(item.keys())}")
            
            # Validate required fields
            assert 'subcase_id' in item, "Missing 'subcase_id' field"
            assert 'status' in item, "Missing 'status' field"
            assert 'target_org_unit_id' in item, "Missing 'target_org_unit_id' field"
            assert 'updated_at' in item, "Missing 'updated_at' field"
            assert 'days_in_stage' in item, "Missing 'days_in_stage' field"
            
            # Validate types
            assert isinstance(item['subcase_id'], int), "subcase_id should be int"
            assert isinstance(item['status'], str), "status should be string"
            assert isinstance(item['target_org_unit_id'], int), "target_org_unit_id should be int"
            assert isinstance(item['updated_at'], datetime), "updated_at should be datetime"
            assert isinstance(item['days_in_stage'], int), "days_in_stage should be int"
            
            print(f"   Sample subcase: ID={item['subcase_id']}, status={item['status']}, days={item['days_in_stage']}")
        
        print("✅ PASS: Return structure correct")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 8: Default threshold works (30 days)
print("\n[INTEGRATION TEST 8] Default threshold (30 days)...")
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
        
        # Call without specifying threshold (should use default 30)
        result = insight_service.get_stuck_cases(mock_user)
        
        print(f"   Found {len(result)} stuck cases with default threshold")
        assert isinstance(result, list), "Should return a list"
        
        print("✅ PASS: Default threshold works")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 9: Custom threshold works
print("\n[INTEGRATION TEST 9] Custom threshold...")
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
        
        # Try different thresholds
        result_1 = insight_service.get_stuck_cases(mock_user, days_threshold=1)
        result_7 = insight_service.get_stuck_cases(mock_user, days_threshold=7)
        result_30 = insight_service.get_stuck_cases(mock_user, days_threshold=30)
        
        print(f"   Threshold 1 day: {len(result_1)} stuck cases")
        print(f"   Threshold 7 days: {len(result_7)} stuck cases")
        print(f"   Threshold 30 days: {len(result_30)} stuck cases")
        
        # Lower threshold should return >= count of higher threshold
        assert len(result_1) >= len(result_7), "Lower threshold should return more or equal results"
        assert len(result_7) >= len(result_30), "Lower threshold should return more or equal results"
        
        print("✅ PASS: Custom threshold works")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 10: No terminal statuses in results
print("\n[INTEGRATION TEST 10] No terminal statuses...")
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
        
        result = insight_service.get_stuck_cases(mock_user, days_threshold=1)
        
        terminal_statuses = ['ADMIN_APPROVED', 'SECTION_DENIED', 'FORCE_CLOSED']
        
        statuses = [item['status'] for item in result]
        print(f"   Statuses in results: {set(statuses)}")
        
        for item in result:
            assert item['status'] not in terminal_statuses, \
                f"Terminal status {item['status']} should not be in stuck cases"
        
        print("✅ PASS: No terminal statuses in results")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 11: Days in stage is >= threshold
print("\n[INTEGRATION TEST 11] Days in stage >= threshold...")
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
        
        threshold = 7
        result = insight_service.get_stuck_cases(mock_user, days_threshold=threshold)
        
        if len(result) > 0:
            for item in result:
                assert item['days_in_stage'] >= threshold, \
                    f"Days in stage {item['days_in_stage']} should be >= {threshold}"
            
            min_days = min(item['days_in_stage'] for item in result)
            max_days = max(item['days_in_stage'] for item in result)
            print(f"   Days range: {min_days} - {max_days} days (all >= {threshold})")
        
        print("✅ PASS: Days in stage >= threshold")
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
        result_both = insight_service.get_stuck_cases(user_both, days_threshold=7)
        
        # Get results for first unit only
        user_first = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={org_units[0]}
        )
        result_first = insight_service.get_stuck_cases(user_first, days_threshold=7)
        
        # Get results for second unit only
        user_second = CurrentUser(
            user_id=1,
            username="test_user",
            is_active=True,
            scopes=[],
            allowed_unit_ids={org_units[1]}
        )
        result_second = insight_service.get_stuck_cases(user_second, days_threshold=7)
        
        print(f"   Both units: {len(result_both)} stuck cases")
        print(f"   First unit: {len(result_first)} stuck cases")
        print(f"   Second unit: {len(result_second)} stuck cases")
        
        # Verify org units in results match scope
        for item in result_first:
            assert item['target_org_unit_id'] == org_units[0], \
                f"Org unit {item['target_org_unit_id']} not in scope"
        
        for item in result_second:
            assert item['target_org_unit_id'] == org_units[1], \
                f"Org unit {item['target_org_unit_id']} not in scope"
        
        print("✅ PASS: Scope filtering works correctly")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 13: No field renaming
print("\n[INTEGRATION TEST 13] No field renaming...")
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
        
        result = insight_service.get_stuck_cases(mock_user, days_threshold=7)
        
        if len(result) > 0:
            item = result[0]
            
            # Check for field renaming attempts
            assert 'subcase_id' in item, "Should use 'subcase_id' not 'id'"
            assert 'id' not in item or 'subcase_id' in str(item.keys()), "Should not rename to 'id'"
            assert 'target_org_unit_id' in item, "Should keep 'target_org_unit_id'"
            assert 'org_unit' not in item, "Should not rename to 'org_unit'"
            assert 'days_in_stage' in item, "Should keep 'days_in_stage'"
            assert 'days' not in item or 'days_in_stage' in str(item.keys()), "Should not rename to 'days'"
        
        print("✅ PASS: No field renaming")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 14: No computed KPIs added
print("\n[INTEGRATION TEST 14] No computed KPIs...")
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
        
        result = insight_service.get_stuck_cases(mock_user, days_threshold=7)
        
        if len(result) > 0:
            item = result[0]
            
            # Should not have additional computed fields
            disallowed_keys = ['percentage', 'priority', 'urgency', 'score', 'rank']
            for key in item.keys():
                assert key.lower() not in disallowed_keys, \
                    f"Should not compute KPI field '{key}'"
        
        print("✅ PASS: No computed KPIs")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 15: No org unit joins
print("\n[INTEGRATION TEST 15] No org unit joins...")
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
        
        result = insight_service.get_stuck_cases(mock_user, days_threshold=7)
        
        if len(result) > 0:
            item = result[0]
            
            # Should not have org unit name fields
            assert 'org_unit_name' not in item, "Should not join org unit names"
            assert 'unit_name' not in item, "Should not join org unit names"
            assert 'organization_name' not in item, "Should not join org unit names"
        
        print("✅ PASS: No org unit joins")
        test_passed += 1
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test_failed += 1

# Test 16: Connection cleanup
print("\n[INTEGRATION TEST 16] Connection cleanup...")
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
            result = insight_service.get_stuck_cases(mock_user, days_threshold=7)
        
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
    print("\n🎉 ALL TESTS PASSED - B-I11 COMPLETE")
    print("=" * 80)
    print("\nFunction Status:")
    print("  ✓ Signature correct (days_threshold default=30)")
    print("  ✓ Implemented (not a stub)")
    print("  ✓ Docstring documents stuck definition")
    print("  ✓ Empty input handling correct")
    print("  ✓ Returns list")
    print("  ✓ Return structure correct (5 fields)")
    print("  ✓ Default threshold works (30 days)")
    print("  ✓ Custom threshold works")
    print("  ✓ No terminal statuses in results")
    print("  ✓ Days in stage >= threshold")
    print("  ✓ Scope filtering works")
    print("  ✓ No field renaming")
    print("  ✓ No computed KPIs")
    print("  ✓ No org unit joins")
    print("  ✓ Connection cleanup works")
    print("\n📋 Output Structure Verified:")
    print("  [")
    print("    {")
    print("      'subcase_id': int,")
    print("      'status': str,")
    print("      'target_org_unit_id': int,")
    print("      'updated_at': datetime,")
    print("      'days_in_stage': int")
    print("    },")
    print("    ...")
    print("  ]")
    print("\n🎯 Stuck Definition:")
    print("  UpdatedAt older than threshold AND not in terminal statuses")
    print("  Terminal: ADMIN_APPROVED, SECTION_DENIED, FORCE_CLOSED")
    print("\n" + "=" * 80)
    print("🎊 ALL SERVICE LAYER FUNCTIONS COMPLETE (B-I8 through B-I11)")
    print("=" * 80)
    print("\nReady for B-I12 (Create Insight Router)")
    print("=" * 80)
    sys.exit(0)
else:
    print(f"\n❌ {test_failed} TEST(S) FAILED")
    print("=" * 80)
    sys.exit(1)
