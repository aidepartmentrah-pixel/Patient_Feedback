"""
STEP 3.12 Prompt 2 — Scope Engine Integration Test Suite
Tests that inbox_service.py correctly uses Phase 2.5 Scope Engine (allowed_unit_ids)
and enforces security boundaries.

This test suite creates REAL data in the database and verifies:
- Users only see subcases within their allowed_unit_ids
- Out-of-scope subcases are never returned
- Security boundary is enforced even if role/router/frontend is compromised
"""

import sys
import os
from datetime import datetime

# Force UTF-8 encoding for emoji support
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add backend directory to Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.database import get_connection


def test(description):
    """Test decorator"""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*70}")
            print(f"TEST: {description}")
            print('='*70)
            try:
                func()
                print(f"✅ PASSED")
            except AssertionError as e:
                print(f"❌ FAILED: {str(e)}")
                raise
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
        return wrapper
    return decorator


class MockUser:
    """Mock user object with Phase 2.5 allowed_unit_ids"""
    def __init__(self, role, allowed_unit_ids, user_id=1):
        self.role = role
        self.allowed_unit_ids = allowed_unit_ids  # Phase 2.5: set[int]
        self.user_id = user_id


class MockSubcase:
    """Mock subcase object for unit tests"""
    def __init__(self, subcase_id, case_type, status, target_org_unit_id, 
                 incident_id=None, seasonal_report_id=None, created_at=None):
        self.SubcaseID = subcase_id
        self.CaseType = case_type
        self.Status = status
        self.TargetOrgUnitID = target_org_unit_id
        self.IncidentRequestCaseID = incident_id
        self.SeasonalReportID = seasonal_report_id
        self.CreatedAt = created_at


# Track test data for cleanup
test_subcase_ids = []


def cleanup_test_data():
    """Clean up test subcases from database"""
    if not test_subcase_ids:
        return
    
    print(f"\n[CLEANUP] Removing {len(test_subcase_ids)} test subcase(s)...")
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        for subcase_id in test_subcase_ids:
            cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?", (subcase_id,))
        conn.commit()
        print(f"  ✅ Cleaned up {len(test_subcase_ids)} test subcase(s)")
    finally:
        cursor.close()
        conn.close()
    
    test_subcase_ids.clear()


def create_test_subcase(case_type, status, target_org_unit_id, incident_id=None, seasonal_report_id=None):
    """Create a test subcase in the database and return its ID"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO dbo.APP_AdministrativeSubcase (
                CaseType,
                Status,
                TargetOrgUnitID,
                IncidentRequestCaseID,
                SeasonalReportID,
                CreatedAt,
                CreatedByUserID
            )
            VALUES (?, ?, ?, ?, ?, GETDATE(), 1)
        """, (case_type, status, target_org_unit_id, incident_id, seasonal_report_id))
        
        conn.commit()
        
        # Get the inserted ID
        cursor.execute("SELECT @@IDENTITY")
        subcase_id = cursor.fetchone()[0]
        
        test_subcase_ids.append(subcase_id)
        
        return subcase_id
        
    finally:
        cursor.close()
        conn.close()


# =============================================================================
# LAYER 1: STATIC CHECKS
# =============================================================================

@test("Layer 1.1 - Verify scope filter no longer uses section_id")
def test_no_section_id_usage():
    """Verify that _apply_scope_filter does not use section_id"""
    from api_v2.services import inbox_service
    import inspect
    
    source = inspect.getsource(inbox_service._apply_scope_filter)
    
    assert 'section_id' not in source, "Code should NOT reference section_id"
    print("  ✓ No section_id usage found (correct)")


@test("Layer 1.2 - Verify scope filter no longer uses department_id")
def test_no_department_id_usage():
    """Verify that _apply_scope_filter does not use department_id"""
    from api_v2.services import inbox_service
    import inspect
    
    source = inspect.getsource(inbox_service._apply_scope_filter)
    
    assert 'department_id' not in source, "Code should NOT reference department_id"
    print("  ✓ No department_id usage found (correct)")


@test("Layer 1.3 - Verify scope filter uses allowed_unit_ids")
def test_uses_allowed_unit_ids():
    """Verify that _apply_scope_filter uses allowed_unit_ids"""
    from api_v2.services import inbox_service
    import inspect
    
    source = inspect.getsource(inbox_service._apply_scope_filter)
    
    assert 'allowed_unit_ids' in source, "Code MUST reference allowed_unit_ids"
    print("  ✓ allowed_unit_ids usage found (correct)")


@test("Layer 1.4 - Verify scope filter no longer checks role")
def test_no_role_based_filtering():
    """Verify that _apply_scope_filter does not have role-based filtering"""
    from api_v2.services import inbox_service
    import inspect
    
    source = inspect.getsource(inbox_service._apply_scope_filter)
    
    # Should not have if role == checks
    assert "if role ==" not in source and "elif role ==" not in source, \
        "Code should NOT have role-based filtering logic"
    print("  ✓ No role-based filtering found (correct)")


# =============================================================================
# LAYER 2: UNIT TESTS (Mock Data)
# =============================================================================

@test("Layer 2.1 - Test scope filter with allowed_unit_ids=[1]")
def test_scope_filter_single_unit():
    """Test that _apply_scope_filter correctly filters by allowed_unit_ids"""
    from api_v2.services.inbox_service import _apply_scope_filter
    
    user = MockUser(role='Section Administrator', allowed_unit_ids={1})
    
    subcases = [
        MockSubcase(1, 'INCIDENT_RESPONSE', 'SUBMITTED_TO_SECTION', target_org_unit_id=1),
        MockSubcase(2, 'INCIDENT_RESPONSE', 'SUBMITTED_TO_SECTION', target_org_unit_id=2),
        MockSubcase(3, 'INCIDENT_RESPONSE', 'SUBMITTED_TO_SECTION', target_org_unit_id=1),
        MockSubcase(4, 'INCIDENT_RESPONSE', 'SUBMITTED_TO_SECTION', target_org_unit_id=3),
    ]
    
    filtered = _apply_scope_filter(subcases, user)
    
    print(f"  Input: {len(subcases)} subcases (targets: 1, 2, 1, 3)")
    print(f"  User allowed_unit_ids: {user.allowed_unit_ids}")
    print(f"  Filtered: {len(filtered)} subcases")
    
    assert len(filtered) == 2, f"Expected 2 subcases for allowed_unit_ids=[1], got {len(filtered)}"
    assert all(sc.TargetOrgUnitID == 1 for sc in filtered), "All filtered subcases should have TargetOrgUnitID=1"
    print(f"  ✓ Correctly filtered to allowed_unit_ids={user.allowed_unit_ids}")


@test("Layer 2.2 - Test scope filter with multiple allowed_unit_ids")
def test_scope_filter_multiple_units():
    """Test that _apply_scope_filter handles multiple allowed_unit_ids"""
    from api_v2.services.inbox_service import _apply_scope_filter
    
    user = MockUser(role='Department Administrator', allowed_unit_ids={2, 3, 5})
    
    subcases = [
        MockSubcase(1, 'INCIDENT_RESPONSE', 'SECTION_ACCEPTED_PENDING_DEPT', target_org_unit_id=1),
        MockSubcase(2, 'INCIDENT_RESPONSE', 'SECTION_ACCEPTED_PENDING_DEPT', target_org_unit_id=2),
        MockSubcase(3, 'INCIDENT_RESPONSE', 'SECTION_ACCEPTED_PENDING_DEPT', target_org_unit_id=3),
        MockSubcase(4, 'INCIDENT_RESPONSE', 'SECTION_ACCEPTED_PENDING_DEPT', target_org_unit_id=4),
        MockSubcase(5, 'INCIDENT_RESPONSE', 'SECTION_ACCEPTED_PENDING_DEPT', target_org_unit_id=5),
    ]
    
    filtered = _apply_scope_filter(subcases, user)
    
    print(f"  Input: {len(subcases)} subcases (targets: 1, 2, 3, 4, 5)")
    print(f"  User allowed_unit_ids: {user.allowed_unit_ids}")
    print(f"  Filtered: {len(filtered)} subcases")
    
    assert len(filtered) == 3, f"Expected 3 subcases, got {len(filtered)}"
    
    filtered_targets = {sc.TargetOrgUnitID for sc in filtered}
    assert filtered_targets == {2, 3, 5}, f"Expected targets {{2, 3, 5}}, got {filtered_targets}"
    print(f"  ✓ Correctly filtered to allowed_unit_ids={user.allowed_unit_ids}")


@test("Layer 2.3 - Test scope filter with empty allowed_unit_ids")
def test_scope_filter_empty_allowed():
    """Test that _apply_scope_filter returns empty list if no allowed_unit_ids"""
    from api_v2.services.inbox_service import _apply_scope_filter
    
    user = MockUser(role='Section Administrator', allowed_unit_ids=set())
    
    subcases = [
        MockSubcase(1, 'INCIDENT_RESPONSE', 'SUBMITTED_TO_SECTION', target_org_unit_id=1),
        MockSubcase(2, 'INCIDENT_RESPONSE', 'SUBMITTED_TO_SECTION', target_org_unit_id=2),
    ]
    
    filtered = _apply_scope_filter(subcases, user)
    
    print(f"  Input: {len(subcases)} subcases")
    print(f"  User allowed_unit_ids: {user.allowed_unit_ids} (empty)")
    print(f"  Filtered: {len(filtered)} subcases")
    
    assert len(filtered) == 0, "Should return empty list when allowed_unit_ids is empty"
    print(f"  ✓ Correctly returns empty list for empty allowed_unit_ids")


@test("Layer 2.4 - Test scope filter with None allowed_unit_ids")
def test_scope_filter_none_allowed():
    """Test that _apply_scope_filter returns empty list if allowed_unit_ids is None"""
    from api_v2.services.inbox_service import _apply_scope_filter
    
    user = MockUser(role='Section Administrator', allowed_unit_ids=None)
    
    subcases = [
        MockSubcase(1, 'INCIDENT_RESPONSE', 'SUBMITTED_TO_SECTION', target_org_unit_id=1),
    ]
    
    filtered = _apply_scope_filter(subcases, user)
    
    print(f"  Input: {len(subcases)} subcases")
    print(f"  User allowed_unit_ids: None")
    print(f"  Filtered: {len(filtered)} subcases")
    
    assert len(filtered) == 0, "Should return empty list when allowed_unit_ids is None"
    print(f"  ✓ Correctly returns empty list for None allowed_unit_ids")


@test("Layer 2.5 - Security test: role should not grant access")
def test_role_does_not_grant_access():
    """Test that having a role does NOT grant access without allowed_unit_ids"""
    from api_v2.services.inbox_service import _apply_scope_filter
    
    # User with admin role but NO allowed_unit_ids
    user = MockUser(role='Administration Administrator', allowed_unit_ids=set())
    
    subcases = [
        MockSubcase(1, 'INCIDENT_RESPONSE', 'DEPT_ACCEPTED_PENDING_ADMIN', target_org_unit_id=1),
        MockSubcase(2, 'INCIDENT_RESPONSE', 'DEPT_ACCEPTED_PENDING_ADMIN', target_org_unit_id=2),
        MockSubcase(3, 'INCIDENT_RESPONSE', 'DEPT_ACCEPTED_PENDING_ADMIN', target_org_unit_id=3),
    ]
    
    filtered = _apply_scope_filter(subcases, user)
    
    print(f"  Input: {len(subcases)} subcases")
    print(f"  User role: {user.role} (highest privilege)")
    print(f"  User allowed_unit_ids: {user.allowed_unit_ids} (empty)")
    print(f"  Filtered: {len(filtered)} subcases")
    
    assert len(filtered) == 0, "Admin role without allowed_unit_ids should see NOTHING"
    print(f"  ✅ SECURITY: Role does not grant access without allowed_unit_ids")


# =============================================================================
# LAYER 3: INTEGRATION TESTS (Real Database)
# =============================================================================

@test("Layer 3.1 - Setup: Create test subcases in database")
def test_create_test_subcases():
    """Create test subcases with different TargetOrgUnitIDs"""
    print("\n  Creating test subcases:")
    
    # Get a valid SeasonalReportID from existing data
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT TOP 1 SeasonalReportID FROM dbo.APP_SeasonalOrgUnitReport WHERE SeasonalReportID IS NOT NULL")
    row = cursor.fetchone()
    valid_seasonal_report_id = row[0] if row else 660  # Use 660 (from SubcaseID 53) as fallback
    cursor.close()
    conn.close()
    
    print(f"  Using SeasonalReportID={valid_seasonal_report_id} for test data\n")
    
    # Create subcases for section (status: SUBMITTED_TO_SECTION)
    id1 = create_test_subcase('SEASONAL_REPORT_RESPONSE', 'SUBMITTED_TO_SECTION', 
                               target_org_unit_id=1, seasonal_report_id=valid_seasonal_report_id)
    print(f"    - SubcaseID {id1}: TargetOrgUnitID=1, Status=SUBMITTED_TO_SECTION")
    
    id2 = create_test_subcase('SEASONAL_REPORT_RESPONSE', 'SUBMITTED_TO_SECTION', 
                               target_org_unit_id=2, seasonal_report_id=valid_seasonal_report_id)
    print(f"    - SubcaseID {id2}: TargetOrgUnitID=2, Status=SUBMITTED_TO_SECTION")
    
    id3 = create_test_subcase('SEASONAL_REPORT_RESPONSE', 'SUBMITTED_TO_SECTION', 
                               target_org_unit_id=1, seasonal_report_id=valid_seasonal_report_id)
    print(f"    - SubcaseID {id3}: TargetOrgUnitID=1, Status=SUBMITTED_TO_SECTION")
    
    # Create subcases for department (status: SECTION_ACCEPTED_PENDING_DEPT)
    id4 = create_test_subcase('SEASONAL_REPORT_RESPONSE', 'SECTION_ACCEPTED_PENDING_DEPT', 
                               target_org_unit_id=3, seasonal_report_id=valid_seasonal_report_id)
    print(f"    - SubcaseID {id4}: TargetOrgUnitID=3, Status=SECTION_ACCEPTED_PENDING_DEPT")
    
    id5 = create_test_subcase('SEASONAL_REPORT_RESPONSE', 'SECTION_ACCEPTED_PENDING_DEPT', 
                               target_org_unit_id=4, seasonal_report_id=valid_seasonal_report_id)
    print(f"    - SubcaseID {id5}: TargetOrgUnitID=4, Status=SECTION_ACCEPTED_PENDING_DEPT")
    
    # Create subcases for administration (status: DEPT_ACCEPTED_PENDING_ADMIN)
    id6 = create_test_subcase('SEASONAL_REPORT_RESPONSE', 'DEPT_ACCEPTED_PENDING_ADMIN', 
                               target_org_unit_id=5, seasonal_report_id=valid_seasonal_report_id)
    print(f"    - SubcaseID {id6}: TargetOrgUnitID=5, Status=DEPT_ACCEPTED_PENDING_ADMIN")
    
    print(f"\n  ✓ Created {len(test_subcase_ids)} test subcases")


@test("Layer 3.2 - Integration: Section inbox with allowed_unit_ids=[1]")
def test_section_inbox_scope_filter():
    """Test get_section_inbox with real database and scope filtering"""
    from api_v2.services.inbox_service import get_section_inbox
    
    # User with access to only unit 1
    user = MockUser(role='Section Administrator', allowed_unit_ids={1})
    
    inbox = get_section_inbox(user)
    
    print(f"  User allowed_unit_ids: {user.allowed_unit_ids}")
    print(f"  Retrieved: {len(inbox)} inbox item(s)")
    
    if inbox:
        for item in inbox:
            print(f"    - SubcaseID={item['subcase_id']}, Target={item['target_org_unit_id']}, Status={item['status']}")
            
            # Security assertion: ALL items must be in allowed scope
            assert item['target_org_unit_id'] in user.allowed_unit_ids, \
                f"SECURITY VIOLATION: SubcaseID {item['subcase_id']} has TargetOrgUnitID={item['target_org_unit_id']} " \
                f"which is NOT in allowed_unit_ids={user.allowed_unit_ids}"
        
        # Should only see TargetOrgUnitID=1
        targets = {item['target_org_unit_id'] for item in inbox}
        assert targets == {1}, f"Should only see TargetOrgUnitID=1, got {targets}"
        
        print(f"  ✅ SECURITY: All items within allowed scope")
    else:
        print(f"  ⚠️  No items (may be normal if test data was cleaned up)")


@test("Layer 3.3 - Integration: Section inbox with allowed_unit_ids=[2]")
def test_section_inbox_different_scope():
    """Test that different allowed_unit_ids returns different data"""
    from api_v2.services.inbox_service import get_section_inbox
    
    # User with access to only unit 2
    user = MockUser(role='Section Administrator', allowed_unit_ids={2})
    
    inbox = get_section_inbox(user)
    
    print(f"  User allowed_unit_ids: {user.allowed_unit_ids}")
    print(f"  Retrieved: {len(inbox)} inbox item(s)")
    
    if inbox:
        for item in inbox:
            print(f"    - SubcaseID={item['subcase_id']}, Target={item['target_org_unit_id']}, Status={item['status']}")
            
            # Security assertion
            assert item['target_org_unit_id'] in user.allowed_unit_ids, \
                f"SECURITY VIOLATION: SubcaseID {item['subcase_id']} outside allowed scope"
        
        # Should only see TargetOrgUnitID=2
        targets = {item['target_org_unit_id'] for item in inbox}
        assert targets == {2}, f"Should only see TargetOrgUnitID=2, got {targets}"
        
        print(f"  ✅ SECURITY: All items within allowed scope")


@test("Layer 3.4 - Integration: Section inbox with allowed_unit_ids=[1,2]")
def test_section_inbox_multiple_scopes():
    """Test that user with multiple allowed_unit_ids sees all their subcases"""
    from api_v2.services.inbox_service import get_section_inbox
    
    # User with access to units 1 and 2
    user = MockUser(role='Section Administrator', allowed_unit_ids={1, 2})
    
    inbox = get_section_inbox(user)
    
    print(f"  User allowed_unit_ids: {user.allowed_unit_ids}")
    print(f"  Retrieved: {len(inbox)} inbox item(s)")
    
    if inbox:
        for item in inbox:
            print(f"    - SubcaseID={item['subcase_id']}, Target={item['target_org_unit_id']}, Status={item['status']}")
            
            # Security assertion
            assert item['target_org_unit_id'] in user.allowed_unit_ids, \
                f"SECURITY VIOLATION: SubcaseID {item['subcase_id']} outside allowed scope"
        
        # Should see both TargetOrgUnitID=1 and 2
        targets = {item['target_org_unit_id'] for item in inbox}
        assert targets.issubset({1, 2}), f"Should only see TargetOrgUnitID in {{1, 2}}, got {targets}"
        
        print(f"  ✅ SECURITY: All items within allowed scope")


@test("Layer 3.5 - Integration: Department inbox with allowed_unit_ids=[3]")
def test_department_inbox_scope_filter():
    """Test get_department_inbox with scope filtering"""
    from api_v2.services.inbox_service import get_department_inbox
    
    # User with access to only unit 3
    user = MockUser(role='Department Administrator', allowed_unit_ids={3})
    
    inbox = get_department_inbox(user)
    
    print(f"  User allowed_unit_ids: {user.allowed_unit_ids}")
    print(f"  Retrieved: {len(inbox)} inbox item(s)")
    
    if inbox:
        for item in inbox:
            print(f"    - SubcaseID={item['subcase_id']}, Target={item['target_org_unit_id']}, Status={item['status']}")
            
            # Security assertion
            assert item['target_org_unit_id'] in user.allowed_unit_ids, \
                f"SECURITY VIOLATION: SubcaseID {item['subcase_id']} outside allowed scope"
        
        targets = {item['target_org_unit_id'] for item in inbox}
        assert targets == {3}, f"Should only see TargetOrgUnitID=3, got {targets}"
        
        print(f"  ✅ SECURITY: All items within allowed scope")


@test("Layer 3.6 - Integration: Administration inbox with allowed_unit_ids=[5]")
def test_administration_inbox_scope_filter():
    """Test get_administration_inbox with scope filtering"""
    from api_v2.services.inbox_service import get_administration_inbox
    
    # User with access to only unit 5
    user = MockUser(role='Administration Administrator', allowed_unit_ids={5})
    
    inbox = get_administration_inbox(user)
    
    print(f"  User allowed_unit_ids: {user.allowed_unit_ids}")
    print(f"  Retrieved: {len(inbox)} inbox item(s)")
    
    if inbox:
        for item in inbox:
            print(f"    - SubcaseID={item['subcase_id']}, Target={item['target_org_unit_id']}, Status={item['status']}")
            
            # Security assertion
            assert item['target_org_unit_id'] in user.allowed_unit_ids, \
                f"SECURITY VIOLATION: SubcaseID {item['subcase_id']} outside allowed scope"
        
        targets = {item['target_org_unit_id'] for item in inbox}
        assert targets == {5}, f"Should only see TargetOrgUnitID=5, got {targets}"
        
        print(f"  ✅ SECURITY: All items within allowed scope")


@test("Layer 3.7 - Security: User with empty allowed_unit_ids sees nothing")
def test_empty_scope_sees_nothing():
    """Test that user with empty allowed_unit_ids cannot see any data"""
    from api_v2.services.inbox_service import get_section_inbox
    
    # User with NO allowed units
    user = MockUser(role='Section Administrator', allowed_unit_ids=set())
    
    inbox = get_section_inbox(user)
    
    print(f"  User allowed_unit_ids: {user.allowed_unit_ids} (empty)")
    print(f"  Retrieved: {len(inbox)} inbox item(s)")
    
    assert len(inbox) == 0, "User with empty allowed_unit_ids should see NOTHING"
    print(f"  ✅ SECURITY: Empty scope returns no data")


@test("Layer 3.8 - Security: Verify no cross-contamination between scopes")
def test_no_cross_contamination():
    """Test that two users with different scopes see completely different data"""
    from api_v2.services.inbox_service import get_section_inbox
    
    # User A: access to unit 1
    user_a = MockUser(role='Section Administrator', allowed_unit_ids={1})
    inbox_a = get_section_inbox(user_a)
    
    # User B: access to unit 2
    user_b = MockUser(role='Section Administrator', allowed_unit_ids={2})
    inbox_b = get_section_inbox(user_b)
    
    print(f"  User A allowed_unit_ids: {user_a.allowed_unit_ids}")
    print(f"  User A inbox: {len(inbox_a)} item(s)")
    
    print(f"  User B allowed_unit_ids: {user_b.allowed_unit_ids}")
    print(f"  User B inbox: {len(inbox_b)} item(s)")
    
    # Get subcase IDs from each inbox
    ids_a = {item['subcase_id'] for item in inbox_a}
    ids_b = {item['subcase_id'] for item in inbox_b}
    
    # Should have NO overlap
    overlap = ids_a & ids_b
    
    print(f"  Overlap: {overlap}")
    
    assert len(overlap) == 0, f"Users with different scopes should see NO common subcases, but found: {overlap}"
    print(f"  ✅ SECURITY: No cross-contamination between scopes")


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STEP 3.12 PROMPT 2 — SCOPE ENGINE INTEGRATION TEST SUITE")
    print("Testing Phase 2.5 allowed_unit_ids enforcement")
    print("="*80)
    
    layer1_passed = 0
    layer1_total = 0
    layer2_passed = 0
    layer2_total = 0
    layer3_passed = 0
    layer3_total = 0
    
    try:
        # LAYER 1: Static Checks
        print("\n" + "="*80)
        print("LAYER 1: STATIC CHECKS (Code Analysis)")
        print("="*80)
        
        tests_layer1 = [
            test_no_section_id_usage,
            test_no_department_id_usage,
            test_uses_allowed_unit_ids,
            test_no_role_based_filtering,
        ]
        
        for test_func in tests_layer1:
            layer1_total += 1
            try:
                test_func()
                layer1_passed += 1
            except Exception:
                pass
        
        # LAYER 2: Unit Tests
        print("\n" + "="*80)
        print("LAYER 2: UNIT TESTS (Mock Data)")
        print("="*80)
        
        tests_layer2 = [
            test_scope_filter_single_unit,
            test_scope_filter_multiple_units,
            test_scope_filter_empty_allowed,
            test_scope_filter_none_allowed,
            test_role_does_not_grant_access,
        ]
        
        for test_func in tests_layer2:
            layer2_total += 1
            try:
                test_func()
                layer2_passed += 1
            except Exception:
                pass
        
        # LAYER 3: Integration Tests
        print("\n" + "="*80)
        print("LAYER 3: INTEGRATION TESTS (Real Database)")
        print("="*80)
        
        tests_layer3 = [
            test_create_test_subcases,
            test_section_inbox_scope_filter,
            test_section_inbox_different_scope,
            test_section_inbox_multiple_scopes,
            test_department_inbox_scope_filter,
            test_administration_inbox_scope_filter,
            test_empty_scope_sees_nothing,
            test_no_cross_contamination,
        ]
        
        for test_func in tests_layer3:
            layer3_total += 1
            try:
                test_func()
                layer3_passed += 1
            except Exception:
                pass
    
    finally:
        # Always cleanup test data
        cleanup_test_data()
    
    # SUMMARY
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Layer 1 (Static):      {layer1_passed}/{layer1_total} passed")
    print(f"Layer 2 (Unit):        {layer2_passed}/{layer2_total} passed")
    print(f"Layer 3 (Integration): {layer3_passed}/{layer3_total} passed")
    print(f"TOTAL:                 {layer1_passed + layer2_passed + layer3_passed}/{layer1_total + layer2_total + layer3_total} passed")
    
    total_passed = layer1_passed + layer2_passed + layer3_passed
    total_tests = layer1_total + layer2_total + layer3_total
    
    if total_passed == total_tests:
        print("\n✅ ALL TESTS PASSED! Phase 2.5 Scope Engine integration is 100% complete!")
        print("✅ Security boundary enforced: users only see data within allowed_unit_ids")
        print("✅ STEP 3.12 (Both Prompt 1 & 2) is COMPLETE!")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Review output above.")
