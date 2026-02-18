"""
Test grouped inbox service layer - Stage 2 Testing

This test file validates the service layer grouping logic
that organizes subcases by organizational unit with supervisor info.

Tests:
1. Verify grouped structure and sorting
2. Verify empty groups are excluded
3. Verify scope filtering works
"""

import sys
sys.path.insert(0, 'backend')

from core.database import get_connection


def test_grouped_inbox_structure():
    """Verify service returns properly grouped structure"""
    from backend.api_v2.services import insight_service
    
    print("\n" + "="*60)
    print("TEST: Grouped inbox structure and sorting")
    print("="*60)
    
    # Get a real admin user for testing
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT TOP 1 u.UserID, u.Username, r.RoleCode
        FROM dbo.APP_Users u
        JOIN dbo.APP_UserRoleScope scope ON u.UserID = scope.UserID
        JOIN dbo.APP_Roles r ON scope.RoleID = r.RoleID
        WHERE r.RoleCode IN ('SECTION_ADMIN', 'DEPARTMENT_ADMIN', 'ADMINISTRATION_ADMIN')
          AND u.IsActive = 1
        ORDER BY u.UserID
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not row:
        print("⚠️  No admin user found for testing")
        return
    
    # Mock user with allowed_unit_ids 
    # For testing, we'll allow access to all org units that have subcases
    class MockScope:
        def __init__(self, role_code):
            self.role_code = role_code
    
    class MockUser:
        def __init__(self, role_code):
            self.scopes = [MockScope(role_code)]
            # Allow access to first 20 org units (simplified for test)
            self.allowed_unit_ids = set(range(1, 21))
    
    user = MockUser(row.RoleCode)
    groups = insight_service.get_grouped_inbox_for_admin(user)
    
    print(f"✅ Found {len(groups)} groups")
    
    if len(groups) > 0:
        first_group = groups[0]
        
        # Verify structure
        assert 'section_id' in first_group, "Missing section_id"
        assert 'section_name' in first_group, "Missing section_name"
        assert 'supervisor_name' in first_group, "Missing supervisor_name"
        assert 'pending_count' in first_group, "Missing pending_count"
        assert 'subcases' in first_group, "Missing subcases"
        assert isinstance(first_group['subcases'], list), "subcases must be a list"
        
        print(f"✅ Group structure valid")
        print(f"   Section: {first_group['section_name']}")
        print(f"   Supervisor: {first_group['supervisor_name']}")
        print(f"   Pending: {first_group['pending_count']}")
        print(f"   Org Type: {first_group.get('org_type', 'N/A')}")
        
        # Verify sorting (groups by pending_count DESC)
        if len(groups) > 1:
            first_count = groups[0]['pending_count']
            second_count = groups[1]['pending_count']
            assert first_count >= second_count, \
                f"Groups not sorted by pending_count: {first_count} < {second_count}"
            print(f"✅ Groups sorted by pending count DESC")
            print(f"   Top group: {first_count} pending")
            print(f"   2nd group: {second_count} pending")
        
        # Verify subcases sorting (by waiting_days DESC)
        if len(first_group['subcases']) > 1:
            first_subcase = first_group['subcases'][0]
            second_subcase = first_group['subcases'][1]
            first_days = first_subcase['waiting_days']
            second_days = second_subcase['waiting_days']
            assert first_days >= second_days, \
                f"Subcases not sorted by waiting_days: {first_days} < {second_days}"
            print(f"✅ Subcases sorted by waiting days DESC")
            print(f"   Oldest: {first_days} days")
            print(f"   Next: {second_days} days")
        
        # Display sample subcase details
        if first_group['subcases']:
            sample = first_group['subcases'][0]
            print(f"\n📋 Sample subcase from top group:")
            print(f"   Subcase ID: {sample.get('subcase_id')}")
            print(f"   Type: {sample.get('case_type')}")
            print(f"   Status: {sample.get('status')}")
            print(f"   Waiting: {sample.get('waiting_days')} days")
            if sample.get('patient_name'):
                print(f"   Patient: {sample.get('patient_name')}")
            if sample.get('season_name'):
                print(f"   Season: {sample.get('season_name')}")
            if sample.get('severity'):
                print(f"   Severity: {sample.get('severity')}")
    else:
        print("⚠️  No groups found (may need test data or wider scope)")


def test_empty_groups_excluded():
    """Verify groups with 0 pending are excluded"""
    from backend.api_v2.services import insight_service
    
    print("\n" + "="*60)
    print("TEST: Empty groups excluded")
    print("="*60)
    
    class MockScope:
        def __init__(self):
            self.role_code = 'SECTION_ADMIN'
    
    class MockUser:
        def __init__(self):
            self.scopes = [MockScope()]
            self.allowed_unit_ids = set(range(1, 21))
    
    user = MockUser()
    groups = insight_service.get_grouped_inbox_for_admin(user)
    
    # All groups must have pending_count > 0
    empty_groups = [g for g in groups if g['pending_count'] <= 0]
    
    if empty_groups:
        print(f"❌ Found {len(empty_groups)} empty groups:")
        for g in empty_groups:
            print(f"   - {g['section_name']}: {g['pending_count']} pending")
        assert False, "Empty groups should be excluded"
    else:
        print(f"✅ All {len(groups)} groups have pending items")
        if groups:
            min_count = min(g['pending_count'] for g in groups)
            max_count = max(g['pending_count'] for g in groups)
            print(f"   Range: {min_count} to {max_count} pending items")


def test_scope_filtering():
    """Verify scope filtering works correctly"""
    from backend.api_v2.services import insight_service
    
    print("\n" + "="*60)
    print("TEST: Scope filtering")
    print("="*60)
    
    class MockScope:
        def __init__(self):
            self.role_code = 'SECTION_ADMIN'
    
    # Test 1: Very limited scope
    class LimitedUser:
        def __init__(self):
            self.scopes = [MockScope()]
            # Only allow access to org units 1 and 2
            self.allowed_unit_ids = set([1, 2])
    
    limited_user = LimitedUser()
    limited_groups = insight_service.get_grouped_inbox_for_admin(limited_user)
    
    print(f"Limited scope (units 1-2): {len(limited_groups)} groups")
    
    # Verify all returned groups have section_id in allowed_unit_ids
    for group in limited_groups:
        assert group['section_id'] in limited_user.allowed_unit_ids, \
            f"Group {group['section_id']} outside allowed scope"
    
    print(f"✅ All groups within allowed scope")
    
    # Test 2: Broader scope
    class BroadUser:
        def __init__(self):
            self.scopes = [MockScope()]
            self.allowed_unit_ids = set(range(1, 50))
    
    broad_user = BroadUser()
    broad_groups = insight_service.get_grouped_inbox_for_admin(broad_user)
    
    print(f"Broad scope (units 1-49): {len(broad_groups)} groups")
    
    # Broader scope should have >= groups than limited scope
    assert len(broad_groups) >= len(limited_groups), \
        "Broader scope should return more or equal groups"
    
    print(f"✅ Scope filtering working correctly")
    print(f"   Limited scope: {len(limited_groups)} groups")
    print(f"   Broad scope: {len(broad_groups)} groups")


def test_different_roles():
    """Test that different roles see appropriate subcases"""
    from backend.api_v2.services import insight_service
    
    print("\n" + "="*60)
    print("TEST: Different admin roles")
    print("="*60)
    
    class MockScope:
        def __init__(self, role_code):
            self.role_code = role_code
    
    class MockUser:
        def __init__(self, role_code):
            self.scopes = [MockScope(role_code)]
            self.allowed_unit_ids = set(range(1, 30))
    
    # Test each role
    roles = ['SECTION_ADMIN', 'DEPARTMENT_ADMIN', 'ADMINISTRATION_ADMIN']
    
    for role in roles:
        user = MockUser(role)
        groups = insight_service.get_grouped_inbox_for_admin(user)
        print(f"\n{role}:")
        print(f"  Groups: {len(groups)}")
        
        if groups:
            total_subcases = sum(g['pending_count'] for g in groups)
            print(f"  Total subcases: {total_subcases}")
            
            # Check statuses
            if groups[0]['subcases']:
                sample_status = groups[0]['subcases'][0].get('status')
                print(f"  Sample status: {sample_status}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" STAGE 2: SERVICE LAYER TESTING - Grouped Inbox")
    print("="*70)
    
    try:
        test_grouped_inbox_structure()
        test_empty_groups_excluded()
        test_scope_filtering()
        test_different_roles()
        
        print("\n" + "="*70)
        print("✅ ALL STAGE 2 TESTS COMPLETED")
        print("="*70)
        print("\nNext Steps:")
        print("  1. Review the grouped structure output above")
        print("  2. Verify sorting is correct (busiest groups first)")
        print("  3. Check supervisor names are properly resolved")
        print("  4. Proceed to Stage 3: Router/API endpoint implementation")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
