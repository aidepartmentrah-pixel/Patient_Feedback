"""
Test File: Inbox Archive Feature
=================================

Tests the new archive functionality that allows users to view
subcases that have moved past their workflow stage.

Archive is READ-ONLY - only 'view' action is allowed.

Tests:
1. Archive endpoint returns correct structure
2. Archive items are view-only (allowed_actions = ['view'])
3. Each role sees appropriate archive statuses
4. Scope filtering is applied to archive
5. Empty scopes return empty archive
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from typing import Set


# =============================================================================
# MOCK USER CLASS
# =============================================================================

class MockScope:
    def __init__(self, role_code: str, org_unit_id: int = None):
        self.role_code = role_code
        self.org_unit_id = org_unit_id


class MockUser:
    def __init__(self, role: str, allowed_unit_ids: Set[int] = None):
        self.scopes = [MockScope(role)]
        self.allowed_unit_ids = allowed_unit_ids or set()
        self.user_id = 1


# =============================================================================
# TEST HELPERS
# =============================================================================

def print_header(text: str):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def print_test(name: str, passed: bool, details: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


def print_section(name: str):
    print(f"\n--- {name} ---")


# =============================================================================
# TESTS
# =============================================================================

def test_archive_service_functions_exist():
    """Test 1: Verify all archive service functions exist"""
    print_section("Test 1: Archive Service Functions Exist")
    
    from api_v2.services import inbox_service
    
    functions = [
        'get_archive',
        'get_section_archive',
        'get_department_archive',
        'get_administration_archive',
        'get_complaint_supervisor_archive',
        'get_worker_archive',
        '_build_archive_item'
    ]
    
    all_exist = True
    for func_name in functions:
        exists = hasattr(inbox_service, func_name)
        print_test(f"Function: {func_name}", exists)
        if not exists:
            all_exist = False
    
    return all_exist


def test_archive_db_functions_exist():
    """Test 2: Verify all archive DB functions exist"""
    print_section("Test 2: Archive DB Functions Exist")
    
    from api_v2.db_layer import administrative_subcase_db
    
    functions = [
        'get_subcases_by_statuses',
        'get_subcases_archived_for_section',
        'get_subcases_archived_for_department',
        'get_subcases_archived_for_administration',
        'get_subcases_archived_for_complaint_supervisor'
    ]
    
    all_exist = True
    for func_name in functions:
        exists = hasattr(administrative_subcase_db, func_name)
        print_test(f"Function: {func_name}", exists)
        if not exists:
            all_exist = False
    
    return all_exist


def test_archive_items_are_view_only():
    """Test 3: Verify archive items only have 'view' action"""
    print_section("Test 3: Archive Items Are View-Only")
    
    from api_v2.services.inbox_service import _build_archive_item
    
    # Mock subcase data
    mock_subcase = {
        'subcase_id': 123,
        'case_type': 'INCIDENT_RESPONSE',
        'incident_request_case_id': 456,
        'seasonal_report_id': None,
        'target_org_unit_id': 10,
        'org_unit_name': 'Test Section',
        'status': 'ADMIN_APPROVED',
        'created_at': '2026-01-15 10:00:00',
        'updated_at': '2026-01-20 14:30:00'
    }
    
    archive_item = _build_archive_item(mock_subcase)
    
    # Check allowed_actions is exactly ['view']
    is_view_only = archive_item.get('allowed_actions') == ['view']
    print_test("Archive item allowed_actions == ['view']", is_view_only,
               f"Got: {archive_item.get('allowed_actions')}")
    
    # Check all expected fields are present
    expected_fields = [
        'subcase_id', 'case_type', 'incident_id', 'seasonal_report_id',
        'target_org_unit_id', 'target_org_unit_name', 'status',
        'created_at', 'updated_at', 'allowed_actions'
    ]
    
    all_fields_present = all(field in archive_item for field in expected_fields)
    print_test("All expected fields present", all_fields_present,
               f"Fields: {list(archive_item.keys())}")
    
    return is_view_only and all_fields_present


def test_get_archive_routing():
    """Test 4: Verify get_archive routes to correct role-specific functions"""
    print_section("Test 4: get_archive Role Routing")
    
    from api_v2.services.inbox_service import get_archive
    
    roles_to_test = [
        ('SECTION_ADMIN', True),
        ('DEPARTMENT_ADMIN', True),
        ('ADMINISTRATION_ADMIN', True),
        ('COMPLAINT_SUPERVISOR', True),
        ('WORKER', True),
        ('SOFTWARE_ADMIN', False),  # Should return empty
    ]
    
    all_pass = True
    for role, should_call_function in roles_to_test:
        user = MockUser(role, allowed_unit_ids={1, 2, 3})
        
        try:
            archive = get_archive(user)
            is_list = isinstance(archive, list)
            print_test(f"Role {role} returns list", is_list,
                       f"Type: {type(archive).__name__}, Length: {len(archive)}")
            if not is_list:
                all_pass = False
        except Exception as e:
            print_test(f"Role {role} returns list", False, f"Exception: {str(e)}")
            all_pass = False
    
    return all_pass


def test_empty_scopes_return_empty_archive():
    """Test 5: Verify users with no scopes get empty archive"""
    print_section("Test 5: Empty Scopes Return Empty Archive")
    
    from api_v2.services.inbox_service import get_archive
    
    # User with no scopes
    user_no_scopes = type('obj', (object,), {'scopes': None, 'allowed_unit_ids': set()})()
    
    archive = get_archive(user_no_scopes)
    is_empty = archive == []
    print_test("No scopes returns empty archive", is_empty, f"Got: {archive}")
    
    # User with empty scopes list
    user_empty_scopes = type('obj', (object,), {'scopes': [], 'allowed_unit_ids': set()})()
    
    archive2 = get_archive(user_empty_scopes)
    is_empty2 = archive2 == []
    print_test("Empty scopes list returns empty archive", is_empty2, f"Got: {archive2}")
    
    return is_empty and is_empty2


def test_archive_endpoint_exists():
    """Test 6: Verify archive endpoint exists in router"""
    print_section("Test 6: Archive Endpoint Exists in Router")
    
    from api_v2.routers import workflow_router as router_instance
    
    # Check router routes - routes include full prefix path
    routes = [route.path for route in router_instance.routes]
    
    # Check that both inbox and inbox/archive routes are registered
    has_inbox_route = any('inbox' in r and 'archive' not in r for r in routes)
    has_archive_route = any('inbox/archive' in r for r in routes)
    
    inbox_routes = [r for r in routes if 'inbox' in r]
    print_test("/inbox route registered", has_inbox_route,
               f"Inbox routes: {inbox_routes}")
    print_test("/inbox/archive route registered", has_archive_route,
               f"Archive route present")
    
    return has_inbox_route and has_archive_route


def test_section_archive_with_real_data():
    """Test 7: Test section archive with real database (integration test)"""
    print_section("Test 7: Section Archive Integration Test")
    
    from api_v2.services.inbox_service import get_section_archive
    
    # Create a user with section admin role and broad scope
    user = MockUser('SECTION_ADMIN', allowed_unit_ids=set(range(1, 300)))
    
    try:
        archive = get_section_archive(user)
        
        is_list = isinstance(archive, list)
        print_test("Returns list", is_list)
        
        if is_list and len(archive) > 0:
            first_item = archive[0]
            
            # Verify structure
            has_subcase_id = 'subcase_id' in first_item
            has_status = 'status' in first_item
            has_view_action = first_item.get('allowed_actions') == ['view']
            
            print_test("First item has subcase_id", has_subcase_id)
            print_test("First item has status", has_status,
                       f"Status: {first_item.get('status')}")
            print_test("First item is view-only", has_view_action,
                       f"Actions: {first_item.get('allowed_actions')}")
            
            # Verify status is one that section would archive
            valid_section_archive_statuses = [
                'SECTION_ACCEPTED_PENDING_DEPT',
                'RETURNED_TO_DEPT_FOR_REVISION',
                'DEPT_ACCEPTED_PENDING_ADMIN',
                'ADMIN_APPROVED',
                'SECTION_DENIED',
                'FORCE_CLOSED'
            ]
            status_valid = first_item.get('status') in valid_section_archive_statuses
            print_test("Status is valid section archive status", status_valid,
                       f"Status: {first_item.get('status')}")
            
            print(f"\n  Archive contains {len(archive)} item(s)")
            
            return has_subcase_id and has_status and has_view_action and status_valid
        else:
            print(f"  Archive contains {len(archive)} items (may be normal if no data)")
            return True  # Empty archive is valid
            
    except Exception as e:
        print_test("Section archive query", False, f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_department_archive_with_real_data():
    """Test 8: Test department archive with real database"""
    print_section("Test 8: Department Archive Integration Test")
    
    from api_v2.services.inbox_service import get_department_archive
    
    user = MockUser('DEPARTMENT_ADMIN', allowed_unit_ids=set(range(1, 300)))
    
    try:
        archive = get_department_archive(user)
        
        is_list = isinstance(archive, list)
        print_test("Returns list", is_list)
        
        if is_list and len(archive) > 0:
            first_item = archive[0]
            
            valid_dept_archive_statuses = [
                'DEPT_ACCEPTED_PENDING_ADMIN',
                'ADMIN_APPROVED',
                'RETURNED_TO_SECTION_FOR_REVISION',
                'FORCE_CLOSED'
            ]
            status_valid = first_item.get('status') in valid_dept_archive_statuses
            print_test("Status is valid department archive status", status_valid,
                       f"Status: {first_item.get('status')}")
            
            print(f"\n  Archive contains {len(archive)} item(s)")
            return status_valid
        else:
            print(f"  Archive contains {len(archive)} items (may be normal if no data)")
            return True
            
    except Exception as e:
        print_test("Department archive query", False, f"Exception: {str(e)}")
        return False


def test_administration_archive_with_real_data():
    """Test 9: Test administration archive with real database"""
    print_section("Test 9: Administration Archive Integration Test")
    
    from api_v2.services.inbox_service import get_administration_archive
    
    user = MockUser('ADMINISTRATION_ADMIN', allowed_unit_ids=set(range(1, 300)))
    
    try:
        archive = get_administration_archive(user)
        
        is_list = isinstance(archive, list)
        print_test("Returns list", is_list)
        
        if is_list and len(archive) > 0:
            first_item = archive[0]
            
            valid_admin_archive_statuses = [
                'ADMIN_APPROVED',
                'FORCE_CLOSED',
                'RETURNED_TO_DEPT_FOR_REVISION'
            ]
            status_valid = first_item.get('status') in valid_admin_archive_statuses
            print_test("Status is valid admin archive status", status_valid,
                       f"Status: {first_item.get('status')}")
            
            print(f"\n  Archive contains {len(archive)} item(s)")
            return status_valid
        else:
            print(f"  Archive contains {len(archive)} items (may be normal if no data)")
            return True
            
    except Exception as e:
        print_test("Administration archive query", False, f"Exception: {str(e)}")
        return False


def test_scope_filtering_applied_to_archive():
    """Test 10: Verify scope filtering is applied to archive"""
    print_section("Test 10: Scope Filtering Applied to Archive")
    
    from api_v2.services.inbox_service import get_section_archive
    
    # User with very restrictive scope (only unit 1)
    user_restricted = MockUser('SECTION_ADMIN', allowed_unit_ids={1})
    
    # User with broad scope
    user_broad = MockUser('SECTION_ADMIN', allowed_unit_ids=set(range(1, 300)))
    
    try:
        archive_restricted = get_section_archive(user_restricted)
        archive_broad = get_section_archive(user_broad)
        
        # Restricted user should see <= broad user items
        restricted_valid = len(archive_restricted) <= len(archive_broad)
        print_test("Restricted scope sees <= broad scope items", restricted_valid,
                   f"Restricted: {len(archive_restricted)}, Broad: {len(archive_broad)}")
        
        # All items in restricted archive should have target_org_unit_id == 1
        if len(archive_restricted) > 0:
            all_in_scope = all(
                item.get('target_org_unit_id') == 1 
                for item in archive_restricted
            )
            print_test("All restricted items have correct org unit", all_in_scope)
        else:
            print(f"  Restricted archive is empty (valid)")
            all_in_scope = True
        
        return restricted_valid and all_in_scope
        
    except Exception as e:
        print_test("Scope filtering test", False, f"Exception: {str(e)}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_header("INBOX ARCHIVE FEATURE - TEST SUITE")
    print("Testing new archive functionality for workflow inbox")
    
    results = []
    
    # Run all tests
    results.append(("Service functions exist", test_archive_service_functions_exist()))
    results.append(("DB functions exist", test_archive_db_functions_exist()))
    results.append(("Archive items view-only", test_archive_items_are_view_only()))
    results.append(("get_archive routing", test_get_archive_routing()))
    results.append(("Empty scopes handling", test_empty_scopes_return_empty_archive()))
    results.append(("Endpoint exists", test_archive_endpoint_exists()))
    results.append(("Section archive integration", test_section_archive_with_real_data()))
    results.append(("Department archive integration", test_department_archive_with_real_data()))
    results.append(("Administration archive integration", test_administration_archive_with_real_data()))
    results.append(("Scope filtering", test_scope_filtering_applied_to_archive()))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    failed = sum(1 for _, result in results if not result)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{len(results)} passed")
    
    if failed == 0:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠️ {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
