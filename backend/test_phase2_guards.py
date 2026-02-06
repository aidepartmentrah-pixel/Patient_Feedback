"""
Phase 2 RBAC: Authorization Guards Tests
Comprehensive tests for role-based authorization guards.
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from fastapi import HTTPException
from api.utils.guards import (
    require_logged_in,
    require_role,
    require_software_admin,
    require_worker,
    require_complaint_supervisor,
    require_section_admin,
    require_department_admin,
    require_administration_admin,
    require_any_admin,
    require_any_supervisor,
    has_role,
    has_any_role,
    get_user_roles,
)
from api.schemas.auth_models import CurrentUser, UserScope
from core.constants.roles import (
    SOFTWARE_ADMIN,
    WORKER,
    COMPLAINT_SUPERVISOR,
    SECTION_ADMIN,
    DEPARTMENT_ADMIN,
    ADMINISTRATION_ADMIN,
)


# ==================== TEST FIXTURES ====================

def create_user(user_id: int, username: str, role_code: str, org_unit_id=None, org_unit_type=None):
    """Create a test user with a single role."""
    return CurrentUser(
        user_id=user_id,
        username=username,
        is_active=True,
        scopes=[UserScope(
            role_code=role_code,
            org_unit_id=org_unit_id,
            org_unit_type=org_unit_type
        )]
    )


def create_multi_role_user(user_id: int, username: str, roles: list):
    """Create a test user with multiple roles."""
    return CurrentUser(
        user_id=user_id,
        username=username,
        is_active=True,
        scopes=[
            UserScope(
                role_code=role_data[0],
                org_unit_id=role_data[1] if len(role_data) > 1 else None,
                org_unit_type=role_data[2] if len(role_data) > 2 else None
            )
            for role_data in roles
        ]
    )


# Test users
software_admin_user = create_user(1, "admin", SOFTWARE_ADMIN, 0, "ADMINISTRATION")
worker_user = create_user(2, "worker", WORKER, 10, "COMPLAINT")
supervisor_user = create_user(3, "supervisor", COMPLAINT_SUPERVISOR, 10, "COMPLAINT")
section_admin_user = create_user(4, "section_admin", SECTION_ADMIN, 10, "SECTION")
dept_admin_user = create_user(5, "dept_admin", DEPARTMENT_ADMIN, 5, "DEPARTMENT")
admin_user = create_user(6, "administration", ADMINISTRATION_ADMIN, 1, "ADMINISTRATION")


# ==================== require_logged_in() TESTS ====================

def test_require_logged_in_with_user():
    """Test require_logged_in allows authenticated user."""
    try:
        require_logged_in(software_admin_user)
        print("✓ require_logged_in: allows authenticated user")
        return True
    except Exception as e:
        print(f"✗ FAILED: require_logged_in with user: {e}")
        return False


def test_require_logged_in_with_none():
    """Test require_logged_in raises 401 for None user."""
    try:
        require_logged_in(None)
        print("✗ FAILED: require_logged_in should raise 401 for None")
        return False
    except HTTPException as e:
        if e.status_code == 401:
            print("✓ require_logged_in: raises 401 for None user")
            return True
        else:
            print(f"✗ FAILED: require_logged_in raised {e.status_code} instead of 401")
            return False


# ==================== require_role() TESTS ====================

def test_require_role_with_matching_role():
    """Test require_role allows user with matching role."""
    try:
        require_role(software_admin_user, [SOFTWARE_ADMIN])
        print("✓ require_role: allows matching role")
        return True
    except Exception as e:
        print(f"✗ FAILED: require_role with matching role: {e}")
        return False


def test_require_role_with_multiple_allowed():
    """Test require_role allows user when role is in allowed list."""
    try:
        require_role(software_admin_user, [WORKER, SOFTWARE_ADMIN, SECTION_ADMIN])
        print("✓ require_role: allows role in allowed list")
        return True
    except Exception as e:
        print(f"✗ FAILED: require_role with multiple allowed: {e}")
        return False


def test_require_role_with_non_matching_role():
    """Test require_role raises 403 for non-matching role."""
    try:
        require_role(worker_user, [SOFTWARE_ADMIN])
        print("✗ FAILED: require_role should raise 403 for non-matching role")
        return False
    except HTTPException as e:
        if e.status_code == 403:
            print("✓ require_role: raises 403 for non-matching role")
            return True
        else:
            print(f"✗ FAILED: require_role raised {e.status_code} instead of 403")
            return False


def test_require_role_with_multi_role_user():
    """Test require_role works with user having multiple roles."""
    multi_role_user = create_multi_role_user(
        99, "multi", 
        [(WORKER, 10, "COMPLAINT"), (SECTION_ADMIN, 10, "SECTION")]
    )
    
    try:
        # Should pass because user has SECTION_ADMIN
        require_role(multi_role_user, [SECTION_ADMIN])
        print("✓ require_role: works with multi-role user")
        return True
    except Exception as e:
        print(f"✗ FAILED: require_role with multi-role user: {e}")
        return False


# ==================== ROLE-SPECIFIC GUARDS TESTS ====================

def test_require_software_admin_success():
    """Test require_software_admin allows SOFTWARE_ADMIN."""
    try:
        require_software_admin(software_admin_user)
        print("✓ require_software_admin: allows SOFTWARE_ADMIN")
        return True
    except Exception as e:
        print(f"✗ FAILED: require_software_admin with correct role: {e}")
        return False


def test_require_software_admin_failure():
    """Test require_software_admin raises 403 for other roles."""
    try:
        require_software_admin(worker_user)
        print("✗ FAILED: require_software_admin should raise 403")
        return False
    except HTTPException as e:
        if e.status_code == 403:
            print("✓ require_software_admin: raises 403 for wrong role")
            return True
        else:
            print(f"✗ FAILED: raised {e.status_code} instead of 403")
            return False


def test_require_worker_success():
    """Test require_worker allows WORKER."""
    try:
        require_worker(worker_user)
        print("✓ require_worker: allows WORKER")
        return True
    except Exception as e:
        print(f"✗ FAILED: require_worker with correct role: {e}")
        return False


def test_require_worker_failure():
    """Test require_worker raises 403 for other roles."""
    try:
        require_worker(software_admin_user)
        print("✗ FAILED: require_worker should raise 403")
        return False
    except HTTPException as e:
        if e.status_code == 403:
            print("✓ require_worker: raises 403 for wrong role")
            return True
        else:
            print(f"✗ FAILED: raised {e.status_code} instead of 403")
            return False


def test_require_complaint_supervisor_success():
    """Test require_complaint_supervisor allows COMPLAINT_SUPERVISOR."""
    try:
        require_complaint_supervisor(supervisor_user)
        print("✓ require_complaint_supervisor: allows COMPLAINT_SUPERVISOR")
        return True
    except Exception as e:
        print(f"✗ FAILED: require_complaint_supervisor with correct role: {e}")
        return False


def test_require_complaint_supervisor_failure():
    """Test require_complaint_supervisor raises 403 for other roles."""
    try:
        require_complaint_supervisor(worker_user)
        print("✗ FAILED: require_complaint_supervisor should raise 403")
        return False
    except HTTPException as e:
        if e.status_code == 403:
            print("✓ require_complaint_supervisor: raises 403 for wrong role")
            return True
        else:
            return False


def test_require_section_admin_success():
    """Test require_section_admin allows SECTION_ADMIN."""
    try:
        require_section_admin(section_admin_user)
        print("✓ require_section_admin: allows SECTION_ADMIN")
        return True
    except Exception as e:
        print(f"✗ FAILED: require_section_admin with correct role: {e}")
        return False


def test_require_section_admin_failure():
    """Test require_section_admin raises 403 for other roles."""
    try:
        require_section_admin(worker_user)
        print("✗ FAILED: require_section_admin should raise 403")
        return False
    except HTTPException as e:
        if e.status_code == 403:
            print("✓ require_section_admin: raises 403 for wrong role")
            return True
        else:
            return False


def test_require_department_admin_success():
    """Test require_department_admin allows DEPARTMENT_ADMIN."""
    try:
        require_department_admin(dept_admin_user)
        print("✓ require_department_admin: allows DEPARTMENT_ADMIN")
        return True
    except Exception as e:
        print(f"✗ FAILED: require_department_admin with correct role: {e}")
        return False


def test_require_department_admin_failure():
    """Test require_department_admin raises 403 for other roles."""
    try:
        require_department_admin(worker_user)
        print("✗ FAILED: require_department_admin should raise 403")
        return False
    except HTTPException as e:
        if e.status_code == 403:
            print("✓ require_department_admin: raises 403 for wrong role")
            return True
        else:
            return False


def test_require_administration_admin_success():
    """Test require_administration_admin allows ADMINISTRATION_ADMIN."""
    try:
        require_administration_admin(admin_user)
        print("✓ require_administration_admin: allows ADMINISTRATION_ADMIN")
        return True
    except Exception as e:
        print(f"✗ FAILED: require_administration_admin with correct role: {e}")
        return False


def test_require_administration_admin_failure():
    """Test require_administration_admin raises 403 for other roles."""
    try:
        require_administration_admin(worker_user)
        print("✗ FAILED: require_administration_admin should raise 403")
        return False
    except HTTPException as e:
        if e.status_code == 403:
            print("✓ require_administration_admin: raises 403 for wrong role")
            return True
        else:
            return False


# ==================== COMBINED GUARDS TESTS ====================

def test_require_any_admin_with_software_admin():
    """Test require_any_admin allows SOFTWARE_ADMIN."""
    try:
        require_any_admin(software_admin_user)
        print("✓ require_any_admin: allows SOFTWARE_ADMIN")
        return True
    except Exception as e:
        print(f"✗ FAILED: require_any_admin with SOFTWARE_ADMIN: {e}")
        return False


def test_require_any_admin_with_section_admin():
    """Test require_any_admin allows SECTION_ADMIN."""
    try:
        require_any_admin(section_admin_user)
        print("✓ require_any_admin: allows SECTION_ADMIN")
        return True
    except Exception as e:
        print(f"✗ FAILED: require_any_admin with SECTION_ADMIN: {e}")
        return False


def test_require_any_admin_with_worker():
    """Test require_any_admin raises 403 for non-admin."""
    try:
        require_any_admin(worker_user)
        print("✗ FAILED: require_any_admin should raise 403 for WORKER")
        return False
    except HTTPException as e:
        if e.status_code == 403:
            print("✓ require_any_admin: raises 403 for non-admin")
            return True
        else:
            return False


def test_require_any_supervisor_with_supervisor():
    """Test require_any_supervisor allows COMPLAINT_SUPERVISOR."""
    try:
        require_any_supervisor(supervisor_user)
        print("✓ require_any_supervisor: allows COMPLAINT_SUPERVISOR")
        return True
    except Exception as e:
        print(f"✗ FAILED: require_any_supervisor with supervisor: {e}")
        return False


def test_require_any_supervisor_with_admin():
    """Test require_any_supervisor allows admin roles."""
    try:
        require_any_supervisor(software_admin_user)
        print("✓ require_any_supervisor: allows SOFTWARE_ADMIN")
        return True
    except Exception as e:
        print(f"✗ FAILED: require_any_supervisor with admin: {e}")
        return False


def test_require_any_supervisor_with_worker():
    """Test require_any_supervisor raises 403 for worker."""
    try:
        require_any_supervisor(worker_user)
        print("✗ FAILED: require_any_supervisor should raise 403 for WORKER")
        return False
    except HTTPException as e:
        if e.status_code == 403:
            print("✓ require_any_supervisor: raises 403 for worker")
            return True
        else:
            return False


# ==================== HELPER FUNCTIONS TESTS ====================

def test_has_role_true():
    """Test has_role returns True for matching role."""
    result = has_role(software_admin_user, SOFTWARE_ADMIN)
    if result is True:
        print("✓ has_role: returns True for matching role")
        return True
    else:
        print("✗ FAILED: has_role should return True")
        return False


def test_has_role_false():
    """Test has_role returns False for non-matching role."""
    result = has_role(worker_user, SOFTWARE_ADMIN)
    if result is False:
        print("✓ has_role: returns False for non-matching role")
        return True
    else:
        print("✗ FAILED: has_role should return False")
        return False


def test_has_any_role_true():
    """Test has_any_role returns True when user has one of roles."""
    result = has_any_role(software_admin_user, [WORKER, SOFTWARE_ADMIN])
    if result is True:
        print("✓ has_any_role: returns True when user has role")
        return True
    else:
        print("✗ FAILED: has_any_role should return True")
        return False


def test_has_any_role_false():
    """Test has_any_role returns False when user has none of roles."""
    result = has_any_role(worker_user, [SOFTWARE_ADMIN, SECTION_ADMIN])
    if result is False:
        print("✓ has_any_role: returns False when user has no role")
        return True
    else:
        print("✗ FAILED: has_any_role should return False")
        return False


def test_get_user_roles():
    """Test get_user_roles returns list of roles."""
    roles = get_user_roles(software_admin_user)
    if roles == [SOFTWARE_ADMIN]:
        print("✓ get_user_roles: returns correct role list")
        return True
    else:
        print(f"✗ FAILED: get_user_roles returned {roles}")
        return False


def test_get_user_roles_multi():
    """Test get_user_roles with multiple roles."""
    multi_role_user = create_multi_role_user(
        99, "multi",
        [(WORKER, 10, "COMPLAINT"), (SECTION_ADMIN, 10, "SECTION")]
    )
    roles = get_user_roles(multi_role_user)
    if roles == [WORKER, SECTION_ADMIN]:
        print("✓ get_user_roles: returns all roles for multi-role user")
        return True
    else:
        print(f"✗ FAILED: get_user_roles returned {roles}")
        return False


# ==================== ERROR FORMAT TESTS ====================

def test_forbidden_error_format():
    """Test that 403 errors have proper format."""
    try:
        require_software_admin(worker_user)
        return False
    except HTTPException as e:
        detail = e.detail
        has_error = "error" in detail
        has_message = "message" in detail
        has_message_ar = "message_ar" in detail
        has_required = "required_roles" in detail
        has_user_roles = "user_roles" in detail
        
        if all([has_error, has_message, has_message_ar, has_required, has_user_roles]):
            print("✓ 403 error format: includes all required fields")
            return True
        else:
            print(f"✗ FAILED: 403 error missing fields")
            return False


def test_unauthorized_error_format():
    """Test that 401 errors have proper format."""
    try:
        require_logged_in(None)
        return False
    except HTTPException as e:
        detail = e.detail
        has_error = "error" in detail
        has_message = "message" in detail
        has_message_ar = "message_ar" in detail
        
        if all([has_error, has_message, has_message_ar]):
            print("✓ 401 error format: includes all required fields")
            return True
        else:
            print("✗ FAILED: 401 error missing fields")
            return False


# ==================== MAIN TEST RUNNER ====================

def run_all_tests():
    """Run all tests and report results."""
    test_functions = [
        # require_logged_in tests
        test_require_logged_in_with_user,
        test_require_logged_in_with_none,
        
        # require_role tests
        test_require_role_with_matching_role,
        test_require_role_with_multiple_allowed,
        test_require_role_with_non_matching_role,
        test_require_role_with_multi_role_user,
        
        # Role-specific guards
        test_require_software_admin_success,
        test_require_software_admin_failure,
        test_require_worker_success,
        test_require_worker_failure,
        test_require_complaint_supervisor_success,
        test_require_complaint_supervisor_failure,
        test_require_section_admin_success,
        test_require_section_admin_failure,
        test_require_department_admin_success,
        test_require_department_admin_failure,
        test_require_administration_admin_success,
        test_require_administration_admin_failure,
        
        # Combined guards
        test_require_any_admin_with_software_admin,
        test_require_any_admin_with_section_admin,
        test_require_any_admin_with_worker,
        test_require_any_supervisor_with_supervisor,
        test_require_any_supervisor_with_admin,
        test_require_any_supervisor_with_worker,
        
        # Helper functions
        test_has_role_true,
        test_has_role_false,
        test_has_any_role_true,
        test_has_any_role_false,
        test_get_user_roles,
        test_get_user_roles_multi,
        
        # Error format tests
        test_forbidden_error_format,
        test_unauthorized_error_format,
    ]
    
    print("\n" + "="*70)
    print("PHASE 2 RBAC: AUTHORIZATION GUARDS TESTS")
    print("="*70 + "\n")
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"✗ ERROR in {test_func.__name__}: {str(e)}")
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {len(test_functions)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(test_functions)*100):.1f}%")
    print("="*70 + "\n")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    sys.exit(0 if failed == 0 else 1)
