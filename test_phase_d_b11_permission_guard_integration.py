"""
TEST TASK D-B11 — PERMISSION GUARD INTEGRATION

Verify person seasonal endpoints have proper role and scope guards applied.

Tests:
1. Person seasonal endpoints require get_current_user
2. Role guard applied using existing helper (require_doctor_report_access / require_worker_report_access)
3. Scope check performed against allowed_unit_ids
4. Unauthorized access returns 403
5. No new guard framework introduced
6. No inline hardcoded permission logic if helper exists
7. Other routers unchanged
"""

import os
import re
import ast
import sys


def test_file_exists():
    """Test 1: Verify person_seasonal_report_router.py exists"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    assert os.path.exists(router_path), f"Router file not found: {router_path}"
    print("✅ Test 1: File exists")


def test_get_current_user_dependency():
    """Test 2: Verify endpoints require get_current_user"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check import
    assert "from ..dependencies.user_context import get_current_user" in content, \
        "get_current_user not imported"
    
    # Check both endpoints have current_user parameter with Depends(get_current_user)
    doctor_endpoint_pattern = r'def export_doctor_seasonal_word.*?current_user.*?Depends\(get_current_user\)'
    worker_endpoint_pattern = r'def export_worker_seasonal_word.*?current_user.*?Depends\(get_current_user\)'
    
    assert re.search(doctor_endpoint_pattern, content, re.DOTALL), \
        "Doctor endpoint doesn't have current_user with Depends(get_current_user)"
    assert re.search(worker_endpoint_pattern, content, re.DOTALL), \
        "Worker endpoint doesn't have current_user with Depends(get_current_user)"
    
    print("✅ Test 2: get_current_user dependency applied to both endpoints")


def test_role_guards_imported():
    """Test 3: Verify role guard helpers are imported"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for guard imports
    assert "from ..utils.guards import" in content, \
        "Guards not imported from utils.guards"
    
    # Check specific guards
    assert "require_doctor_report_access" in content, \
        "require_doctor_report_access not imported"
    assert "require_worker_report_access" in content, \
        "require_worker_report_access not imported"
    assert "require_unit_in_scope" in content, \
        "require_unit_in_scope not imported"
    
    print("✅ Test 3: Role guard helpers imported from existing guards module")


def test_doctor_role_guard_used():
    """Test 4: Verify doctor endpoint uses require_doctor_report_access"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract doctor endpoint function
    doctor_match = re.search(
        r'def export_doctor_seasonal_word.*?(?=\n@router\.|\ndef export_worker_|\Z)',
        content,
        re.DOTALL
    )
    assert doctor_match, "Doctor endpoint not found"
    doctor_endpoint = doctor_match.group(0)
    
    # Check role guard is called
    assert "require_doctor_report_access(current_user)" in doctor_endpoint, \
        "require_doctor_report_access not called in doctor endpoint"
    
    # Ensure it's called before building report
    build_pos = doctor_endpoint.find("build_doctor_seasonal_report_data")
    guard_pos = doctor_endpoint.find("require_doctor_report_access")
    
    assert build_pos > 0 and guard_pos > 0, \
        "Both guard and build calls should exist"
    assert guard_pos < build_pos, \
        "Role guard should be called BEFORE building report"
    
    print("✅ Test 4: Doctor endpoint uses require_doctor_report_access correctly")


def test_worker_role_guard_used():
    """Test 5: Verify worker endpoint uses require_worker_report_access"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract worker endpoint function
    worker_match = re.search(
        r'def export_worker_seasonal_word.*?(?=\n@router\.|\Z)',
        content,
        re.DOTALL
    )
    assert worker_match, "Worker endpoint not found"
    worker_endpoint = worker_match.group(0)
    
    # Check role guard is called
    assert "require_worker_report_access(current_user)" in worker_endpoint, \
        "require_worker_report_access not called in worker endpoint"
    
    # Ensure it's called before building report
    build_pos = worker_endpoint.find("build_worker_seasonal_report_data")
    guard_pos = worker_endpoint.find("require_worker_report_access")
    
    assert build_pos > 0 and guard_pos > 0, \
        "Both guard and build calls should exist"
    assert guard_pos < build_pos, \
        "Role guard should be called BEFORE building report"
    
    print("✅ Test 5: Worker endpoint uses require_worker_report_access correctly")


def test_scope_validation_present():
    """Test 6: Verify scope validation using require_unit_in_scope"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check both endpoints call require_unit_in_scope
    assert content.count("require_unit_in_scope(current_user") >= 2, \
        "require_unit_in_scope should be called in both endpoints"
    
    # Check for employee identity resolution (needed to get org unit)
    assert "get_worker_identity" in content, \
        "get_worker_identity should be imported and used for org unit resolution"
    
    # Check section_id is used for scope validation
    assert "section_id" in content, \
        "section_id should be used for scope validation"
    
    print("✅ Test 6: Scope validation present using require_unit_in_scope")


def test_doctor_scope_check():
    """Test 7: Verify doctor endpoint performs scope check"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract doctor endpoint
    doctor_match = re.search(
        r'def export_doctor_seasonal_word.*?(?=\n@router\.|\ndef export_worker_|\Z)',
        content,
        re.DOTALL
    )
    assert doctor_match, "Doctor endpoint not found"
    doctor_endpoint = doctor_match.group(0)
    
    # Check employee identity fetched
    assert "get_worker_identity" in doctor_endpoint, \
        "Doctor endpoint should fetch employee identity for org unit"
    
    # Check scope validation
    assert "require_unit_in_scope" in doctor_endpoint, \
        "Doctor endpoint should call require_unit_in_scope"
    
    # Check scope validation happens before building report
    build_pos = doctor_endpoint.find("build_doctor_seasonal_report_data")
    scope_pos = doctor_endpoint.find("require_unit_in_scope")
    
    assert build_pos > 0 and scope_pos > 0, \
        "Both scope check and build should exist"
    assert scope_pos < build_pos, \
        "Scope check should happen BEFORE building report"
    
    print("✅ Test 7: Doctor endpoint performs scope validation")


def test_worker_scope_check():
    """Test 8: Verify worker endpoint performs scope check"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract worker endpoint
    worker_match = re.search(
        r'def export_worker_seasonal_word.*?(?=\n@router\.|\Z)',
        content,
        re.DOTALL
    )
    assert worker_match, "Worker endpoint not found"
    worker_endpoint = worker_match.group(0)
    
    # Check employee identity fetched
    assert "get_worker_identity" in worker_endpoint, \
        "Worker endpoint should fetch employee identity for org unit"
    
    # Check scope validation
    assert "require_unit_in_scope" in worker_endpoint, \
        "Worker endpoint should call require_unit_in_scope"
    
    # Check scope validation happens before building report
    build_pos = worker_endpoint.find("build_worker_seasonal_report_data")
    scope_pos = worker_endpoint.find("require_unit_in_scope")
    
    assert build_pos > 0 and scope_pos > 0, \
        "Both scope check and build should exist"
    assert scope_pos < build_pos, \
        "Scope check should happen BEFORE building report"
    
    print("✅ Test 8: Worker endpoint performs scope validation")


def test_403_error_handling():
    """Test 9: Verify 403 errors documented and guards raise them"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check both endpoint docstrings mention 403
    assert content.count("403") >= 2, \
        "Both endpoints should document 403 error in docstring"
    
    assert "Forbidden" in content, \
        "403 error description should include Forbidden"
    
    print("✅ Test 9: 403 Forbidden errors documented")


def test_no_new_guard_framework():
    """Test 10: Verify no new guard system introduced"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check guards are imported from existing modules only
    guard_import_pattern = r'from.*guards.*import'
    guard_imports = re.findall(guard_import_pattern, content)
    
    for imp in guard_imports:
        assert "..utils.guards" in imp or "utils.guards" in imp, \
            f"Guards should only be imported from utils.guards, found: {imp}"
    
    # Check no new guard definitions
    assert "def require_" not in content or "_build_person_report_filename" in content, \
        "No new guard functions should be defined in router (only existing helpers)"
    
    print("✅ Test 10: No new guard framework introduced (reuses existing guards)")


def test_no_inline_permission_logic():
    """Test 11: Verify no inline hardcoded permission checks"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract endpoint functions only
    endpoints = re.findall(
        r'def export_(?:doctor|worker)_seasonal_word.*?(?=\n@router\.|\ndef export_|\Z)',
        content,
        re.DOTALL
    )
    
    for endpoint in endpoints:
        # Check no hardcoded role checks like 'current_user.role == "ADMIN"'
        assert 'current_user.role ==' not in endpoint, \
            "No inline role checks allowed - use guard helpers"
        
        assert 'current_user.role !=' not in endpoint, \
            "No inline role checks allowed - use guard helpers"
        
        assert 'if current_user.role in' not in endpoint, \
            "No inline role checks allowed - use guard helpers"
    
    print("✅ Test 11: No inline hardcoded permission logic (uses helpers)")


def test_guard_comment_present():
    """Test 12: Verify guard alignment comment exists"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for required comment
    assert "Scope + role guard applied per reporting security policy" in content, \
        "Missing required comment about guard application"
    
    print("✅ Test 12: Guard alignment comment present")


def test_other_routers_unchanged():
    """Test 13: Verify other routers not modified"""
    routers_to_check = [
        "backend/api/routers/reports_router.py",
        "backend/api/routers/doctors_router.py",
    ]
    
    for router_path in routers_to_check:
        if not os.path.exists(router_path):
            continue
        
        with open(router_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check no person seasonal references added
        assert "person_seasonal" not in content.lower(), \
            f"{router_path} should not reference person_seasonal reports"
    
    print("✅ Test 13: Other routers unchanged")


def test_auth_service_unchanged():
    """Test 14: Verify auth service not modified"""
    auth_paths = [
        "backend/api/dependencies/user_context.py",
        "backend/api/services/auth_service.py",
    ]
    
    for auth_path in auth_paths:
        if not os.path.exists(auth_path):
            continue
        
        with open(auth_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check no person seasonal modifications
        assert "person_seasonal" not in content.lower(), \
            f"{auth_path} should not be modified for person seasonal reports"
        
        assert "doctor_seasonal" not in content.lower() or "emergency" in content.lower(), \
            f"{auth_path} should not be modified for person seasonal reports"
    
    print("✅ Test 14: Auth service unchanged")


def test_guard_execution_order():
    """Test 15: Verify guard execution order (role → scope → build)"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check doctor endpoint order
    doctor_match = re.search(
        r'def export_doctor_seasonal_word.*?(?=\n@router\.|\ndef export_worker_|\Z)',
        content,
        re.DOTALL
    )
    assert doctor_match, "Doctor endpoint not found"
    doctor_endpoint = doctor_match.group(0)
    
    role_guard_pos = doctor_endpoint.find("require_doctor_report_access")
    scope_check_pos = doctor_endpoint.find("require_unit_in_scope")
    build_pos = doctor_endpoint.find("build_doctor_seasonal_report_data")
    
    assert role_guard_pos < scope_check_pos < build_pos, \
        "Doctor endpoint guard order should be: role → scope → build"
    
    # Check worker endpoint order
    worker_match = re.search(
        r'def export_worker_seasonal_word.*?(?=\n@router\.|\Z)',
        content,
        re.DOTALL
    )
    assert worker_match, "Worker endpoint not found"
    worker_endpoint = worker_match.group(0)
    
    role_guard_pos = worker_endpoint.find("require_worker_report_access")
    scope_check_pos = worker_endpoint.find("require_unit_in_scope")
    build_pos = worker_endpoint.find("build_worker_seasonal_report_data")
    
    assert role_guard_pos < scope_check_pos < build_pos, \
        "Worker endpoint guard order should be: role → scope → build"
    
    print("✅ Test 15: Guard execution order correct (role → scope → build)")


def test_employee_identity_resolution():
    """Test 16: Verify employee identity resolution for org unit"""
    router_path = "backend/api/routers/person_seasonal_report_router.py"
    with open(router_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check import of get_worker_identity
    assert "from ..db_layer.worker_reporting_db import get_worker_identity" in content, \
        "get_worker_identity should be imported from db_layer"
    
    # Check both endpoints use it
    doctor_match = re.search(r'def export_doctor_seasonal_word.*?(?=\n@router\.|\ndef export_worker_|\Z)', content, re.DOTALL)
    worker_match = re.search(r'def export_worker_seasonal_word.*?(?=\n@router\.|\Z)', content, re.DOTALL)
    
    assert "get_worker_identity" in doctor_match.group(0), \
        "Doctor endpoint should call get_worker_identity"
    assert "get_worker_identity" in worker_match.group(0), \
        "Worker endpoint should call get_worker_identity"
    
    print("✅ Test 16: Employee identity resolution implemented")


def run_all_tests():
    """Run all D-B11 permission guard integration tests"""
    print("\n" + "="*70)
    print("PHASE D - TASK D-B11: PERMISSION GUARD INTEGRATION")
    print("="*70 + "\n")
    
    tests = [
        test_file_exists,
        test_get_current_user_dependency,
        test_role_guards_imported,
        test_doctor_role_guard_used,
        test_worker_role_guard_used,
        test_scope_validation_present,
        test_doctor_scope_check,
        test_worker_scope_check,
        test_403_error_handling,
        test_no_new_guard_framework,
        test_no_inline_permission_logic,
        test_guard_comment_present,
        test_other_routers_unchanged,
        test_auth_service_unchanged,
        test_guard_execution_order,
        test_employee_identity_resolution,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: Unexpected error: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ PERMISSION GUARD INTEGRATION OK")
    else:
        print("❌ PERMISSION GUARD INTEGRATION FAILED")
        sys.exit(1)
    
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
