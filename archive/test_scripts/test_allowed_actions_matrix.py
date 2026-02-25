"""
ALLOWED ACTIONS STRICT MATRIX TEST SUITE

Tests that _compute_allowed_actions() enforces strict (role, status) matrix:
- No supervisory override (SOFTWARE_ADMIN, WORKER, COMPLAINT_SUPERVISOR)
- Actions only granted at exact responsibility matches
- submit_response only for section stage
- accept/reject only at correct stage for responsible role
"""

import sys
import os

# Force UTF-8 encoding for emoji support
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add backend directory to Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_test(test_name, passed, message=""):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {test_name}")
    if message:
        print(f"   {message}")
    return passed


class MockUser:
    """Mock user for testing with proper scopes"""
    def __init__(self, role_code):
        self.role_code = role_code
        self.scopes = [type('obj', (object,), {'role_code': role_code})]


# =============================================================================
# UNIT TESTS - Direct testing of _compute_allowed_actions
# =============================================================================

def test_section_admin_at_submitted_to_section():
    """
    TEST: SECTION_ADMIN + SUBMITTED_TO_SECTION
    EXPECT: ["view", "submit_response", "reject"]
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('SECTION_ADMIN')
    subcase = {'status': 'SUBMITTED_TO_SECTION'}
    
    actions = _compute_allowed_actions(subcase, user)
    
    expected = {"view", "submit_response", "reject"}
    actual = set(actions)
    
    passed = expected == actual
    return print_test(
        "SECTION_ADMIN + SUBMITTED_TO_SECTION → [view, submit_response, reject]",
        passed,
        f"Expected: {sorted(expected)}, Got: {sorted(actual)}"
    )


def test_section_admin_at_returned_to_section():
    """
    TEST: SECTION_ADMIN + RETURNED_TO_SECTION_FOR_REVISION
    EXPECT: ["view", "submit_response", "reject"]
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('SECTION_ADMIN')
    subcase = {'status': 'RETURNED_TO_SECTION_FOR_REVISION'}
    
    actions = _compute_allowed_actions(subcase, user)
    
    expected = {"view", "submit_response", "reject"}
    actual = set(actions)
    
    passed = expected == actual
    return print_test(
        "SECTION_ADMIN + RETURNED_TO_SECTION_FOR_REVISION → [view, submit_response, reject]",
        passed,
        f"Expected: {sorted(expected)}, Got: {sorted(actual)}"
    )


def test_section_admin_at_wrong_status():
    """
    TEST: SECTION_ADMIN + SECTION_ACCEPTED_PENDING_DEPT (wrong status)
    EXPECT: ["view"] only
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('SECTION_ADMIN')
    subcase = {'status': 'SECTION_ACCEPTED_PENDING_DEPT'}
    
    actions = _compute_allowed_actions(subcase, user)
    
    expected = ["view"]
    
    passed = actions == expected
    return print_test(
        "SECTION_ADMIN + SECTION_ACCEPTED_PENDING_DEPT → [view] only (no cross-stage)",
        passed,
        f"Expected: {expected}, Got: {actions}"
    )


def test_department_admin_at_section_accepted_pending_dept():
    """
    TEST: DEPARTMENT_ADMIN + SECTION_ACCEPTED_PENDING_DEPT
    EXPECT: ["view", "accept", "reject"]
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('DEPARTMENT_ADMIN')
    subcase = {'status': 'SECTION_ACCEPTED_PENDING_DEPT'}
    
    actions = _compute_allowed_actions(subcase, user)
    
    expected = {"view", "accept", "reject"}
    actual = set(actions)
    
    passed = expected == actual
    return print_test(
        "DEPARTMENT_ADMIN + SECTION_ACCEPTED_PENDING_DEPT → [view, accept, reject]",
        passed,
        f"Expected: {sorted(expected)}, Got: {sorted(actual)}"
    )


def test_department_admin_at_returned_to_dept():
    """
    TEST: DEPARTMENT_ADMIN + RETURNED_TO_DEPT_FOR_REVISION
    EXPECT: ["view", "accept", "reject"]
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('DEPARTMENT_ADMIN')
    subcase = {'status': 'RETURNED_TO_DEPT_FOR_REVISION'}
    
    actions = _compute_allowed_actions(subcase, user)
    
    expected = {"view", "accept", "reject"}
    actual = set(actions)
    
    passed = expected == actual
    return print_test(
        "DEPARTMENT_ADMIN + RETURNED_TO_DEPT_FOR_REVISION → [view, accept, reject]",
        passed,
        f"Expected: {sorted(expected)}, Got: {sorted(actual)}"
    )


def test_department_admin_at_wrong_status():
    """
    TEST: DEPARTMENT_ADMIN + SUBMITTED_TO_SECTION (wrong status)
    EXPECT: ["view"] only
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('DEPARTMENT_ADMIN')
    subcase = {'status': 'SUBMITTED_TO_SECTION'}
    
    actions = _compute_allowed_actions(subcase, user)
    
    expected = ["view"]
    
    passed = actions == expected
    return print_test(
        "DEPARTMENT_ADMIN + SUBMITTED_TO_SECTION → [view] only (no cross-stage)",
        passed,
        f"Expected: {expected}, Got: {actions}"
    )


def test_administration_admin_at_dept_accepted_pending_admin():
    """
    TEST: ADMINISTRATION_ADMIN + DEPT_ACCEPTED_PENDING_ADMIN
    EXPECT: ["view", "accept", "reject"]
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('ADMINISTRATION_ADMIN')
    subcase = {'status': 'DEPT_ACCEPTED_PENDING_ADMIN'}
    
    actions = _compute_allowed_actions(subcase, user)
    
    expected = {"view", "accept", "reject"}
    actual = set(actions)
    
    passed = expected == actual
    return print_test(
        "ADMINISTRATION_ADMIN + DEPT_ACCEPTED_PENDING_ADMIN → [view, accept, reject]",
        passed,
        f"Expected: {sorted(expected)}, Got: {sorted(actual)}"
    )


def test_administration_admin_at_section_stage():
    """
    TEST: ADMINISTRATION_ADMIN + SUBMITTED_TO_SECTION (section stage)
    EXPECT: ["view"] only
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('ADMINISTRATION_ADMIN')
    subcase = {'status': 'SUBMITTED_TO_SECTION'}
    
    actions = _compute_allowed_actions(subcase, user)
    
    expected = ["view"]
    
    passed = actions == expected
    return print_test(
        "ADMINISTRATION_ADMIN + SUBMITTED_TO_SECTION → [view] only (no section override)",
        passed,
        f"Expected: {expected}, Got: {actions}"
    )


def test_administration_admin_at_dept_stage():
    """
    TEST: ADMINISTRATION_ADMIN + SECTION_ACCEPTED_PENDING_DEPT (dept stage)
    EXPECT: ["view"] only
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('ADMINISTRATION_ADMIN')
    subcase = {'status': 'SECTION_ACCEPTED_PENDING_DEPT'}
    
    actions = _compute_allowed_actions(subcase, user)
    
    expected = ["view"]
    
    passed = actions == expected
    return print_test(
        "ADMINISTRATION_ADMIN + SECTION_ACCEPTED_PENDING_DEPT → [view] only (no dept override)",
        passed,
        f"Expected: {expected}, Got: {actions}"
    )


def test_worker_no_override():
    """
    TEST: WORKER + ANY STATUS (should be view-only, no override)
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('WORKER')
    
    statuses_to_test = [
        'SUBMITTED_TO_SECTION',
        'SECTION_ACCEPTED_PENDING_DEPT',
        'DEPT_ACCEPTED_PENDING_ADMIN'
    ]
    
    all_passed = True
    for status in statuses_to_test:
        subcase = {'status': status}
        actions = _compute_allowed_actions(subcase, user)
        
        expected = ["view"]
        passed = actions == expected
        
        all_passed = all_passed and print_test(
            f"WORKER + {status} → [view] only (no override)",
            passed,
            f"Expected: {expected}, Got: {actions}"
        )
    
    return all_passed


def test_complaint_supervisor_no_override():
    """
    TEST: COMPLAINT_SUPERVISOR + ANY STATUS (should be view-only, no override)
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('COMPLAINT_SUPERVISOR')
    
    statuses_to_test = [
        'SUBMITTED_TO_SECTION',
        'SECTION_ACCEPTED_PENDING_DEPT',
        'DEPT_ACCEPTED_PENDING_ADMIN'
    ]
    
    all_passed = True
    for status in statuses_to_test:
        subcase = {'status': status}
        actions = _compute_allowed_actions(subcase, user)
        
        expected = ["view"]
        passed = actions == expected
        
        all_passed = all_passed and print_test(
            f"COMPLAINT_SUPERVISOR + {status} → [view] only (no override)",
            passed,
            f"Expected: {expected}, Got: {actions}"
        )
    
    return all_passed


def test_software_admin_no_override():
    """
    TEST: SOFTWARE_ADMIN has limited actions
    - STEP 5 CHANGE: Now has final-stage authority at DEPT_ACCEPTED_PENDING_ADMIN
    - View-only at section and department stages (no cross-stage override)
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('SOFTWARE_ADMIN')
    
    # Test view-only at non-final stages
    view_only_statuses = [
        'SUBMITTED_TO_SECTION',
        'SECTION_ACCEPTED_PENDING_DEPT'
    ]
    
    all_passed = True
    for status in view_only_statuses:
        subcase = {'status': status}
        actions = _compute_allowed_actions(subcase, user)
        
        expected = ["view"]
        passed = actions == expected
        
        all_passed = all_passed and print_test(
            f"SOFTWARE_ADMIN + {status} → [view] only (no override)",
            passed,
            f"Expected: {expected}, Got: {actions}"
        )
    
    # STEP 5: Test final-stage authority at DEPT_ACCEPTED_PENDING_ADMIN
    subcase_final = {'status': 'DEPT_ACCEPTED_PENDING_ADMIN'}
    actions_final = _compute_allowed_actions(subcase_final, user)
    
    expected_final = {"view", "accept", "reject"}
    actual_final = set(actions_final)
    passed_final = expected_final == actual_final
    
    all_passed = all_passed and print_test(
        f"SOFTWARE_ADMIN + DEPT_ACCEPTED_PENDING_ADMIN → [view, accept, reject] (final-stage authority)",
        passed_final,
        f"Expected: {sorted(expected_final)}, Got: {sorted(actual_final)}"
    )
    
    return all_passed


def test_submit_response_only_at_section_stage():
    """
    TEST: "submit_response" action only appears for SECTION_ADMIN at section statuses
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    print_section("submit_response Action Restriction Test")
    
    # Test that SECTION_ADMIN gets submit_response at correct statuses
    section_user = MockUser('SECTION_ADMIN')
    
    should_have = ['SUBMITTED_TO_SECTION', 'RETURNED_TO_SECTION_FOR_REVISION']
    should_not_have = ['SECTION_ACCEPTED_PENDING_DEPT', 'DEPT_ACCEPTED_PENDING_ADMIN']
    
    all_passed = True
    
    for status in should_have:
        subcase = {'status': status}
        actions = _compute_allowed_actions(subcase, section_user)
        has_submit = 'submit_response' in actions
        
        all_passed = all_passed and print_test(
            f"submit_response present for SECTION_ADMIN + {status}",
            has_submit,
            f"Actions: {actions}"
        )
    
    for status in should_not_have:
        subcase = {'status': status}
        actions = _compute_allowed_actions(subcase, section_user)
        has_submit = 'submit_response' in actions
        
        all_passed = all_passed and print_test(
            f"submit_response absent for SECTION_ADMIN + {status}",
            not has_submit,
            f"Actions: {actions}"
        )
    
    # Test that other roles never get submit_response
    other_roles = ['DEPARTMENT_ADMIN', 'ADMINISTRATION_ADMIN', 'WORKER', 'SOFTWARE_ADMIN']
    test_statuses = ['SUBMITTED_TO_SECTION', 'SECTION_ACCEPTED_PENDING_DEPT', 'DEPT_ACCEPTED_PENDING_ADMIN']
    
    for role in other_roles:
        user = MockUser(role)
        for status in test_statuses:
            subcase = {'status': status}
            actions = _compute_allowed_actions(subcase, user)
            has_submit = 'submit_response' in actions
            
            all_passed = all_passed and print_test(
                f"submit_response absent for {role} + {status}",
                not has_submit,
                f"Actions: {actions}" if has_submit else "Clean"
            )
    
    return all_passed


def test_accept_reject_only_at_responsible_stage():
    """
    TEST: "accept" and "reject" only appear at exact responsible stage
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    print_section("accept/reject Action Restriction Test")
    
    # SECTION_ADMIN should have reject (but not accept) at section stage
    section_user = MockUser('SECTION_ADMIN')
    subcase_section = {'status': 'SUBMITTED_TO_SECTION'}
    actions_section = _compute_allowed_actions(subcase_section, section_user)
    
    has_reject_section = 'reject' in actions_section
    has_accept_section = 'accept' in actions_section
    
    passed_section = has_reject_section and not has_accept_section
    print_test(
        "SECTION_ADMIN has 'reject' but not 'accept' at section stage",
        passed_section,
        f"Actions: {actions_section}"
    )
    
    # DEPARTMENT_ADMIN should have both accept and reject at their stage
    dept_user = MockUser('DEPARTMENT_ADMIN')
    subcase_dept = {'status': 'SECTION_ACCEPTED_PENDING_DEPT'}
    actions_dept = _compute_allowed_actions(subcase_dept, dept_user)
    
    has_both_dept = 'accept' in actions_dept and 'reject' in actions_dept
    print_test(
        "DEPARTMENT_ADMIN has both 'accept' and 'reject' at dept stage",
        has_both_dept,
        f"Actions: {actions_dept}"
    )
    
    # ADMINISTRATION_ADMIN should have both accept and reject at their stage
    admin_user = MockUser('ADMINISTRATION_ADMIN')
    subcase_admin = {'status': 'DEPT_ACCEPTED_PENDING_ADMIN'}
    actions_admin = _compute_allowed_actions(subcase_admin, admin_user)
    
    has_both_admin = 'accept' in actions_admin and 'reject' in actions_admin
    print_test(
        "ADMINISTRATION_ADMIN has both 'accept' and 'reject' at admin stage",
        has_both_admin,
        f"Actions: {actions_admin}"
    )
    
    # SOFTWARE_ADMIN (STEP 5): should have accept/reject ONLY at final stage
    software_user = MockUser('SOFTWARE_ADMIN')
    
    # No accept/reject at section or dept stages
    no_override_stages = ['SUBMITTED_TO_SECTION', 'SECTION_ACCEPTED_PENDING_DEPT']
    software_restricted = True
    for status in no_override_stages:
        subcase = {'status': status}
        actions = _compute_allowed_actions(subcase, software_user)
        has_accept_reject = 'accept' in actions or 'reject' in actions
        software_restricted = software_restricted and not has_accept_reject
    
    # Has accept/reject at final stage
    subcase_final = {'status': 'DEPT_ACCEPTED_PENDING_ADMIN'}
    actions_final = _compute_allowed_actions(subcase_final, software_user)
    software_has_final = 'accept' in actions_final and 'reject' in actions_final
    
    passed_software = software_restricted and software_has_final
    print_test(
        "SOFTWARE_ADMIN has accept/reject ONLY at final stage (DEPT_ACCEPTED_PENDING_ADMIN)",
        passed_software,
        f"Section/Dept: view-only, Admin stage: {actions_final}"
    )
    
    # Other non-responsible roles should NEVER have accept or reject
    other_roles = ['WORKER', 'COMPLAINT_SUPERVISOR']
    test_statuses = ['SUBMITTED_TO_SECTION', 'SECTION_ACCEPTED_PENDING_DEPT', 'DEPT_ACCEPTED_PENDING_ADMIN']
    
    no_override = True
    for role in other_roles:
        user = MockUser(role)
        for status in test_statuses:
            subcase = {'status': status}
            actions = _compute_allowed_actions(subcase, user)
            has_accept_reject = 'accept' in actions or 'reject' in actions
            
            no_override = no_override and not has_accept_reject
            if has_accept_reject:
                print_test(
                    f"{role} should NOT have accept/reject at {status}",
                    False,
                    f"Actions: {actions}"
                )
    
    if no_override:
        print_test(
            "WORKER and COMPLAINT_SUPERVISOR never have accept/reject",
            True,
            "All non-responsible roles verified"
        )
    
    return passed_section and has_both_dept and has_both_admin and passed_software and no_override


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all allowedActions strict matrix tests"""
    print("\n" + "="*80)
    print(" ALLOWED ACTIONS STRICT MATRIX TEST SUITE")
    print("="*80)
    
    results = {}
    
    # Individual role + status tests
    print_section("SECTION_ADMIN Action Tests")
    results['section_submitted'] = test_section_admin_at_submitted_to_section()
    results['section_returned'] = test_section_admin_at_returned_to_section()
    results['section_wrong'] = test_section_admin_at_wrong_status()
    
    print_section("DEPARTMENT_ADMIN Action Tests")
    results['dept_pending'] = test_department_admin_at_section_accepted_pending_dept()
    results['dept_returned'] = test_department_admin_at_returned_to_dept()
    results['dept_wrong'] = test_department_admin_at_wrong_status()
    
    print_section("ADMINISTRATION_ADMIN Action Tests")
    results['admin_pending'] = test_administration_admin_at_dept_accepted_pending_admin()
    results['admin_section'] = test_administration_admin_at_section_stage()
    results['admin_dept'] = test_administration_admin_at_dept_stage()
    
    print_section("Supervisory Override Removal Tests")
    results['worker_no_override'] = test_worker_no_override()
    results['supervisor_no_override'] = test_complaint_supervisor_no_override()
    results['software_admin_no_override'] = test_software_admin_no_override()
    
    # Action-specific tests
    results['submit_response_restriction'] = test_submit_response_only_at_section_stage()
    results['accept_reject_restriction'] = test_accept_reject_only_at_responsible_stage()
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} test groups")
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Strict action matrix verified!")
        print("✓ Supervisory override removed")
        print("✓ Actions enforced by exact (role, status) match")
        print("✓ No cross-stage actions")
    else:
        print(f"\n⚠️  {total - passed} test group(s) failed")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
