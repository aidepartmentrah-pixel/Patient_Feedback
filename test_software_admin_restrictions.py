"""
SOFTWARE ADMIN ACTION RESTRICTION TEST SUITE

Tests that SOFTWARE_ADMIN has strict action restrictions:
- Can only act at final administration stage (DEPT_ACCEPTED_PENDING_ADMIN)
- Cannot act at section stage (no submit_response, no reject)
- Cannot act at department stage (no accept, no reject)
- Has same final-stage authority as ADMINISTRATION_ADMIN
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
# UNIT TESTS - SOFTWARE_ADMIN Action Restrictions
# =============================================================================

def test_software_admin_at_submitted_to_section():
    """
    TEST: SOFTWARE_ADMIN + SUBMITTED_TO_SECTION
    EXPECT: ["view"] only (no section override)
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('SOFTWARE_ADMIN')
    subcase = {'status': 'SUBMITTED_TO_SECTION'}
    
    actions = _compute_allowed_actions(subcase, user)
    
    expected = ["view"]
    
    passed = actions == expected
    return print_test(
        "SOFTWARE_ADMIN + SUBMITTED_TO_SECTION → [view] only",
        passed,
        f"Expected: {expected}, Got: {actions}"
    )


def test_software_admin_at_returned_to_section():
    """
    TEST: SOFTWARE_ADMIN + RETURNED_TO_SECTION_FOR_REVISION
    EXPECT: ["view"] only (no section override)
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('SOFTWARE_ADMIN')
    subcase = {'status': 'RETURNED_TO_SECTION_FOR_REVISION'}
    
    actions = _compute_allowed_actions(subcase, user)
    
    expected = ["view"]
    
    passed = actions == expected
    return print_test(
        "SOFTWARE_ADMIN + RETURNED_TO_SECTION_FOR_REVISION → [view] only",
        passed,
        f"Expected: {expected}, Got: {actions}"
    )


def test_software_admin_at_section_accepted_pending_dept():
    """
    TEST: SOFTWARE_ADMIN + SECTION_ACCEPTED_PENDING_DEPT
    EXPECT: ["view"] only (no department override)
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('SOFTWARE_ADMIN')
    subcase = {'status': 'SECTION_ACCEPTED_PENDING_DEPT'}
    
    actions = _compute_allowed_actions(subcase, user)
    
    expected = ["view"]
    
    passed = actions == expected
    return print_test(
        "SOFTWARE_ADMIN + SECTION_ACCEPTED_PENDING_DEPT → [view] only",
        passed,
        f"Expected: {expected}, Got: {actions}"
    )


def test_software_admin_at_returned_to_dept():
    """
    TEST: SOFTWARE_ADMIN + RETURNED_TO_DEPT_FOR_REVISION
    EXPECT: ["view"] only (no department override)
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('SOFTWARE_ADMIN')
    subcase = {'status': 'RETURNED_TO_DEPT_FOR_REVISION'}
    
    actions = _compute_allowed_actions(subcase, user)
    
    expected = ["view"]
    
    passed = actions == expected
    return print_test(
        "SOFTWARE_ADMIN + RETURNED_TO_DEPT_FOR_REVISION → [view] only",
        passed,
        f"Expected: {expected}, Got: {actions}"
    )


def test_software_admin_at_dept_accepted_pending_admin():
    """
    TEST: SOFTWARE_ADMIN + DEPT_ACCEPTED_PENDING_ADMIN
    EXPECT: ["view", "accept", "reject"] (final-stage authority)
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('SOFTWARE_ADMIN')
    subcase = {'status': 'DEPT_ACCEPTED_PENDING_ADMIN'}
    
    actions = _compute_allowed_actions(subcase, user)
    
    expected = {"view", "accept", "reject"}
    actual = set(actions)
    
    passed = expected == actual
    return print_test(
        "SOFTWARE_ADMIN + DEPT_ACCEPTED_PENDING_ADMIN → [view, accept, reject]",
        passed,
        f"Expected: {sorted(expected)}, Got: {sorted(actual)}"
    )


def test_software_admin_never_gets_submit_response():
    """
    TEST: SOFTWARE_ADMIN never gets "submit_response" action at any status
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    print_section("SOFTWARE_ADMIN submit_response Restriction")
    
    user = MockUser('SOFTWARE_ADMIN')
    
    statuses_to_test = [
        'SUBMITTED_TO_SECTION',
        'RETURNED_TO_SECTION_FOR_REVISION',
        'SECTION_ACCEPTED_PENDING_DEPT',
        'RETURNED_TO_DEPT_FOR_REVISION',
        'DEPT_ACCEPTED_PENDING_ADMIN'
    ]
    
    all_passed = True
    for status in statuses_to_test:
        subcase = {'status': status}
        actions = _compute_allowed_actions(subcase, user)
        has_submit_response = 'submit_response' in actions
        
        all_passed = all_passed and print_test(
            f"submit_response absent for SOFTWARE_ADMIN + {status}",
            not has_submit_response,
            f"Actions: {actions}" if has_submit_response else "Clean"
        )
    
    return all_passed


def test_software_admin_no_section_stage_actions():
    """
    TEST: SOFTWARE_ADMIN gets no actionable buttons at section stage
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    print_section("SOFTWARE_ADMIN Section Stage Restriction")
    
    user = MockUser('SOFTWARE_ADMIN')
    
    section_statuses = [
        'SUBMITTED_TO_SECTION',
        'RETURNED_TO_SECTION_FOR_REVISION'
    ]
    
    all_passed = True
    for status in section_statuses:
        subcase = {'status': status}
        actions = _compute_allowed_actions(subcase, user)
        
        # Should only have "view", no workflow actions
        has_workflow_actions = any(action in actions for action in ['submit_response', 'accept', 'reject'])
        is_view_only = actions == ["view"]
        
        all_passed = all_passed and print_test(
            f"SOFTWARE_ADMIN has no workflow actions at {status}",
            not has_workflow_actions and is_view_only,
            f"Actions: {actions}"
        )
    
    return all_passed


def test_software_admin_no_dept_stage_actions():
    """
    TEST: SOFTWARE_ADMIN gets no actionable buttons at department stage
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    print_section("SOFTWARE_ADMIN Department Stage Restriction")
    
    user = MockUser('SOFTWARE_ADMIN')
    
    dept_statuses = [
        'SECTION_ACCEPTED_PENDING_DEPT',
        'RETURNED_TO_DEPT_FOR_REVISION'
    ]
    
    all_passed = True
    for status in dept_statuses:
        subcase = {'status': status}
        actions = _compute_allowed_actions(subcase, user)
        
        # Should only have "view", no workflow actions
        has_workflow_actions = any(action in actions for action in ['accept', 'reject'])
        is_view_only = actions == ["view"]
        
        all_passed = all_passed and print_test(
            f"SOFTWARE_ADMIN has no workflow actions at {status}",
            not has_workflow_actions and is_view_only,
            f"Actions: {actions}"
        )
    
    return all_passed


def test_software_admin_vs_administration_admin_at_final_stage():
    """
    TEST: SOFTWARE_ADMIN has same actions as ADMINISTRATION_ADMIN at final stage
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    software_admin = MockUser('SOFTWARE_ADMIN')
    administration_admin = MockUser('ADMINISTRATION_ADMIN')
    subcase = {'status': 'DEPT_ACCEPTED_PENDING_ADMIN'}
    
    software_actions = set(_compute_allowed_actions(subcase, software_admin))
    admin_actions = set(_compute_allowed_actions(subcase, administration_admin))
    
    are_equal = software_actions == admin_actions
    
    return print_test(
        "SOFTWARE_ADMIN has same actions as ADMINISTRATION_ADMIN at final stage",
        are_equal,
        f"SOFTWARE_ADMIN: {sorted(software_actions)}, ADMINISTRATION_ADMIN: {sorted(admin_actions)}"
    )


def test_software_admin_vs_section_admin_different():
    """
    TEST: SOFTWARE_ADMIN has different (restricted) actions from SECTION_ADMIN at section stage
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    software_admin = MockUser('SOFTWARE_ADMIN')
    section_admin = MockUser('SECTION_ADMIN')
    subcase = {'status': 'SUBMITTED_TO_SECTION'}
    
    software_actions = set(_compute_allowed_actions(subcase, software_admin))
    section_actions = set(_compute_allowed_actions(subcase, section_admin))
    
    # SOFTWARE_ADMIN should have fewer actions (just view)
    software_more_restricted = len(software_actions) < len(section_actions)
    software_view_only = software_actions == {"view"}
    section_has_more = 'submit_response' in section_actions or 'reject' in section_actions
    
    passed = software_more_restricted and software_view_only and section_has_more
    
    return print_test(
        "SOFTWARE_ADMIN more restricted than SECTION_ADMIN at section stage",
        passed,
        f"SOFTWARE_ADMIN: {sorted(software_actions)}, SECTION_ADMIN: {sorted(section_actions)}"
    )


def test_software_admin_vs_dept_admin_different():
    """
    TEST: SOFTWARE_ADMIN has different (restricted) actions from DEPARTMENT_ADMIN at dept stage
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    software_admin = MockUser('SOFTWARE_ADMIN')
    dept_admin = MockUser('DEPARTMENT_ADMIN')
    subcase = {'status': 'SECTION_ACCEPTED_PENDING_DEPT'}
    
    software_actions = set(_compute_allowed_actions(subcase, software_admin))
    dept_actions = set(_compute_allowed_actions(subcase, dept_admin))
    
    # SOFTWARE_ADMIN should have fewer actions (just view)
    software_more_restricted = len(software_actions) < len(dept_actions)
    software_view_only = software_actions == {"view"}
    dept_has_more = 'accept' in dept_actions or 'reject' in dept_actions
    
    passed = software_more_restricted and software_view_only and dept_has_more
    
    return print_test(
        "SOFTWARE_ADMIN more restricted than DEPARTMENT_ADMIN at dept stage",
        passed,
        f"SOFTWARE_ADMIN: {sorted(software_actions)}, DEPARTMENT_ADMIN: {sorted(dept_actions)}"
    )


def test_software_admin_action_summary():
    """
    TEST: Summary - SOFTWARE_ADMIN has exactly 1 actionable status
    """
    from backend.api_v2.services.inbox_service import _compute_allowed_actions
    
    print_section("SOFTWARE_ADMIN Action Summary")
    
    user = MockUser('SOFTWARE_ADMIN')
    
    all_statuses = [
        'SUBMITTED_TO_SECTION',
        'RETURNED_TO_SECTION_FOR_REVISION',
        'SECTION_ACCEPTED_PENDING_DEPT',
        'RETURNED_TO_DEPT_FOR_REVISION',
        'DEPT_ACCEPTED_PENDING_ADMIN'
    ]
    
    actionable_count = 0
    view_only_count = 0
    
    for status in all_statuses:
        subcase = {'status': status}
        actions = _compute_allowed_actions(subcase, user)
        
        if len(actions) == 1 and actions[0] == 'view':
            view_only_count += 1
        elif len(actions) > 1:
            actionable_count += 1
            print(f"  ✓ Actionable at {status}: {actions}")
        else:
            print(f"  ? Unexpected at {status}: {actions}")
    
    passed = actionable_count == 1 and view_only_count == 4
    
    return print_test(
        "SOFTWARE_ADMIN has exactly 1 actionable status (final stage only)",
        passed,
        f"Actionable: {actionable_count}, View-only: {view_only_count}"
    )


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all SOFTWARE_ADMIN restriction tests"""
    print("\n" + "="*80)
    print(" SOFTWARE ADMIN ACTION RESTRICTION TEST SUITE")
    print("="*80)
    
    results = {}
    
    # Individual status tests
    print_section("SOFTWARE_ADMIN at Each Status")
    results['section_submitted'] = test_software_admin_at_submitted_to_section()
    results['section_returned'] = test_software_admin_at_returned_to_section()
    results['dept_pending'] = test_software_admin_at_section_accepted_pending_dept()
    results['dept_returned'] = test_software_admin_at_returned_to_dept()
    results['admin_pending'] = test_software_admin_at_dept_accepted_pending_admin()
    
    # Action-specific restrictions
    results['no_submit_response'] = test_software_admin_never_gets_submit_response()
    results['no_section_actions'] = test_software_admin_no_section_stage_actions()
    results['no_dept_actions'] = test_software_admin_no_dept_stage_actions()
    
    # Comparison tests
    print_section("SOFTWARE_ADMIN vs Other Roles")
    results['same_as_admin_final'] = test_software_admin_vs_administration_admin_at_final_stage()
    results['restricted_vs_section'] = test_software_admin_vs_section_admin_different()
    results['restricted_vs_dept'] = test_software_admin_vs_dept_admin_different()
    
    # Summary test
    results['action_summary'] = test_software_admin_action_summary()
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} test groups")
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - SOFTWARE_ADMIN restrictions verified!")
        print("✓ Final-stage authority only (DEPT_ACCEPTED_PENDING_ADMIN)")
        print("✓ No section-stage override")
        print("✓ No department-stage override")
        print("✓ No submit_response action")
        print("✓ Same final-stage authority as ADMINISTRATION_ADMIN")
    else:
        print(f"\n⚠️  {total - passed} test group(s) failed")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
