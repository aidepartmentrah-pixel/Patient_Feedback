"""
WORKFLOW INBOX RESPONSIBILITY TEST SUITE

Comprehensive unit and service tests for the strict responsibility inbox model (Model A).

Test Coverage:
1. Role-based inbox visibility (only responsible statuses visible)
2. STATUS_ROLE_MAP correctness
3. AllowedActions matrix (strict role×status computation)
4. Worker empty inbox safety
5. Software admin restriction (final-stage authority only)
6. Scope filtering enforcement

Uses pytest with parametrized tests for comprehensive role×status matrix coverage.
"""

import pytest
from decimal import Decimal
from typing import Set, List, Dict, Any


# =============================================================================
# FIXTURES
# =============================================================================

class MockUser:
    """Mock user with role and scope attributes"""
    def __init__(self, role_code: str, allowed_unit_ids: Set[int] = None):
        self.role_code = role_code
        self.scopes = [type('obj', (object,), {'role_code': role_code})]
        self.allowed_unit_ids = allowed_unit_ids if allowed_unit_ids else set()


@pytest.fixture
def section_admin():
    """Section admin with scope for unit 10"""
    return MockUser('SECTION_ADMIN', allowed_unit_ids={10, 20})


@pytest.fixture
def department_admin():
    """Department admin with scope for unit 10"""
    return MockUser('DEPARTMENT_ADMIN', allowed_unit_ids={10, 20})


@pytest.fixture
def administration_admin():
    """Administration admin with scope for unit 10"""
    return MockUser('ADMINISTRATION_ADMIN', allowed_unit_ids={10, 20})


@pytest.fixture
def software_admin():
    """Software admin with scope for unit 10"""
    return MockUser('SOFTWARE_ADMIN', allowed_unit_ids={10, 20})


@pytest.fixture
def worker():
    """Worker with no workflow responsibility"""
    return MockUser('WORKER', allowed_unit_ids={10})


@pytest.fixture
def complaint_supervisor():
    """Complaint supervisor with no workflow responsibility"""
    return MockUser('COMPLAINT_SUPERVISOR', allowed_unit_ids={10})


@pytest.fixture
def subcase_factory():
    """Factory for creating test subcase dictionaries"""
    def create_subcase(status: str, target_org_unit_id: int = 10, subcase_id: int = 1) -> Dict[str, Any]:
        return {
            'subcase_id': subcase_id,
            'incident_request_case_id': 100,
            'target_org_unit_id': target_org_unit_id,
            'status': status,
            'case_type': 'INCIDENT_RESPONSE',
            'created_at': '2026-02-11',
        }
    return create_subcase


# =============================================================================
# TEST GROUP 1: STATUS_ROLE_MAP Correctness
# =============================================================================

def test_status_role_map_exists():
    """STATUS_ROLE_MAP constant exists and has expected structure"""
    from api_v2.services.inbox_service import STATUS_ROLE_MAP
    
    assert STATUS_ROLE_MAP is not None
    assert isinstance(STATUS_ROLE_MAP, dict)
    assert 'SECTION_ADMIN' in STATUS_ROLE_MAP
    assert 'DEPARTMENT_ADMIN' in STATUS_ROLE_MAP
    assert 'ADMINISTRATION_ADMIN' in STATUS_ROLE_MAP


def test_status_role_map_section_admin():
    """SECTION_ADMIN mapped to correct statuses"""
    from api_v2.services.inbox_service import STATUS_ROLE_MAP
    
    expected_statuses = {'SUBMITTED_TO_SECTION', 'RETURNED_TO_SECTION_FOR_REVISION'}
    actual_statuses = set(STATUS_ROLE_MAP['SECTION_ADMIN'])
    
    assert actual_statuses == expected_statuses


def test_status_role_map_department_admin():
    """DEPARTMENT_ADMIN mapped to correct statuses"""
    from api_v2.services.inbox_service import STATUS_ROLE_MAP
    
    expected_statuses = {'SECTION_ACCEPTED_PENDING_DEPT', 'RETURNED_TO_DEPT_FOR_REVISION'}
    actual_statuses = set(STATUS_ROLE_MAP['DEPARTMENT_ADMIN'])
    
    assert actual_statuses == expected_statuses


def test_status_role_map_administration_admin():
    """ADMINISTRATION_ADMIN mapped to correct statuses"""
    from api_v2.services.inbox_service import STATUS_ROLE_MAP
    
    expected_statuses = {'DEPT_ACCEPTED_PENDING_ADMIN'}
    actual_statuses = set(STATUS_ROLE_MAP['ADMINISTRATION_ADMIN'])
    
    assert actual_statuses == expected_statuses


def test_status_role_map_no_overlaps():
    """Status mappings don't overlap between roles (strict responsibility)"""
    from api_v2.services.inbox_service import STATUS_ROLE_MAP
    
    section_statuses = set(STATUS_ROLE_MAP['SECTION_ADMIN'])
    dept_statuses = set(STATUS_ROLE_MAP['DEPARTMENT_ADMIN'])
    admin_statuses = set(STATUS_ROLE_MAP['ADMINISTRATION_ADMIN'])
    
    # No overlaps
    assert not section_statuses & dept_statuses
    assert not section_statuses & admin_statuses
    assert not dept_statuses & admin_statuses


def test_status_role_map_terminal_statuses_excluded():
    """Terminal statuses not in STATUS_ROLE_MAP"""
    from api_v2.services.inbox_service import STATUS_ROLE_MAP
    
    terminal_statuses = {'ADMIN_APPROVED', 'SECTION_DENIED', 'FORCE_CLOSED', 'CLOSED'}
    all_mapped_statuses = set()
    for statuses in STATUS_ROLE_MAP.values():
        all_mapped_statuses.update(statuses)
    
    # No terminal statuses in map
    assert not all_mapped_statuses & terminal_statuses


# =============================================================================
# TEST GROUP 2: Scope Filtering (_apply_scope_filter)
# =============================================================================

def test_scope_filter_includes_allowed_units(subcase_factory):
    """Subcases with allowed target_org_unit_id are included"""
    from api_v2.services.inbox_service import _apply_scope_filter
    
    user = MockUser('SECTION_ADMIN', allowed_unit_ids={10, 20})
    subcases = [
        subcase_factory('SUBMITTED_TO_SECTION', target_org_unit_id=10, subcase_id=1),
        subcase_factory('SUBMITTED_TO_SECTION', target_org_unit_id=20, subcase_id=2),
    ]
    
    filtered = _apply_scope_filter(subcases, user)
    
    assert len(filtered) == 2
    assert all(sc['target_org_unit_id'] in {10, 20} for sc in filtered)


def test_scope_filter_excludes_disallowed_units(subcase_factory):
    """Subcases with disallowed target_org_unit_id are excluded"""
    from api_v2.services.inbox_service import _apply_scope_filter
    
    user = MockUser('SECTION_ADMIN', allowed_unit_ids={10})
    subcases = [
        subcase_factory('SUBMITTED_TO_SECTION', target_org_unit_id=10, subcase_id=1),
        subcase_factory('SUBMITTED_TO_SECTION', target_org_unit_id=20, subcase_id=2),
        subcase_factory('SUBMITTED_TO_SECTION', target_org_unit_id=30, subcase_id=3),
    ]
    
    filtered = _apply_scope_filter(subcases, user)
    
    assert len(filtered) == 1
    assert filtered[0]['target_org_unit_id'] == 10


def test_scope_filter_empty_allowed_units_returns_empty(subcase_factory):
    """User with empty allowed_unit_ids gets empty result"""
    from api_v2.services.inbox_service import _apply_scope_filter
    
    user = MockUser('SECTION_ADMIN', allowed_unit_ids=set())
    subcases = [
        subcase_factory('SUBMITTED_TO_SECTION', target_org_unit_id=10, subcase_id=1),
    ]
    
    filtered = _apply_scope_filter(subcases, user)
    
    assert len(filtered) == 0


def test_scope_filter_no_attribute_returns_empty(subcase_factory):
    """User without allowed_unit_ids attribute gets empty result (security default)"""
    from api_v2.services.inbox_service import _apply_scope_filter
    
    user = type('obj', (object,), {'role_code': 'SECTION_ADMIN'})
    subcases = [
        subcase_factory('SUBMITTED_TO_SECTION', target_org_unit_id=10, subcase_id=1),
    ]
    
    filtered = _apply_scope_filter(subcases, user)
    
    assert len(filtered) == 0


def test_scope_filter_excludes_force_closed(subcase_factory):
    """FORCE_CLOSED subcases excluded even if in allowed scope"""
    from api_v2.services.inbox_service import _apply_scope_filter
    
    user = MockUser('SECTION_ADMIN', allowed_unit_ids={10})
    subcases = [
        subcase_factory('SUBMITTED_TO_SECTION', target_org_unit_id=10, subcase_id=1),
        subcase_factory('FORCE_CLOSED', target_org_unit_id=10, subcase_id=2),
    ]
    
    filtered = _apply_scope_filter(subcases, user)
    
    assert len(filtered) == 1
    assert filtered[0]['status'] != 'FORCE_CLOSED'


# =============================================================================
# TEST GROUP 3: AllowedActions Matrix (Strict Role×Status)
# =============================================================================

@pytest.mark.parametrize("status,expected_actions", [
    ('SUBMITTED_TO_SECTION', {'view', 'submit_response', 'reject'}),
    ('RETURNED_TO_SECTION_FOR_REVISION', {'view', 'submit_response', 'reject'}),
    ('SECTION_ACCEPTED_PENDING_DEPT', {'view'}),  # Cross-stage: view only
    ('DEPT_ACCEPTED_PENDING_ADMIN', {'view'}),  # Cross-stage: view only
])
def test_allowed_actions_section_admin(section_admin, subcase_factory, status, expected_actions):
    """SECTION_ADMIN has correct actions for each status"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    subcase = subcase_factory(status)
    actions = _compute_allowed_actions(subcase, section_admin)
    
    assert set(actions) == expected_actions


@pytest.mark.parametrize("status,expected_actions", [
    ('SECTION_ACCEPTED_PENDING_DEPT', {'view', 'accept', 'reject'}),
    ('RETURNED_TO_DEPT_FOR_REVISION', {'view', 'accept', 'reject'}),
    ('SUBMITTED_TO_SECTION', {'view'}),  # Cross-stage: view only
    ('DEPT_ACCEPTED_PENDING_ADMIN', {'view'}),  # Cross-stage: view only
])
def test_allowed_actions_department_admin(department_admin, subcase_factory, status, expected_actions):
    """DEPARTMENT_ADMIN has correct actions for each status"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    subcase = subcase_factory(status)
    actions = _compute_allowed_actions(subcase, department_admin)
    
    assert set(actions) == expected_actions


@pytest.mark.parametrize("status,expected_actions", [
    ('DEPT_ACCEPTED_PENDING_ADMIN', {'view', 'accept', 'reject'}),
    ('SUBMITTED_TO_SECTION', {'view'}),  # Cross-stage: view only
    ('SECTION_ACCEPTED_PENDING_DEPT', {'view'}),  # Cross-stage: view only
])
def test_allowed_actions_administration_admin(administration_admin, subcase_factory, status, expected_actions):
    """ADMINISTRATION_ADMIN has correct actions for each status"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    subcase = subcase_factory(status)
    actions = _compute_allowed_actions(subcase, administration_admin)
    
    assert set(actions) == expected_actions


@pytest.mark.parametrize("status,expected_actions", [
    ('SUBMITTED_TO_SECTION', {'view'}),  # View only at section
    ('SECTION_ACCEPTED_PENDING_DEPT', {'view'}),  # View only at dept
    ('DEPT_ACCEPTED_PENDING_ADMIN', {'view', 'accept', 'reject'}),  # Full authority at final stage
])
def test_allowed_actions_software_admin(software_admin, subcase_factory, status, expected_actions):
    """SOFTWARE_ADMIN has final-stage authority only (Step 5)"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    subcase = subcase_factory(status)
    actions = _compute_allowed_actions(subcase, software_admin)
    
    assert set(actions) == expected_actions


@pytest.mark.parametrize("status", [
    'SUBMITTED_TO_SECTION',
    'SECTION_ACCEPTED_PENDING_DEPT',
    'DEPT_ACCEPTED_PENDING_ADMIN',
])
def test_allowed_actions_worker_view_only(worker, subcase_factory, status):
    """WORKER has view-only access at all statuses (no workflow authority)"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    subcase = subcase_factory(status)
    actions = _compute_allowed_actions(subcase, worker)
    
    assert actions == ['view']


@pytest.mark.parametrize("status", [
    'SUBMITTED_TO_SECTION',
    'SECTION_ACCEPTED_PENDING_DEPT',
    'DEPT_ACCEPTED_PENDING_ADMIN',
])
def test_allowed_actions_complaint_supervisor_view_only(complaint_supervisor, subcase_factory, status):
    """COMPLAINT_SUPERVISOR has view-only access at all statuses"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    subcase = subcase_factory(status)
    actions = _compute_allowed_actions(subcase, complaint_supervisor)
    
    assert actions == ['view']


def test_submit_response_only_for_section_admin(section_admin, department_admin, administration_admin, subcase_factory):
    """submit_response action only appears for SECTION_ADMIN at section statuses"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    subcase = subcase_factory('SUBMITTED_TO_SECTION')
    
    # SECTION_ADMIN should have submit_response
    section_actions = _compute_allowed_actions(subcase, section_admin)
    assert 'submit_response' in section_actions
    
    # Other roles should NOT have submit_response
    dept_actions = _compute_allowed_actions(subcase, department_admin)
    assert 'submit_response' not in dept_actions
    
    admin_actions = _compute_allowed_actions(subcase, administration_admin)
    assert 'submit_response' not in admin_actions


def test_accept_reject_only_at_responsible_stage(section_admin, department_admin, administration_admin, subcase_factory):
    """accept/reject actions only appear at responsible stage"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    # SECTION_ADMIN has reject (but not accept) at section stage
    section_subcase = subcase_factory('SUBMITTED_TO_SECTION')
    section_actions = _compute_allowed_actions(section_subcase, section_admin)
    assert 'reject' in section_actions
    assert 'accept' not in section_actions
    
    # DEPARTMENT_ADMIN has both accept and reject at dept stage
    dept_subcase = subcase_factory('SECTION_ACCEPTED_PENDING_DEPT')
    dept_actions = _compute_allowed_actions(dept_subcase, department_admin)
    assert 'accept' in dept_actions
    assert 'reject' in dept_actions
    
    # ADMINISTRATION_ADMIN has both accept and reject at admin stage
    admin_subcase = subcase_factory('DEPT_ACCEPTED_PENDING_ADMIN')
    admin_actions = _compute_allowed_actions(admin_subcase, administration_admin)
    assert 'accept' in admin_actions
    assert 'reject' in admin_actions


# =============================================================================
# TEST GROUP 4: Role-Based Inbox Routing
# =============================================================================

def test_get_inbox_section_admin_routing(section_admin):
    """get_inbox routes SECTION_ADMIN to get_section_inbox"""
    from api_v2.services.inbox_service import get_inbox
    
    # Should return list (no exception)
    inbox = get_inbox(section_admin)
    
    assert isinstance(inbox, list)


def test_get_inbox_department_admin_routing(department_admin):
    """get_inbox routes DEPARTMENT_ADMIN to get_department_inbox"""
    from api_v2.services.inbox_service import get_inbox
    
    # Should return list (no exception)
    inbox = get_inbox(department_admin)
    
    assert isinstance(inbox, list)


def test_get_inbox_administration_admin_routing(administration_admin):
    """get_inbox routes ADMINISTRATION_ADMIN to get_administration_inbox"""
    from api_v2.services.inbox_service import get_inbox
    
    # Should return list (no exception)
    inbox = get_inbox(administration_admin)
    
    assert isinstance(inbox, list)


def test_get_inbox_worker_returns_empty(worker):
    """get_inbox returns empty list for WORKER (explicit handling, Step 4)"""
    from api_v2.services.inbox_service import get_inbox
    
    inbox = get_inbox(worker)
    
    assert isinstance(inbox, list)
    assert len(inbox) == 0


def test_get_inbox_software_admin_returns_empty(software_admin):
    """get_inbox returns empty list for SOFTWARE_ADMIN (no unified inbox, Step 1)"""
    from api_v2.services.inbox_service import get_inbox
    
    inbox = get_inbox(software_admin)
    
    assert isinstance(inbox, list)
    assert len(inbox) == 0


def test_get_inbox_complaint_supervisor_returns_empty(complaint_supervisor):
    """get_inbox returns empty list for COMPLAINT_SUPERVISOR (no unified inbox)"""
    from api_v2.services.inbox_service import get_inbox
    
    inbox = get_inbox(complaint_supervisor)
    
    assert isinstance(inbox, list)
    assert len(inbox) == 0


def test_get_inbox_no_scopes_returns_empty():
    """get_inbox returns empty list for user without scopes (defensive)"""
    from api_v2.services.inbox_service import get_inbox
    
    user = type('obj', (object,), {})  # No scopes attribute
    inbox = get_inbox(user)
    
    assert isinstance(inbox, list)
    assert len(inbox) == 0


def test_get_inbox_none_user_returns_empty():
    """get_inbox returns empty list for None user (defensive)"""
    from api_v2.services.inbox_service import get_inbox
    
    inbox = get_inbox(None)
    
    assert isinstance(inbox, list)
    assert len(inbox) == 0


# =============================================================================
# TEST GROUP 5: Worker Safety (Step 4)
# =============================================================================

def test_worker_explicit_handling_before_try_catch(worker):
    """WORKER is explicitly handled before try-catch block"""
    from api_v2.services.inbox_service import get_inbox
    
    # Worker should get empty immediately, not through exception handling
    inbox = get_inbox(worker)
    
    assert inbox == []


def test_worker_inbox_no_exception(worker):
    """WORKER inbox call does not raise exception"""
    from api_v2.services.inbox_service import get_inbox
    
    # Should not raise
    inbox = get_inbox(worker)
    
    assert isinstance(inbox, list)


def test_worker_inbox_fast():
    """WORKER inbox returns quickly (no DB overhead)"""
    from api_v2.services.inbox_service import get_inbox
    import time
    
    worker = MockUser('WORKER', allowed_unit_ids={10})
    
    start = time.time()
    inbox = get_inbox(worker)
    duration = time.time() - start
    
    # Should be near-instant (< 50ms)
    assert duration < 0.05
    assert inbox == []


# =============================================================================
# TEST GROUP 6: Software Admin Restriction (Step 5)
# =============================================================================

def test_software_admin_view_only_at_section_stage(software_admin, subcase_factory):
    """SOFTWARE_ADMIN has view-only at section stage (no override)"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    subcase = subcase_factory('SUBMITTED_TO_SECTION')
    actions = _compute_allowed_actions(subcase, software_admin)
    
    assert actions == ['view']


def test_software_admin_view_only_at_dept_stage(software_admin, subcase_factory):
    """SOFTWARE_ADMIN has view-only at department stage (no override)"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    subcase = subcase_factory('SECTION_ACCEPTED_PENDING_DEPT')
    actions = _compute_allowed_actions(subcase, software_admin)
    
    assert actions == ['view']


def test_software_admin_full_authority_at_final_stage(software_admin, subcase_factory):
    """SOFTWARE_ADMIN has accept/reject at final administration stage"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    subcase = subcase_factory('DEPT_ACCEPTED_PENDING_ADMIN')
    actions = _compute_allowed_actions(subcase, software_admin)
    
    assert set(actions) == {'view', 'accept', 'reject'}


def test_software_admin_no_submit_response(software_admin, subcase_factory):
    """SOFTWARE_ADMIN never has submit_response action"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    statuses = [
        'SUBMITTED_TO_SECTION',
        'RETURNED_TO_SECTION_FOR_REVISION',
        'SECTION_ACCEPTED_PENDING_DEPT',
        'RETURNED_TO_DEPT_FOR_REVISION',
        'DEPT_ACCEPTED_PENDING_ADMIN'
    ]
    
    for status in statuses:
        subcase = subcase_factory(status)
        actions = _compute_allowed_actions(subcase, software_admin)
        assert 'submit_response' not in actions


def test_software_admin_same_as_admin_at_final_stage(software_admin, administration_admin, subcase_factory):
    """SOFTWARE_ADMIN has same actions as ADMINISTRATION_ADMIN at final stage"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    subcase = subcase_factory('DEPT_ACCEPTED_PENDING_ADMIN')
    
    software_actions = set(_compute_allowed_actions(subcase, software_admin))
    admin_actions = set(_compute_allowed_actions(subcase, administration_admin))
    
    assert software_actions == admin_actions


# =============================================================================
# TEST GROUP 7: Inbox Item Structure
# =============================================================================

def test_build_inbox_item_structure(section_admin, subcase_factory):
    """_build_inbox_item returns correct structure"""
    from api_v2.services.inbox_service import _build_inbox_item
    
    subcase = subcase_factory('SUBMITTED_TO_SECTION')
    item = _build_inbox_item(subcase, section_admin)
    
    # Check required fields
    assert 'subcase_id' in item
    assert 'case_type' in item
    assert 'status' in item
    assert 'target_org_unit_id' in item
    assert 'allowed_actions' in item
    
    # Check allowed_actions is computed
    assert isinstance(item['allowed_actions'], list)
    assert len(item['allowed_actions']) > 0


def test_inbox_item_includes_allowed_actions(section_admin, subcase_factory):
    """Inbox item includes allowed_actions array"""
    from api_v2.services.inbox_service import _build_inbox_item
    
    subcase = subcase_factory('SUBMITTED_TO_SECTION')
    item = _build_inbox_item(subcase, section_admin)
    
    expected_actions = {'view', 'submit_response', 'reject'}
    assert set(item['allowed_actions']) == expected_actions


# =============================================================================
# TEST GROUP 8: Role×Status Full Matrix (Comprehensive)
# =============================================================================

@pytest.mark.parametrize("role_code,status,should_have_actions", [
    # SECTION_ADMIN - has actions at section statuses only
    ('SECTION_ADMIN', 'SUBMITTED_TO_SECTION', True),
    ('SECTION_ADMIN', 'RETURNED_TO_SECTION_FOR_REVISION', True),
    ('SECTION_ADMIN', 'SECTION_ACCEPTED_PENDING_DEPT', False),
    ('SECTION_ADMIN', 'DEPT_ACCEPTED_PENDING_ADMIN', False),
    
    # DEPARTMENT_ADMIN - has actions at dept statuses only
    ('DEPARTMENT_ADMIN', 'SUBMITTED_TO_SECTION', False),
    ('DEPARTMENT_ADMIN', 'SECTION_ACCEPTED_PENDING_DEPT', True),
    ('DEPARTMENT_ADMIN', 'RETURNED_TO_DEPT_FOR_REVISION', True),
    ('DEPARTMENT_ADMIN', 'DEPT_ACCEPTED_PENDING_ADMIN', False),
    
    # ADMINISTRATION_ADMIN - has actions at admin status only
    ('ADMINISTRATION_ADMIN', 'SUBMITTED_TO_SECTION', False),
    ('ADMINISTRATION_ADMIN', 'SECTION_ACCEPTED_PENDING_DEPT', False),
    ('ADMINISTRATION_ADMIN', 'DEPT_ACCEPTED_PENDING_ADMIN', True),
    
    # SOFTWARE_ADMIN - has actions at final stage only (Step 5)
    ('SOFTWARE_ADMIN', 'SUBMITTED_TO_SECTION', False),
    ('SOFTWARE_ADMIN', 'SECTION_ACCEPTED_PENDING_DEPT', False),
    ('SOFTWARE_ADMIN', 'DEPT_ACCEPTED_PENDING_ADMIN', True),
    
    # WORKER - never has workflow actions
    ('WORKER', 'SUBMITTED_TO_SECTION', False),
    ('WORKER', 'SECTION_ACCEPTED_PENDING_DEPT', False),
    ('WORKER', 'DEPT_ACCEPTED_PENDING_ADMIN', False),
    
    # COMPLAINT_SUPERVISOR - never has workflow actions
    ('COMPLAINT_SUPERVISOR', 'SUBMITTED_TO_SECTION', False),
    ('COMPLAINT_SUPERVISOR', 'SECTION_ACCEPTED_PENDING_DEPT', False),
    ('COMPLAINT_SUPERVISOR', 'DEPT_ACCEPTED_PENDING_ADMIN', False),
])
def test_role_status_action_matrix(role_code, status, should_have_actions, subcase_factory):
    """Comprehensive role×status matrix test for workflow action authority"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser(role_code, allowed_unit_ids={10})
    subcase = subcase_factory(status)
    actions = _compute_allowed_actions(subcase, user)
    
    workflow_actions = [a for a in actions if a != 'view']
    has_workflow_actions = len(workflow_actions) > 0
    
    assert has_workflow_actions == should_have_actions, \
        f"{role_code} at {status}: expected workflow_actions={should_have_actions}, got actions={actions}"


# =============================================================================
# TEST GROUP 9: Terminal Status Handling
# =============================================================================

@pytest.mark.parametrize("terminal_status", [
    'ADMIN_APPROVED',
    'SECTION_DENIED',
    'FORCE_CLOSED',
    'CLOSED',
])
def test_terminal_statuses_view_only(section_admin, subcase_factory, terminal_status):
    """Terminal statuses only have view action (no workflow actions)"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    subcase = subcase_factory(terminal_status)
    actions = _compute_allowed_actions(subcase, section_admin)
    
    # Terminal statuses should only have 'view'
    assert actions == ['view']


# =============================================================================
# TEST GROUP 10: Security Lock Documentation (Step 6)
# =============================================================================

def test_security_lock_comments_present():
    """SECURITY LOCK comment blocks present in inbox functions (Step 6)"""
    import inspect
    from api_v2.services import inbox_service
    
    section_source = inspect.getsource(inbox_service.get_section_inbox)
    dept_source = inspect.getsource(inbox_service.get_department_inbox)
    admin_source = inspect.getsource(inbox_service.get_administration_inbox)
    
    assert 'SECURITY LOCK' in section_source
    assert 'SECURITY LOCK' in dept_source
    assert 'SECURITY LOCK' in admin_source


def test_apply_scope_filter_called_in_all_inboxes():
    """_apply_scope_filter is called in all active inbox functions (Step 6)"""
    import inspect
    from api_v2.services import inbox_service
    
    section_source = inspect.getsource(inbox_service.get_section_inbox)
    dept_source = inspect.getsource(inbox_service.get_department_inbox)
    admin_source = inspect.getsource(inbox_service.get_administration_inbox)
    
    assert '_apply_scope_filter' in section_source
    assert '_apply_scope_filter' in dept_source
    assert '_apply_scope_filter' in admin_source


# =============================================================================
# TEST GROUP 11: Edge Cases and Defensive Programming
# =============================================================================

def test_empty_subcases_list_returns_empty(section_admin):
    """Empty subcases list returns empty (no crash)"""
    from api_v2.services.inbox_service import _apply_scope_filter
    
    filtered = _apply_scope_filter([], section_admin)
    
    assert filtered == []


def test_subcase_without_status_gets_view_only(section_admin):
    """Subcase without status field gets view-only actions"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    subcase = {'subcase_id': 1, 'target_org_unit_id': 10}  # No status
    actions = _compute_allowed_actions(subcase, section_admin)
    
    # Should default to view-only (defensive)
    assert 'view' in actions


def test_compute_allowed_actions_unknown_role(subcase_factory):
    """Unknown role gets view-only actions (defensive)"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    user = MockUser('UNKNOWN_ROLE', allowed_unit_ids={10})
    subcase = subcase_factory('SUBMITTED_TO_SECTION')
    actions = _compute_allowed_actions(subcase, user)
    
    assert actions == ['view']


def test_compute_allowed_actions_unknown_status(section_admin):
    """Unknown status gets view-only actions (defensive)"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    subcase = {'subcase_id': 1, 'status': 'UNKNOWN_STATUS', 'target_org_unit_id': 10}
    actions = _compute_allowed_actions(subcase, section_admin)
    
    assert actions == ['view']


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

def test_scope_filter_performance(subcase_factory):
    """Scope filter performs well with large subcase lists"""
    from api_v2.services.inbox_service import _apply_scope_filter
    import time
    
    user = MockUser('SECTION_ADMIN', allowed_unit_ids={10, 20, 30})
    
    # Create 1000 subcases
    subcases = [
        subcase_factory('SUBMITTED_TO_SECTION', target_org_unit_id=(i % 50), subcase_id=i)
        for i in range(1000)
    ]
    
    start = time.time()
    filtered = _apply_scope_filter(subcases, user)
    duration = time.time() - start
    
    # Should complete in < 100ms
    assert duration < 0.1
    
    # Should include only allowed units (10, 20, 30)
    assert all(sc['target_org_unit_id'] in {10, 20, 30} for sc in filtered)


def test_compute_allowed_actions_performance(section_admin, subcase_factory):
    """_compute_allowed_actions performs well"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    import time
    
    subcase = subcase_factory('SUBMITTED_TO_SECTION')
    
    # Compute 10000 times
    start = time.time()
    for _ in range(10000):
        actions = _compute_allowed_actions(subcase, section_admin)
    duration = time.time() - start
    
    # Should complete 10k computations in < 1 second
    assert duration < 1.0


# =============================================================================
# INTEGRATION-STYLE TESTS (No DB, but realistic flow)
# =============================================================================

def test_full_inbox_flow_section_admin(section_admin, subcase_factory):
    """Full inbox flow: routing → scope → actions"""
    from api_v2.services.inbox_service import get_inbox, _apply_scope_filter, _build_inbox_item
    
    # This test simulates the full flow without DB
    # In real flow: get_inbox → get_section_inbox → DB → scope filter → build items
    
    # 1. User requests inbox
    inbox = get_inbox(section_admin)
    
    # 2. Should return list
    assert isinstance(inbox, list)


def test_strict_responsibility_no_cross_stage_visibility(section_admin, department_admin, subcase_factory):
    """Strict responsibility: section admin doesn't see dept subcases"""
    from api_v2.services.inbox_service import _compute_allowed_actions
    
    # Section admin looking at dept status
    dept_subcase = subcase_factory('SECTION_ACCEPTED_PENDING_DEPT')
    section_actions = _compute_allowed_actions(dept_subcase, section_admin)
    
    # Should only have view (no workflow actions)
    assert section_actions == ['view']
    
    # Dept admin looking at section status
    section_subcase = subcase_factory('SUBMITTED_TO_SECTION')
    dept_actions = _compute_allowed_actions(section_subcase, department_admin)
    
    # Should only have view (no workflow actions)
    assert dept_actions == ['view']
