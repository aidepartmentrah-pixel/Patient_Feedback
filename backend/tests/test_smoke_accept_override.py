"""
Smoke tests — Accept/Override feature additions.

Covers:
  1. ACCEPT_TEXT constant exists and has correct Arabic value
  2. accept_section_complaint() exists in case_response_service
  3. approve_department() now calls update_department_explanation (no empty text)
  4. approve_administration() now calls update_administration_explanation (no empty text)
  5. inbox_service._compute_allowed_actions gives section admin 'accept_complaint'
  6. inbox_service._compute_allowed_actions gives dept admin 'override'
  7. inbox_service._compute_allowed_actions gives admin admin 'override'
  8. workflow_router handles ACCEPT_COMPLAINT action code (import check)
  9. accept_section_complaint() calls correct DB functions in correct order (mocked)
 10. accept_section_complaint() raises on wrong status (mocked)

Run:
    cd backend
    python -m pytest tests/test_smoke_accept_override.py -v
"""

import sys
import os
import types
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ────────────────────────────────────────────────────────────
# 1. ACCEPT_TEXT constant
# ────────────────────────────────────────────────────────────

def test_accept_text_constant_exists():
    from api_v2.services.case_response_service import ACCEPT_TEXT
    assert ACCEPT_TEXT == "قبول الشكوى", f"ACCEPT_TEXT is '{ACCEPT_TEXT}', expected 'قبول الشكوى'"
    print(f"  ACCEPT_TEXT = '{ACCEPT_TEXT}' ✓")


# ────────────────────────────────────────────────────────────
# 2. accept_section_complaint exists and is callable
# ────────────────────────────────────────────────────────────

def test_accept_section_complaint_exists():
    from api_v2.services import case_response_service as svc
    assert hasattr(svc, "accept_section_complaint"), "accept_section_complaint missing from service"
    assert callable(svc.accept_section_complaint)
    print("  accept_section_complaint callable ✓")


# ────────────────────────────────────────────────────────────
# 3. approve_department source includes update_department_explanation
# ────────────────────────────────────────────────────────────

def test_approve_department_calls_explanation_update():
    import inspect
    from api_v2.services.case_response_service import approve_department
    src = inspect.getsource(approve_department)
    assert "update_department_explanation" in src, (
        "approve_department does not call update_department_explanation — "
        "قبول text will not be written"
    )
    assert "ACCEPT_TEXT" in src, "approve_department does not use ACCEPT_TEXT constant"
    print("  approve_department calls update_department_explanation with ACCEPT_TEXT ✓")


# ────────────────────────────────────────────────────────────
# 4. approve_administration source includes update_administration_explanation
# ────────────────────────────────────────────────────────────

def test_approve_administration_calls_explanation_update():
    import inspect
    from api_v2.services.case_response_service import approve_administration
    src = inspect.getsource(approve_administration)
    assert "update_administration_explanation" in src, (
        "approve_administration does not call update_administration_explanation"
    )
    assert "ACCEPT_TEXT" in src, "approve_administration does not use ACCEPT_TEXT constant"
    print("  approve_administration calls update_administration_explanation with ACCEPT_TEXT ✓")


# ────────────────────────────────────────────────────────────
# 5-7. inbox_service._compute_allowed_actions correct per role
# ────────────────────────────────────────────────────────────

def _mock_user(role_code: str, unit_ids=None):
    scope = MagicMock()
    scope.role_code = role_code
    user = MagicMock()
    user.scopes = [scope]
    user.allowed_unit_ids = unit_ids or {1}
    return user


def _mock_subcase(status: str, with_response: bool = True) -> dict:
    return {
        "subcase_id": 99,
        "status": status,
        "target_org_unit_id": 1,
        "section_explanation_text": "some text" if with_response else None,
        "department_explanation_text": None,
        "administration_explanation_text": None,
        "section_rejection_text": None,
    }


def test_section_allowed_actions_includes_accept_complaint():
    from api_v2.services.inbox_service import _compute_allowed_actions
    user = _mock_user("SECTION_ADMIN")
    subcase = _mock_subcase("SUBMITTED_TO_SECTION", with_response=False)
    actions = _compute_allowed_actions(subcase, user)
    assert "accept_complaint" in actions, f"accept_complaint missing: {actions}"
    assert "submit_response" in actions
    assert "reject" in actions
    print(f"  SECTION_ADMIN SUBMITTED_TO_SECTION actions: {actions} ✓")


def test_section_allowed_actions_returned_for_revision():
    from api_v2.services.inbox_service import _compute_allowed_actions
    user = _mock_user("SECTION_ADMIN")
    subcase = _mock_subcase("RETURNED_TO_SECTION_FOR_REVISION", with_response=True)
    actions = _compute_allowed_actions(subcase, user)
    assert "accept_complaint" in actions, f"accept_complaint missing on returned: {actions}"
    print(f"  SECTION_ADMIN RETURNED_TO_SECTION_FOR_REVISION actions: {actions} ✓")


def test_dept_allowed_actions_includes_override():
    from api_v2.services.inbox_service import _compute_allowed_actions
    user = _mock_user("DEPARTMENT_ADMIN")
    subcase = _mock_subcase("SECTION_ACCEPTED_PENDING_DEPT")
    actions = _compute_allowed_actions(subcase, user)
    assert "override" in actions, f"override missing from dept actions: {actions}"
    assert "accept" in actions
    assert "reject" in actions
    print(f"  DEPARTMENT_ADMIN actions: {actions} ✓")


def test_admin_allowed_actions_includes_override():
    from api_v2.services.inbox_service import _compute_allowed_actions
    user = _mock_user("ADMINISTRATION_ADMIN")
    subcase = _mock_subcase("DEPT_ACCEPTED_PENDING_ADMIN")
    actions = _compute_allowed_actions(subcase, user)
    assert "override" in actions, f"override missing from admin actions: {actions}"
    assert "accept" in actions
    print(f"  ADMINISTRATION_ADMIN actions: {actions} ✓")


def test_software_admin_allowed_actions_includes_override():
    from api_v2.services.inbox_service import _compute_allowed_actions
    user = _mock_user("SOFTWARE_ADMIN")
    subcase = _mock_subcase("DEPT_ACCEPTED_PENDING_ADMIN")
    actions = _compute_allowed_actions(subcase, user)
    assert "override" in actions, f"override missing from software admin actions: {actions}"
    print(f"  SOFTWARE_ADMIN actions: {actions} ✓")


# ────────────────────────────────────────────────────────────
# 8. workflow_router handles ACCEPT_COMPLAINT (source check)
# ────────────────────────────────────────────────────────────

def test_workflow_router_has_accept_complaint_handler():
    router_path = os.path.join(
        os.path.dirname(__file__), "..", "api_v2", "routers", "workflow_router.py"
    )
    with open(router_path, encoding="utf-8") as f:
        src = f.read()
    assert "ACCEPT_COMPLAINT" in src, "ACCEPT_COMPLAINT action missing from workflow_router"
    assert "accept_section_complaint" in src, "accept_section_complaint not called in router"
    print("  workflow_router ACCEPT_COMPLAINT handler present ✓")


# ────────────────────────────────────────────────────────────
# 9. accept_section_complaint mocked end-to-end
# ────────────────────────────────────────────────────────────

def test_accept_section_complaint_mocked():
    """Verify function calls correct DB operations in correct order."""
    from api_v2.services import case_response_service as svc

    fake_subcase = {
        "subcase_id": 42,
        "status": "SUBMITTED_TO_SECTION",
        "incident_request_case_id": 10,
    }
    fake_user = MagicMock()
    fake_user.user_id = 7

    with patch.object(svc.administrative_subcase_db, "get_subcase_by_id", return_value=fake_subcase), \
         patch.object(svc.administrative_subcase_db, "update_section_explanation") as mock_explain, \
         patch.object(svc.administrative_subcase_db, "update_subcase_status") as mock_status, \
         patch.object(svc.action_item_subcase_db, "bulk_update_action_items_status_by_subcase") as mock_items:

        svc.accept_section_complaint(subcase_id=42, current_user=fake_user)

        # Must write ACCEPT_TEXT to section explanation
        mock_explain.assert_called_once_with(
            subcase_id=42,
            text=svc.ACCEPT_TEXT,
            updated_by_user_id=7
        )
        # Must advance status
        mock_status.assert_called_once_with(
            subcase_id=42,
            new_status="SECTION_ACCEPTED_PENDING_DEPT",
            updated_by_user_id=7
        )
        # Must transition action items (even if zero, call must happen)
        mock_items.assert_called_once()
        print("  accept_section_complaint mock calls all correct ✓")


# ────────────────────────────────────────────────────────────
# 10. accept_section_complaint raises on wrong status
# ────────────────────────────────────────────────────────────

def test_accept_section_complaint_wrong_status_raises():
    from api_v2.services import case_response_service as svc

    fake_subcase = {"subcase_id": 42, "status": "DEPT_ACCEPTED_PENDING_ADMIN"}
    fake_user = MagicMock()
    fake_user.user_id = 7

    with patch.object(svc.administrative_subcase_db, "get_subcase_by_id", return_value=fake_subcase):
        with pytest.raises(Exception, match="must be one of"):
            svc.accept_section_complaint(subcase_id=42, current_user=fake_user)

    print("  accept_section_complaint raises correctly on wrong status ✓")


# ────────────────────────────────────────────────────────────
# 11. approve_department mocked — verify explanation is written
# ────────────────────────────────────────────────────────────

def test_approve_department_mocked_writes_text():
    from api_v2.services import case_response_service as svc

    fake_subcase = {"subcase_id": 55, "status": "SECTION_ACCEPTED_PENDING_DEPT"}
    fake_user = MagicMock()
    fake_user.user_id = 3

    with patch.object(svc.administrative_subcase_db, "get_subcase_by_id", return_value=fake_subcase), \
         patch.object(svc.administrative_subcase_db, "update_department_explanation") as mock_explain, \
         patch.object(svc.administrative_subcase_db, "update_subcase_status"), \
         patch.object(svc.action_item_subcase_db, "bulk_update_action_items_status_by_subcase"):

        svc.approve_department(subcase_id=55, current_user=fake_user)

        mock_explain.assert_called_once_with(
            subcase_id=55,
            text=svc.ACCEPT_TEXT,
            updated_by_user_id=3
        )
    print("  approve_department writes ACCEPT_TEXT to DeptExplanation ✓")


# ────────────────────────────────────────────────────────────
# 12. approve_administration mocked — verify explanation is written
# ────────────────────────────────────────────────────────────

def test_approve_administration_mocked_writes_text():
    from api_v2.services import case_response_service as svc

    fake_subcase = {"subcase_id": 77, "status": "DEPT_ACCEPTED_PENDING_ADMIN"}
    fake_user = MagicMock()
    fake_user.user_id = 5

    with patch.object(svc.administrative_subcase_db, "get_subcase_by_id", return_value=fake_subcase), \
         patch.object(svc.administrative_subcase_db, "update_administration_explanation") as mock_explain, \
         patch.object(svc.administrative_subcase_db, "update_subcase_status"), \
         patch.object(svc.action_item_subcase_db, "bulk_update_action_items_status_by_subcase"):

        svc.approve_administration(subcase_id=77, current_user=fake_user)

        mock_explain.assert_called_once_with(
            subcase_id=77,
            text=svc.ACCEPT_TEXT,
            updated_by_user_id=5
        )
    print("  approve_administration writes ACCEPT_TEXT to AdminExplanation ✓")
