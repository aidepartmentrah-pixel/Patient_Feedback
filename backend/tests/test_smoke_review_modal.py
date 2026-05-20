"""
Smoke tests — CaseReviewModal frontend + backend integration.

Verifies:
 1. CaseReviewModal.jsx exists and exports a default component
 2. CaseReviewModal imports actOnSubcase, getWorkflowIncidentDetail, getSubcaseResponse
 3. CaseReviewModal handles all action codes: SUBMIT_RESPONSE, ACCEPT_COMPLAINT,
    APPROVE, OVERRIDE, REJECT, REOPEN
 4. WorkflowInboxPage no longer references handleActionClick or openResponseViewer
 5. WorkflowInboxPage imports CaseReviewModal (not old separate modals)
 6. WorkflowInboxPage uses reviewModalOpen state (not modalOpen)
 7. WorkflowInboxPage renders single 'مراجعة الحالة' button logic
 8. Archive row uses openReviewModal (not old handlers)
 9. CaseReviewModal always fetches response (no needsResponse gate)
10. Backend: inbox still returns correct allowed_actions for all roles (reuse mocks)

Run:
    cd backend
    python -m pytest tests/test_smoke_review_modal.py -v
"""

import os
import sys
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── file paths ───────────────────────────────────────────────────────────────
SRC = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "Front_End_Feedback_Analysis", "src"
)

def read_src(rel_path: str) -> str:
    full = os.path.normpath(os.path.join(SRC, rel_path))
    with open(full, encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────────────
# 1. CaseReviewModal file exists
# ─────────────────────────────────────────────────────────────────────────────

def test_case_review_modal_file_exists():
    path = os.path.normpath(os.path.join(SRC, "components", "workflow", "CaseReviewModal.jsx"))
    assert os.path.isfile(path), f"CaseReviewModal.jsx not found at {path}"
    print("  CaseReviewModal.jsx exists ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CaseReviewModal imports required API functions
# ─────────────────────────────────────────────────────────────────────────────

def test_case_review_modal_imports_api():
    src = read_src("components/workflow/CaseReviewModal.jsx")
    assert "actOnSubcase" in src,               "actOnSubcase not imported"
    assert "getWorkflowIncidentDetail" in src,  "getWorkflowIncidentDetail not imported"
    assert "getSubcaseResponse" in src,         "getSubcaseResponse not imported"
    print("  CaseReviewModal API imports ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CaseReviewModal handles all action codes
# ─────────────────────────────────────────────────────────────────────────────

def test_case_review_modal_action_codes():
    src = read_src("components/workflow/CaseReviewModal.jsx")
    for code in ["SUBMIT_RESPONSE", "ACCEPT_COMPLAINT", "APPROVE", "OVERRIDE", "REJECT", "REOPEN"]:
        assert code in src, f"Action code '{code}' missing from CaseReviewModal"
    print("  All 6 action codes present ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 4. WorkflowInboxPage no longer has removed handlers
# ─────────────────────────────────────────────────────────────────────────────

def test_inbox_page_removed_old_handlers():
    src = read_src("pages/WorkflowInboxPage.jsx")
    assert "handleActionClick" not in src, "handleActionClick still present — should be removed"
    assert "openResponseViewer" not in src, "openResponseViewer still present — should be removed"
    print("  Old handlers removed from WorkflowInboxPage ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 5. WorkflowInboxPage imports CaseReviewModal, not old separate modals
# ─────────────────────────────────────────────────────────────────────────────

def test_inbox_page_imports_review_modal():
    src = read_src("pages/WorkflowInboxPage.jsx")
    assert "CaseReviewModal" in src,          "CaseReviewModal not imported"
    assert "CaseActionModal" not in src,      "CaseActionModal should be removed"
    assert "ResponseViewerModal" not in src,  "ResponseViewerModal should be removed"
    assert "IncidentViewerModal" not in src,  "IncidentViewerModal should be removed"
    print("  WorkflowInboxPage imports correct ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 6. WorkflowInboxPage uses reviewModalOpen state
# ─────────────────────────────────────────────────────────────────────────────

def test_inbox_page_uses_review_modal_state():
    src = read_src("pages/WorkflowInboxPage.jsx")
    assert "reviewModalOpen" in src,  "reviewModalOpen state missing"
    assert "reviewModalItem" in src,  "reviewModalItem state missing"
    assert "openReviewModal" in src,  "openReviewModal function missing"
    print("  reviewModal state wiring correct ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 7. WorkflowInboxPage renders مراجعة الحالة button
# ─────────────────────────────────────────────────────────────────────────────

def test_inbox_page_has_single_review_button():
    src = read_src("pages/WorkflowInboxPage.jsx")
    assert "مراجعة الحالة" in src, "مراجعة الحالة button text missing"
    print("  مراجعة الحالة button present ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Archive row uses openReviewModal
# ─────────────────────────────────────────────────────────────────────────────

def test_archive_row_uses_open_review_modal():
    src = read_src("pages/WorkflowInboxPage.jsx")
    # Count occurrences — should appear in both active + archive renders
    count = src.count("openReviewModal")
    assert count >= 2, f"openReviewModal appears only {count} times — expected in both active + archive rows"
    print(f"  openReviewModal referenced {count} times ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 9. CaseReviewModal always fetches response (no needsResponse gate)
# ─────────────────────────────────────────────────────────────────────────────

def test_review_modal_always_fetches_response():
    src = read_src("components/workflow/CaseReviewModal.jsx")
    assert "needsResponse" not in src, (
        "needsResponse guard still present — response should always be fetched"
    )
    # Confirm getSubcaseResponse is called unconditionally
    assert "getSubcaseResponse(subcaseId)" in src, "getSubcaseResponse call missing"
    print("  Response always fetched (no gate) ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Backend allowed_actions still correct after all edits
# ─────────────────────────────────────────────────────────────────────────────

def _mock_user(role_code):
    scope = MagicMock(); scope.role_code = role_code
    user  = MagicMock(); user.scopes = [scope]; user.allowed_unit_ids = {1}
    return user

def _subcase(status, has_resp=True):
    return {
        "subcase_id": 1, "status": status, "target_org_unit_id": 1,
        "section_explanation_text":       "text" if has_resp else None,
        "department_explanation_text":    None,
        "administration_explanation_text": None,
        "section_rejection_text": None,
    }

def test_backend_section_actions():
    from api_v2.services.inbox_service import _compute_allowed_actions
    actions = _compute_allowed_actions(_subcase("SUBMITTED_TO_SECTION", False), _mock_user("SECTION_ADMIN"))
    assert "submit_response"  in actions
    assert "accept_complaint" in actions
    assert "reject"           in actions
    print(f"  SECTION_ADMIN actions: {actions} ✓")

def test_backend_dept_actions():
    from api_v2.services.inbox_service import _compute_allowed_actions
    actions = _compute_allowed_actions(_subcase("SECTION_ACCEPTED_PENDING_DEPT"), _mock_user("DEPARTMENT_ADMIN"))
    assert "accept"   in actions
    assert "override" in actions
    assert "reject"   in actions
    print(f"  DEPARTMENT_ADMIN actions: {actions} ✓")

def test_backend_admin_actions():
    from api_v2.services.inbox_service import _compute_allowed_actions
    actions = _compute_allowed_actions(_subcase("DEPT_ACCEPTED_PENDING_ADMIN"), _mock_user("ADMINISTRATION_ADMIN"))
    assert "accept"   in actions
    assert "override" in actions
    assert "reject"   in actions
    print(f"  ADMINISTRATION_ADMIN actions: {actions} ✓")

def test_backend_supervisor_actions():
    from api_v2.services.inbox_service import _compute_allowed_actions
    actions = _compute_allowed_actions(_subcase("SECTION_DENIED"), _mock_user("COMPLAINT_SUPERVISOR"))
    assert "reopen" in actions
    print(f"  COMPLAINT_SUPERVISOR actions: {actions} ✓")
