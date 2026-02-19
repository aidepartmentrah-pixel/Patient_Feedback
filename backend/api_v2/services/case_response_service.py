"""
Case Response Service (API V2)
Handles workflow state transitions for administrative subcases.

This is a SERVICE LAYER file - contains workflow logic only.
Does NOT contain: HTTP logic, DB SQL, permissions, scopes.

Complete workflow engine with:
- Section-level actions
- Department-level actions
- Administration-level actions
- Force close capability
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import HTTPException
from backend.api_v2.db_layer import administrative_subcase_db
from backend.api_v2.db_layer import action_item_subcase_db
from backend.api.db_layer import auth_db
from backend.api.db_layer import incident_case_feedback


# ============================================================
# INTERNAL HELPER FUNCTIONS
# ============================================================

def _load_subcase_or_fail(subcase_id: int) -> Dict[str, Any]:
    """
    Load subcase by ID or raise exception if not found.
    
    Args:
        subcase_id: Subcase ID
    
    Returns:
        Subcase dict
    
    Raises:
        Exception: If subcase not found
    """
    subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
    if subcase is None:
        raise Exception(f"Subcase {subcase_id} not found")
    return subcase


def _assert_status(subcase: Dict[str, Any], allowed_statuses: List[str]) -> None:
    """
    Assert subcase status is in allowed list.
    
    Args:
        subcase: Subcase dict
        allowed_statuses: List of allowed status values
    
    Raises:
        Exception: If status not in allowed list
    """
    current_status = subcase.get('status')
    if current_status not in allowed_statuses:
        subcase_id = subcase.get('subcase_id')
        if len(allowed_statuses) == 1:
            raise Exception(
                f"Subcase {subcase_id} has status '{current_status}' "
                f"but must be '{allowed_statuses[0]}'"
            )
        else:
            raise Exception(
                f"Subcase {subcase_id} has status '{current_status}' "
                f"but must be one of: {', '.join(allowed_statuses)}"
            )


def _replace_action_items(
    subcase_id: int,
    action_items: List[Dict[str, Any]],
    current_user
) -> None:
    """
    Replace all existing action items for a subcase with new ones.
    Used by department and administration override functions.
    
    Args:
        subcase_id: Subcase ID
        action_items: List of new action items to create
        current_user: Current user object (must have user_id attribute)
    """
    # Get all existing action items
    existing_items = action_item_subcase_db.get_action_items_by_subcase(subcase_id)
    
    # Delete all existing action items
    for item in existing_items:
        action_item_subcase_db.delete_action_item(item['action_item_id'])
    
    # Create new action items (DRAFT status)
    for item in action_items:
        action_item_subcase_db.create_action_item(
            subcase_id=subcase_id,
            title=item['title'],
            description=item['description'],
            created_by_user_id=current_user.user_id,
            due_date=item.get('due_date'),
            initial_status='DRAFT',
            assigned_to_user_id=item.get('assigned_to_user_id')
        )


# ============================================================
# RESPONSE VIEWER (read-only)
# ============================================================


def _has_submitted_response(subcase: Dict[str, Any]) -> bool:
    """
    Check whether a subcase has ever had a response submitted.
    A response exists if any explanation text field is non-empty.
    """
    return bool(
        subcase.get('section_explanation_text')
        or subcase.get('department_explanation_text')
        or subcase.get('administration_explanation_text')
    )


def _pick_latest_explanation(subcase: Dict[str, Any]) -> Optional[str]:
    """
    Return the highest-priority (most recent level) explanation text.
    Priority: administration > department > section.
    """
    return (
        subcase.get('administration_explanation_text')
        or subcase.get('department_explanation_text')
        or subcase.get('section_explanation_text')
    )


def _pick_latest_rejection(subcase: Dict[str, Any]) -> Optional[str]:
    """
    Return the highest-priority rejection text.
    Priority: administration > department > section.
    Used when no explanation text exists (e.g. SECTION_DENIED cases).
    """
    return (
        subcase.get('administration_rejection_text')
        or subcase.get('department_rejection_text')
        or subcase.get('section_rejection_text')
    )


# Statuses that indicate a response has been submitted at least once.
# If the subcase is still at SUBMITTED_TO_SECTION (initial state)
# and section_explanation_text is NULL, no response exists yet.
_RESPONSE_EXISTS_STATUSES = {
    'SECTION_ACCEPTED_PENDING_DEPT',
    'RETURNED_TO_DEPT_FOR_REVISION',
    'DEPT_ACCEPTED_PENDING_ADMIN',
    'RETURNED_TO_SECTION_FOR_REVISION',
    'ADMIN_APPROVED',
    'SECTION_DENIED',
    'FORCE_CLOSED',
}


def get_subcase_response(subcase_id: int, current_user) -> Dict[str, Any]:
    """
    Return the latest submitted response (explanation + action items)
    for a subcase, for read-only viewing by reviewers.

    Authorization:
    - User's allowed_unit_ids must include the subcase's target org unit.
    - User must be a reviewer-level role OR a section admin re-viewing
      their own submitted response.

    Returns dict matching the frontend ResponseViewerModal contract:
        {
            "explanation_text": str,
            "action_items": [...],
            "submitted_by": str,
            "submitted_at": str | None
        }

    Raises:
        HTTPException 403 if not authorised
        HTTPException 404 if subcase not found or no response exists
    """
    if current_user is None:
        raise HTTPException(status_code=403, detail="Not authorized to view this subcase response")

    # Load subcase -------------------------------------------------------
    subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
    if subcase is None:
        raise HTTPException(status_code=404, detail="No response found for this subcase")

    # --- Scope check (Phase 2.5) ----------------------------------------
    allowed_unit_ids = getattr(current_user, 'allowed_unit_ids', None) or set()
    target_org_unit_id = subcase.get('target_org_unit_id')
    if target_org_unit_id not in allowed_unit_ids:
        raise HTTPException(status_code=403, detail="Not authorized to view this subcase response")

    # --- Role check ------------------------------------------------------
    role_code = (
        current_user.scopes[0].role_code
        if current_user.scopes
        else None
    )
    reviewer_roles = {
        'DEPARTMENT_ADMIN', 'ADMINISTRATION_ADMIN',
        'SOFTWARE_ADMIN', 'COMPLAINT_SUPERVISOR',
    }
    is_reviewer = role_code in reviewer_roles
    
    # Section admin can view responses on subcases in their scope.
    # This covers:
    # - Active inbox: RETURNED_TO_SECTION_FOR_REVISION (re-review before resubmit)
    # - Archive view: SECTION_ACCEPTED_PENDING_DEPT, ADMIN_APPROVED, etc.
    #   (reviewing what they previously submitted)
    # Scope filtering above ensures they can only see their own org unit's cases.
    is_section_admin = role_code == 'SECTION_ADMIN'
    
    # Worker can also view responses for cases they may have reopened
    is_worker = role_code == 'WORKER'
    
    if not is_reviewer and not is_section_admin and not is_worker:
        raise HTTPException(status_code=403, detail="Not authorized to view this subcase response")

    # --- Response existence check ----------------------------------------
    explanation_text = _pick_latest_explanation(subcase)
    rejection_text = _pick_latest_rejection(subcase)
    is_rejection = False
    
    if not explanation_text and not rejection_text:
        raise HTTPException(status_code=404, detail="No response found for this subcase")
    
    # If no explanation but rejection exists, show the rejection text
    if not explanation_text and rejection_text:
        explanation_text = rejection_text
        is_rejection = True

    # --- Action items ----------------------------------------------------
    raw_items = action_item_subcase_db.get_action_items_by_subcase(subcase_id)
    action_items = [
        {
            "title": item.get('title'),
            "description": item.get('description'),
            "due_date": (
                item['due_date'].isoformat()
                if item.get('due_date') else None
            ),
            "status": item.get('status'),
        }
        for item in raw_items
    ]

    # --- Submitter info --------------------------------------------------
    submitted_by = None
    updated_by_id = subcase.get('updated_by_user_id')
    if updated_by_id:
        user_record = auth_db.get_user_by_id(updated_by_id)
        if user_record:
            submitted_by = user_record.get('username')

    submitted_at = subcase.get('updated_at')
    if submitted_at and hasattr(submitted_at, 'isoformat'):
        submitted_at = submitted_at.isoformat()
    else:
        submitted_at = str(submitted_at) if submitted_at else None

    result = {
        "explanation_text": explanation_text,
        "action_items": action_items,
        "submitted_by": submitted_by,
        "submitted_at": submitted_at,
    }
    
    if is_rejection:
        result["is_rejection"] = True
        result["rejection_text"] = rejection_text
    
    return result


# ============================================================
# SECTION-LEVEL ACTIONS
# ============================================================


def submit_section_response(
    subcase_id: int,
    explanation_text: str,
    action_items: List[Dict[str, Any]],
    current_user,
    rca_feedback: Optional[Dict[str, Any]] = None
) -> None:
    """
    Section Administrator accepts responsibility and provides response.
    
    Status transition: 
    - SUBMITTED_TO_SECTION -> SECTION_ACCEPTED_PENDING_DEPT (initial submission)
    - RETURNED_TO_SECTION_FOR_REVISION -> SECTION_ACCEPTED_PENDING_DEPT (resubmission after rejection)
    
    WORKFLOW CONTRACT: When resubmitting from RETURNED_TO_SECTION_FOR_REVISION,
    existing action items are replaced (not appended).
    
    Args:
        subcase_id: Subcase ID
        explanation_text: Section's explanation text
        action_items: List of action items to create (DRAFT status)
            [{"title": str, "description": str, "due_date": str or None}, ...]
        current_user: Current user object (must have user_id attribute)
        rca_feedback: Optional RCA feedback data with Cause_* and Preventive_* fields
            RCA is mandatory for inbox submissions.
    
    Raises:
        Exception: If user is None, subcase not found, status invalid, or RCA missing
    """
    if current_user is None:
        raise Exception("current_user cannot be None")
    
    # RCA is mandatory for section submissions
    if rca_feedback is None:
        raise Exception("RCA feedback is required for section response submission")
    
    subcase = _load_subcase_or_fail(subcase_id)
    
    # Allow both initial submission and resubmission after rejection
    _assert_status(subcase, ['SUBMITTED_TO_SECTION', 'RETURNED_TO_SECTION_FOR_REVISION'])
    
    # Get incident ID for RCA linkage
    incident_id = subcase.get('incident_request_case_id')
    if not incident_id:
        raise Exception(f"Subcase {subcase_id} has no linked incident case")
    
    # Check if RCA already exists for this subcase (prevent duplicates)
    existing_rca = incident_case_feedback.get_rca_feedback_by_subcase(subcase_id)
    if existing_rca:
        raise Exception(f"RCA feedback already exists for subcase {subcase_id}. RCA cannot be edited.")
    
    # If resubmitting (returned for revision), replace existing action items
    if subcase.get('status') == 'RETURNED_TO_SECTION_FOR_REVISION':
        _replace_action_items(subcase_id, action_items, current_user)
    else:
        # Initial submission: create new action items
        for item in action_items:
            action_item_subcase_db.create_action_item(
                subcase_id=subcase_id,
                title=item['title'],
                description=item['description'],
                created_by_user_id=current_user.user_id,
                due_date=item.get('due_date'),
                initial_status='DRAFT',
                assigned_to_user_id=item.get('assigned_to_user_id')
            )
    
    # Create RCA feedback record linked to subcase
    incident_case_feedback.create_subcase_rca_feedback(
        subcase_id=subcase_id,
        incident_id=incident_id,
        feedback_data=rca_feedback,
        created_by_user_id=current_user.user_id
    )
    
    # Update section explanation
    administrative_subcase_db.update_section_explanation(
        subcase_id=subcase_id,
        text=explanation_text,
        updated_by_user_id=current_user.user_id
    )
    
    # Transition to next workflow stage
    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='SECTION_ACCEPTED_PENDING_DEPT',
        updated_by_user_id=current_user.user_id
    )
    
    # PARALLEL ACTION ITEM TRANSITION: DRAFT -> SUBMITTED_TO_DEPT
    # Action items follow the subcase through the approval pipeline
    action_item_subcase_db.bulk_update_action_items_status_by_subcase(
        subcase_id=subcase_id,
        to_status='SUBMITTED_TO_DEPT',
        updated_by_user_id=current_user.user_id,
        from_statuses=['DRAFT']
    )


def reject_responsibility(
    subcase_id: int,
    rejection_text: str,
    current_user
) -> None:
    """
    Section Administrator rejects responsibility for the subcase.
    
    Status transition:
    - SUBMITTED_TO_SECTION -> SECTION_DENIED (TERMINAL)
    - RETURNED_TO_SECTION_FOR_REVISION -> SECTION_DENIED (TERMINAL)
    
    Both initial submissions and returned-for-revision subcases can be rejected
    by the section administrator. This matches the allowed_actions matrix which
    grants ["view", "submit_response", "reject"] for both statuses.
    
    Args:
        subcase_id: Subcase ID
        rejection_text: Section's rejection explanation
        current_user: Current user object (must have user_id attribute)
    
    Raises:
        Exception: If user is None, subcase not found, or status invalid
    """
    if current_user is None:
        raise Exception("current_user cannot be None")
    
    subcase = _load_subcase_or_fail(subcase_id)
    _assert_status(subcase, ['SUBMITTED_TO_SECTION', 'RETURNED_TO_SECTION_FOR_REVISION'])
    
    # Update rejection text
    administrative_subcase_db.update_section_rejection(
        subcase_id=subcase_id,
        text=rejection_text,
        updated_by_user_id=current_user.user_id
    )
    
    # Transition to terminal state
    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='SECTION_DENIED',
        updated_by_user_id=current_user.user_id
    )


# ============================================================
# COMPLAINT SUPERVISOR ACTIONS
# ============================================================

def reopen_denied_case(
    subcase_id: int,
    rejection_text: str,
    current_user
) -> None:
    """
    Complaint Supervisor reopens a case that was denied by section.
    
    The supervisor reviews the denied subcase, optionally edits the incident
    severity (done separately via incident API), and returns it to section
    for reconsideration.
    
    Status transition: SECTION_DENIED -> RETURNED_TO_SECTION_FOR_REVISION
    
    This breaks the terminal status of SECTION_DENIED and puts the case
    back into the section's inbox for revision.
    
    Args:
        subcase_id: Subcase ID
        rejection_text: Supervisor's note explaining why it's being reopened
        current_user: Current user object (must be COMPLAINT_SUPERVISOR)
    
    Raises:
        Exception: If user is None, subcase not found, or status invalid
        HTTPException(403): If user is not COMPLAINT_SUPERVISOR or SOFTWARE_ADMIN
    """
    if current_user is None:
        raise Exception("current_user cannot be None")
    
    # Role check: only COMPLAINT_SUPERVISOR, WORKER, and SOFTWARE_ADMIN can reopen
    role_code = (
        current_user.scopes[0].role_code
        if current_user.scopes
        else None
    )
    if role_code not in ('COMPLAINT_SUPERVISOR', 'WORKER', 'SOFTWARE_ADMIN'):
        raise HTTPException(
            status_code=403,
            detail="Only COMPLAINT_SUPERVISOR, WORKER, or SOFTWARE_ADMIN can reopen denied cases."
        )
    
    subcase = _load_subcase_or_fail(subcase_id)
    _assert_status(subcase, ['SECTION_DENIED'])
    
    # Scope check: subcase must be within user's allowed units
    target_org_unit_id = subcase.get('target_org_unit_id')
    allowed_unit_ids = getattr(current_user, 'allowed_unit_ids', None) or set()
    if target_org_unit_id not in allowed_unit_ids:
        raise HTTPException(
            status_code=403,
            detail="Subcase is outside your organizational scope."
        )
    
    # Store the supervisor's reopen note in the section rejection field
    # (appends context for the section to understand why it's back)
    if rejection_text:
        administrative_subcase_db.update_section_rejection(
            subcase_id=subcase_id,
            text=rejection_text,
            updated_by_user_id=current_user.user_id
        )
    
    # Transition from terminal SECTION_DENIED -> RETURNED_TO_SECTION_FOR_REVISION
    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='RETURNED_TO_SECTION_FOR_REVISION',
        updated_by_user_id=current_user.user_id
    )


# ============================================================
# DEPARTMENT-LEVEL ACTIONS
# ============================================================

def approve_department(
    subcase_id: int,
    current_user
) -> None:
    """
    Department Administrator approves the section's response.
    
    Status transition:
    - SECTION_ACCEPTED_PENDING_DEPT -> DEPT_ACCEPTED_PENDING_ADMIN
    - RETURNED_TO_DEPT_FOR_REVISION -> DEPT_ACCEPTED_PENDING_ADMIN
    
    Args:
        subcase_id: Subcase ID
        current_user: Current user object (must have user_id attribute)
    
    Raises:
        Exception: If user is None, subcase not found, or status invalid
    """
    if current_user is None:
        raise Exception("current_user cannot be None")
    
    subcase = _load_subcase_or_fail(subcase_id)
    _assert_status(subcase, ['SECTION_ACCEPTED_PENDING_DEPT', 'RETURNED_TO_DEPT_FOR_REVISION'])
    
    # Transition to next workflow stage
    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='DEPT_ACCEPTED_PENDING_ADMIN',
        updated_by_user_id=current_user.user_id
    )
    
    # PARALLEL ACTION ITEM TRANSITION: SUBMITTED_TO_DEPT -> SUBMITTED_TO_ADMIN
    # Action items follow the subcase through the approval pipeline
    action_item_subcase_db.bulk_update_action_items_status_by_subcase(
        subcase_id=subcase_id,
        to_status='SUBMITTED_TO_ADMIN',
        updated_by_user_id=current_user.user_id,
        from_statuses=['SUBMITTED_TO_DEPT']
    )


def reject_department(
    subcase_id: int,
    rejection_text: str,
    current_user
) -> None:
    """
    Department Administrator rejects the section's response.
    
    WORKFLOW CONTRACT CHANGE: Rejection is NOT terminal - it returns for revision.
    
    Status transition:
    - SECTION_ACCEPTED_PENDING_DEPT -> RETURNED_TO_SECTION_FOR_REVISION
    - RETURNED_TO_DEPT_FOR_REVISION -> RETURNED_TO_SECTION_FOR_REVISION
    
    This creates a rework loop where section must resubmit using OVERRIDE.
    Action items remain untouched (will be replaced on resubmission).
    
    Args:
        subcase_id: Subcase ID
        rejection_text: Department's rejection explanation
        current_user: Current user object (must have user_id attribute)
    
    Raises:
        Exception: If user is None, subcase not found, or status invalid
    """
    if current_user is None:
        raise Exception("current_user cannot be None")
    
    subcase = _load_subcase_or_fail(subcase_id)
    _assert_status(subcase, ['SECTION_ACCEPTED_PENDING_DEPT', 'RETURNED_TO_DEPT_FOR_REVISION'])
    
    # Update rejection text
    administrative_subcase_db.update_department_rejection(
        subcase_id=subcase_id,
        text=rejection_text,
        updated_by_user_id=current_user.user_id
    )
    
    # Return to section for revision (NOT terminal)
    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='RETURNED_TO_SECTION_FOR_REVISION',
        updated_by_user_id=current_user.user_id
    )
    
    # PARALLEL ACTION ITEM TRANSITION: SUBMITTED_TO_DEPT -> DEPT_REJECTED
    # Action items mirror the rejection; will be replaced on resubmission
    action_item_subcase_db.bulk_update_action_items_status_by_subcase(
        subcase_id=subcase_id,
        to_status='DEPT_REJECTED',
        updated_by_user_id=current_user.user_id,
        from_statuses=['SUBMITTED_TO_DEPT']
    )


def override_department(
    subcase_id: int,
    explanation_text: str,
    action_items: List[Dict[str, Any]],
    current_user
) -> None:
    """
    Department Administrator overrides section's action items with their own.
    
    Status transition:
    - SECTION_ACCEPTED_PENDING_DEPT -> DEPT_ACCEPTED_PENDING_ADMIN
    - RETURNED_TO_DEPT_FOR_REVISION -> DEPT_ACCEPTED_PENDING_ADMIN
    
    Args:
        subcase_id: Subcase ID
        explanation_text: Department's explanation text
        action_items: List of replacement action items (DRAFT status)
            [{"title": str, "description": str, "due_date": str or None}, ...]
        current_user: Current user object (must have user_id attribute)
    
    Raises:
        Exception: If user is None, subcase not found, or status invalid
    """
    if current_user is None:
        raise Exception("current_user cannot be None")
    
    subcase = _load_subcase_or_fail(subcase_id)
    _assert_status(subcase, ['SECTION_ACCEPTED_PENDING_DEPT', 'RETURNED_TO_DEPT_FOR_REVISION'])
    
    # Replace all action items
    _replace_action_items(subcase_id, action_items, current_user)
    
    # Update department explanation
    administrative_subcase_db.update_department_explanation(
        subcase_id=subcase_id,
        text=explanation_text,
        updated_by_user_id=current_user.user_id
    )
    
    # Transition to next workflow stage
    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='DEPT_ACCEPTED_PENDING_ADMIN',
        updated_by_user_id=current_user.user_id
    )
    
    # PARALLEL ACTION ITEM TRANSITION: DRAFT -> SUBMITTED_TO_ADMIN
    # Dept override creates new items as DRAFT, then forwards directly to admin
    action_item_subcase_db.bulk_update_action_items_status_by_subcase(
        subcase_id=subcase_id,
        to_status='SUBMITTED_TO_ADMIN',
        updated_by_user_id=current_user.user_id,
        from_statuses=['DRAFT']
    )


# ============================================================
# ADMINISTRATION-LEVEL ACTIONS
# ============================================================

def approve_administration(
    subcase_id: int,
    current_user
) -> None:
    """
    Administration Administrator approves the case.
    
    Status transition: DEPT_ACCEPTED_PENDING_ADMIN -> ADMIN_APPROVED
    
    Args:
        subcase_id: Subcase ID
        current_user: Current user object (must have user_id attribute)
    
    Raises:
        Exception: If user is None, subcase not found, or status invalid
    """
    if current_user is None:
        raise Exception("current_user cannot be None")
    
    subcase = _load_subcase_or_fail(subcase_id)
    _assert_status(subcase, ['DEPT_ACCEPTED_PENDING_ADMIN'])
    
    # Transition to final approval state
    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='ADMIN_APPROVED',
        updated_by_user_id=current_user.user_id
    )
    
    # PARALLEL ACTION ITEM TRANSITION: SUBMITTED_TO_ADMIN -> ADMIN_APPROVED
    # Action items are now approved and ready for execution (follow-up / calendar)
    action_item_subcase_db.bulk_update_action_items_status_by_subcase(
        subcase_id=subcase_id,
        to_status='ADMIN_APPROVED',
        updated_by_user_id=current_user.user_id,
        from_statuses=['SUBMITTED_TO_ADMIN']
    )


def reject_administration(
    subcase_id: int,
    rejection_text: str,
    current_user
) -> None:
    """
    Administration Administrator rejects the case.
    
    WORKFLOW CONTRACT CHANGE: Rejection is NOT terminal - it returns for revision.
    
    Status transition: DEPT_ACCEPTED_PENDING_ADMIN -> RETURNED_TO_DEPT_FOR_REVISION
    
    This creates a rework loop where department must resubmit using OVERRIDE.
    Action items remain untouched (will be replaced on resubmission).
    
    Args:
        subcase_id: Subcase ID
        rejection_text: Administration's rejection explanation
        current_user: Current user object (must have user_id attribute)
    
    Raises:
        Exception: If user is None, subcase not found, or status invalid
    """
    if current_user is None:
        raise Exception("current_user cannot be None")
    
    subcase = _load_subcase_or_fail(subcase_id)
    _assert_status(subcase, ['DEPT_ACCEPTED_PENDING_ADMIN'])
    
    # Update rejection text
    administrative_subcase_db.update_administration_rejection(
        subcase_id=subcase_id,
        text=rejection_text,
        updated_by_user_id=current_user.user_id
    )
    
    # Return to department for revision (NOT terminal)
    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='RETURNED_TO_DEPT_FOR_REVISION',
        updated_by_user_id=current_user.user_id
    )
    
    # PARALLEL ACTION ITEM TRANSITION: SUBMITTED_TO_ADMIN -> ADMIN_REJECTED
    # Action items mirror the rejection; will be replaced on resubmission
    action_item_subcase_db.bulk_update_action_items_status_by_subcase(
        subcase_id=subcase_id,
        to_status='ADMIN_REJECTED',
        updated_by_user_id=current_user.user_id,
        from_statuses=['SUBMITTED_TO_ADMIN']
    )


def override_administration(
    subcase_id: int,
    explanation_text: str,
    action_items: List[Dict[str, Any]],
    current_user
) -> None:
    """
    Administration Administrator overrides action items with their own.
    
    Status transition: DEPT_ACCEPTED_PENDING_ADMIN -> ADMIN_APPROVED
    
    Args:
        subcase_id: Subcase ID
        explanation_text: Administration's explanation text
        action_items: List of replacement action items (DRAFT status)
            [{"title": str, "description": str, "due_date": str or None}, ...]
        current_user: Current user object (must have user_id attribute)
    
    Raises:
        Exception: If user is None, subcase not found, or status invalid
    """
    if current_user is None:
        raise Exception("current_user cannot be None")
    
    subcase = _load_subcase_or_fail(subcase_id)
    _assert_status(subcase, ['DEPT_ACCEPTED_PENDING_ADMIN'])
    
    # Replace all action items
    _replace_action_items(subcase_id, action_items, current_user)
    
    # Update administration explanation
    administrative_subcase_db.update_administration_explanation(
        subcase_id=subcase_id,
        text=explanation_text,
        updated_by_user_id=current_user.user_id
    )
    
    # Transition to final approval state
    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='ADMIN_APPROVED',
        updated_by_user_id=current_user.user_id
    )
    
    # PARALLEL ACTION ITEM TRANSITION: DRAFT -> ADMIN_APPROVED
    # Admin override creates new items as DRAFT, then approves them directly
    action_item_subcase_db.bulk_update_action_items_status_by_subcase(
        subcase_id=subcase_id,
        to_status='ADMIN_APPROVED',
        updated_by_user_id=current_user.user_id,
        from_statuses=['DRAFT']
    )


# ============================================================
# FORCE CLOSE
# ============================================================

def force_close_subcase(
    subcase_id: int,
    reason_text: str,
    current_user
) -> None:
    """
    Force close a subcase from any non-terminal state.
    Administrative override that bypasses normal workflow.
    
    Status transition: ANY (except terminal) -> FORCE_CLOSED (TERMINAL)
    
    Args:
        subcase_id: Subcase ID
        reason_text: Reason for force closing
        current_user: Current user object (must have user_id attribute)
    
    Raises:
        Exception: If user is None, subcase not found, or already in terminal state
    """
    if current_user is None:
        raise Exception("current_user cannot be None")
    
    subcase = _load_subcase_or_fail(subcase_id)
    
    # Check if already in a terminal closed state
    current_status = subcase.get('status')
    if current_status in ['CLOSED', 'FORCE_CLOSED']:
        raise Exception(
            f"Cannot force close subcase {subcase_id}: "
            f"already in terminal state '{current_status}'"
        )
    
    # Use the new force_close_subcase_with_tracking function
    updated = administrative_subcase_db.force_close_subcase_with_tracking(
        subcase_id=subcase_id,
        force_closed_by_user_id=current_user.user_id,
        force_close_reason=reason_text
    )
    if not updated:
        raise Exception(
            f"Failed to update subcase {subcase_id} in database. "
            "The record may have been modified by another user."
        )
    
    # PARALLEL ACTION ITEM TRANSITION: Cancel all non-final action items
    action_item_subcase_db.bulk_update_action_items_status_by_subcase(
        subcase_id=subcase_id,
        to_status='CANCELLED',
        updated_by_user_id=current_user.user_id
    )


def force_close_incident(
    incident_id: int,
    reason_text: str,
    current_user
) -> Dict[str, Any]:
    """
    Force close an incident and ALL its subcases.
    
    This is the main entry point for the force-close feature.
    Closes:
    1. All subcases (regardless of status)
    2. The main incident record (sets force_close tracking)
    
    Args:
        incident_id: Incident ID
        reason_text: Reason for force closing (min 10 chars)
        current_user: Current user object (must have user_id attribute)
    
    Returns:
        Dictionary with:
        - success: True
        - incident_id: Incident ID
        - incident_status: New status
        - subcases_closed: List of subcase IDs closed
        - total_subcases_closed: Count
        - closed_at: Timestamp
        - closed_by: Username
        - reason: Reason text
    
    Raises:
        Exception: If validation fails
    """
    if current_user is None:
        raise Exception("current_user cannot be None")
    
    # Validate reason length
    if not reason_text or len(reason_text) < 10:
        raise Exception("Reason is required and must be at least 10 characters")
    
    # Get all subcases for this incident
    subcases = administrative_subcase_db.get_subcases_by_incident(incident_id)
    
    # Close each subcase
    closed_subcase_ids = []
    for subcase in subcases:
        subcase_id = subcase['subcase_id']
        current_status = subcase.get('status')
        
        # Skip if already force closed (idempotent)
        if current_status == 'FORCE_CLOSED':
            closed_subcase_ids.append(subcase_id)
            continue
        
        # Force close the subcase
        administrative_subcase_db.force_close_subcase_with_tracking(
            subcase_id=subcase_id,
            force_closed_by_user_id=current_user.user_id,
            force_close_reason=reason_text
        )
        
        # PARALLEL ACTION ITEM TRANSITION: Cancel all non-final action items
        action_item_subcase_db.bulk_update_action_items_status_by_subcase(
            subcase_id=subcase_id,
            to_status='CANCELLED',
            updated_by_user_id=current_user.user_id
        )
        closed_subcase_ids.append(subcase_id)
    
    # Update the incident record with force_close tracking
    # (This requires adding a similar function in incident_case.py)
    from backend.api.db_layer import incident_case
    incident_case.update_force_close_tracking(
        incident_id=incident_id,
        force_closed_by_user_id=current_user.user_id,
        force_close_reason=reason_text
    )
    
    # Return result summary
    return {
        "success": True,
        "incident_id": incident_id,
        "incident_status": "FORCE_CLOSED",
        "subcases_closed": closed_subcase_ids,
        "total_subcases_closed": len(closed_subcase_ids),
        "closed_at": datetime.now().isoformat(),
        "closed_by": getattr(current_user, 'username', str(current_user.user_id)),
        "reason": reason_text
    }


# ============================================================
# UNIVERSAL SECTION: DIRECT APPROVAL TO ADMIN
# ============================================================

def direct_approve_to_admin(
    subcase_id: int,
    explanation_text: str,
    action_items: List[Dict[str, Any]],
    current_user,
    rca_feedback: Optional[Dict[str, Any]] = None
) -> None:
    """
    UNIVERSAL_SECTION direct approval: Submit response + approve directly to ADMIN_APPROVED.
    
    This is an operational bridge function that allows UNIVERSAL_SECTION role
    to bypass the normal multi-level workflow and approve directly.
    
    Status transition: SUBMITTED_TO_SECTION or RETURNED_TO_SECTION_FOR_REVISION -> ADMIN_APPROVED
    
    What this does:
    1. Validates user has UNIVERSAL_SECTION role
    2. Validates subcase is in appropriate status
    3. Sets the section explanation text
    4. Creates RCA feedback record (mandatory)
    5. Creates action items in ADMIN_APPROVED status
    6. Transitions subcase directly to ADMIN_APPROVED
    
    Args:
        subcase_id: Subcase ID
        explanation_text: Explanation text for the case
        action_items: List of action items to create
        current_user: Current user object (must have UNIVERSAL_SECTION role)
        rca_feedback: RCA feedback data with Cause_* and Preventive_* fields (mandatory)
    
    Raises:
        Exception: If user is None, not UNIVERSAL_SECTION, subcase not found, status invalid, or RCA missing
    """
    if current_user is None:
        raise Exception("current_user cannot be None")
    
    # RCA is mandatory for direct approval
    if rca_feedback is None:
        raise Exception("RCA feedback is required for direct approval")
    
    # Validate UNIVERSAL_SECTION role
    if not hasattr(current_user, 'scopes') or not current_user.scopes:
        raise Exception("User must have UNIVERSAL_SECTION role for direct approval")
    
    role_code = current_user.scopes[0].role_code
    if role_code != 'UNIVERSAL_SECTION':
        raise Exception(f"User must have UNIVERSAL_SECTION role for direct approval, got {role_code}")
    
    # Load subcase and validate status
    subcase = _load_subcase_or_fail(subcase_id)
    _assert_status(subcase, ['SUBMITTED_TO_SECTION', 'RETURNED_TO_SECTION_FOR_REVISION'])
    
    # Get incident ID for RCA linkage
    incident_id = subcase.get('incident_request_case_id')
    if not incident_id:
        raise Exception(f"Subcase {subcase_id} has no linked incident case")
    
    # Check if RCA already exists for this subcase (prevent duplicates)
    existing_rca = incident_case_feedback.get_rca_feedback_by_subcase(subcase_id)
    if existing_rca:
        raise Exception(f"RCA feedback already exists for subcase {subcase_id}. RCA cannot be edited.")
    
    # Set section explanation text
    if explanation_text:
        administrative_subcase_db.update_section_explanation(
            subcase_id=subcase_id,
            text=explanation_text,
            updated_by_user_id=current_user.user_id
        )
    
    # Create RCA feedback record linked to subcase
    incident_case_feedback.create_subcase_rca_feedback(
        subcase_id=subcase_id,
        incident_id=incident_id,
        feedback_data=rca_feedback,
        created_by_user_id=current_user.user_id
    )
    
    # Clear any existing action items for this subcase and create new ones
    existing_items = action_item_subcase_db.get_action_items_by_subcase(subcase_id)
    for item in existing_items:
        action_item_subcase_db.delete_action_item(item['action_item_id'])
    
    # Create new action items directly in ADMIN_APPROVED status
    for item in action_items:
        action_item_subcase_db.create_action_item(
            subcase_id=subcase_id,
            title=item['title'],
            description=item.get('description', ''),
            created_by_user_id=current_user.user_id,
            due_date=item.get('due_date'),
            initial_status='ADMIN_APPROVED'  # Skip workflow, go directly to approved
        )
    
    # Transition subcase directly to ADMIN_APPROVED
    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='ADMIN_APPROVED',
        updated_by_user_id=current_user.user_id
    )



