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
from core.database import get_connection
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


def _upsert_action_items(
    subcase_id: int,
    action_items: List[Dict[str, Any]],
    current_user
) -> None:
    """
    Upsert action items for a subcase.
    - Items with action_item_id present → UPDATE (title, description, due_date)
    - Items without action_item_id → CREATE new
    - Existing items not referenced in the payload → DELETE

    This replaces _replace_action_items to preserve stable record identities.
    """
    existing = action_item_subcase_db.get_action_items_by_subcase(subcase_id)
    existing_ids = {item['action_item_id'] for item in existing}
    kept_ids = set()

    for item in action_items:
        aid = item.get('action_item_id')
        if aid and aid in existing_ids:
            action_item_subcase_db.update_action_item(
                action_item_id=aid,
                title=item['title'],
                description=item.get('description'),
                due_date=item.get('due_date'),
                updated_by_user_id=current_user.user_id
            )
            kept_ids.add(aid)
        else:
            action_item_subcase_db.create_action_item(
                subcase_id=subcase_id,
                title=item['title'],
                description=item.get('description'),
                created_by_user_id=current_user.user_id,
                due_date=item.get('due_date'),
                initial_status='DRAFT',
                assigned_to_user_id=item.get('assigned_to_user_id')
            )

    for eid in existing_ids - kept_ids:
        action_item_subcase_db.delete_action_item(eid)


def _replace_action_items(
    subcase_id: int,
    action_items: List[Dict[str, Any]],
    current_user
) -> None:
    """Kept for backward compatibility. Delegates to _upsert_action_items."""
    _upsert_action_items(subcase_id, action_items, current_user)


def _is_force_close_data_complete(subcase: Dict[str, Any]) -> bool:
    """
    Return True if all three workflow-level explanations are present.

    Used at force-close time to decide whether to land on
    FORCE_CLOSED_DRAFT (missing data) or FORCE_CLOSED_COMPLETE (all present).
    Also used by complete_force_closed_draft() to gate the transition.
    """
    return bool(
        subcase.get('section_explanation_text')
        and subcase.get('department_explanation_text')
        and subcase.get('administration_explanation_text')
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
    'FORCE_CLOSED_DRAFT',
    'FORCE_CLOSED_COMPLETE',
}

# All statuses that represent a force-closed case (any variant).
_FORCE_CLOSED_STATUSES = {'FORCE_CLOSED', 'FORCE_CLOSED_DRAFT', 'FORCE_CLOSED_COMPLETE'}


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
            "action_item_id": item.get('action_item_id'),
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


# Constant text written by all quick-accept paths
ACCEPT_TEXT = 'قبول الشكوى'


# ============================================================
# SECTION-LEVEL ACTIONS
# ============================================================


def accept_section_complaint(
    subcase_id: int,
    current_user
) -> None:
    """
    Section Administrator accepts complaint with a simple acknowledgement.

    No action items, no RCA required.
    Writes the constant text 'قبول الشكوى' to SectionExplanationText.

    Status transition:
    - SUBMITTED_TO_SECTION            -> SECTION_ACCEPTED_PENDING_DEPT
    - RETURNED_TO_SECTION_FOR_REVISION -> SECTION_ACCEPTED_PENDING_DEPT
    """
    if current_user is None:
        raise Exception("current_user cannot be None")

    subcase = _load_subcase_or_fail(subcase_id)
    _assert_status(subcase, ['SUBMITTED_TO_SECTION', 'RETURNED_TO_SECTION_FOR_REVISION'])

    administrative_subcase_db.update_section_explanation(
        subcase_id=subcase_id,
        text=ACCEPT_TEXT,
        updated_by_user_id=current_user.user_id
    )

    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='SECTION_ACCEPTED_PENDING_DEPT',
        updated_by_user_id=current_user.user_id
    )

    # Transition any DRAFT action items forward (normally zero on this path)
    action_item_subcase_db.bulk_update_action_items_status_by_subcase(
        subcase_id=subcase_id,
        to_status='SUBMITTED_TO_DEPT',
        updated_by_user_id=current_user.user_id,
        from_statuses=['DRAFT']
    )


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
    
    # RCA feedback is optional — new flow saves structured selections separately
    # via APP_SubcaseRCASuggestionSelection; old boolean feedback still accepted if provided
    
    subcase = _load_subcase_or_fail(subcase_id)
    
    # Allow both initial submission and resubmission after rejection
    _assert_status(subcase, ['SUBMITTED_TO_SECTION', 'RETURNED_TO_SECTION_FOR_REVISION'])
    
    # Get incident ID for RCA linkage
    incident_id = subcase.get('incident_request_case_id')
    if not incident_id:
        raise Exception(f"Subcase {subcase_id} has no linked incident case")
    
    # Check if RCA already exists for this incident (prevent duplicates)
    # Note: RCA table uses IncidentRequestCaseID as primary key, so only one RCA per incident
    existing_rca = incident_case_feedback.get_incident_case_feedback(incident_id)
    
    # Upsert action items — handles initial submission (all new) and
    # resubmission (RETURNED_TO_SECTION_FOR_REVISION, may have existing items).
    _upsert_action_items(subcase_id, action_items, current_user)
    
    # Save legacy boolean RCA feedback if provided (old flow kept for backward compat)
    if rca_feedback is not None and not existing_rca:
        try:
            incident_case_feedback.create_subcase_rca_feedback(
                subcase_id=subcase_id,
                incident_id=incident_id,
                feedback_data=rca_feedback,
                created_by_user_id=current_user.user_id
            )
        except Exception as e:
            if "duplicate key" not in str(e).lower() and "primary key" not in str(e).lower():
                raise
    
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
    _assert_status(subcase, ['SECTION_ACCEPTED_PENDING_DEPT', 'RETURNED_TO_DEPT_FOR_REVISION', 'FORCE_CLOSED_AT_SECTION'])

    # Write acceptance text so the field is never left empty
    administrative_subcase_db.update_department_explanation(
        subcase_id=subcase_id,
        text=ACCEPT_TEXT,
        updated_by_user_id=current_user.user_id
    )

    # Transition to next workflow stage
    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='DEPT_ACCEPTED_PENDING_ADMIN',
        updated_by_user_id=current_user.user_id
    )

    # PARALLEL ACTION ITEM TRANSITION: SUBMITTED_TO_DEPT -> SUBMITTED_TO_ADMIN
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
    _assert_status(subcase, ['SECTION_ACCEPTED_PENDING_DEPT', 'RETURNED_TO_DEPT_FOR_REVISION', 'FORCE_CLOSED_AT_SECTION'])

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
    _assert_status(subcase, ['SECTION_ACCEPTED_PENDING_DEPT', 'RETURNED_TO_DEPT_FOR_REVISION', 'FORCE_CLOSED_AT_SECTION'])

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

    Status transition for administrative complaint subcases (ClinicalRiskTypeID=1):
        DEPT_ACCEPTED_PENDING_ADMIN -> WAITING_PATIENT_SERVICES_DECISION
        Action items are left untouched (per spec).

    Status transition for all other case types (Notice, Improvement Opportunity,
    Seasonal Report):
        DEPT_ACCEPTED_PENDING_ADMIN -> ADMIN_APPROVED
        Action items: SUBMITTED_TO_ADMIN -> ADMIN_APPROVED (existing behaviour).

    Args:
        subcase_id: Subcase ID
        current_user: Current user object (must have user_id attribute)

    Raises:
        Exception: If user is None, subcase not found, or status invalid
    """
    if current_user is None:
        raise Exception("current_user cannot be None")

    subcase = _load_subcase_or_fail(subcase_id)
    _assert_status(subcase, ['DEPT_ACCEPTED_PENDING_ADMIN', 'FORCE_CLOSED_AT_DEPARTMENT'])

    # Detect whether this is an administrative complaint subcase.
    # ClinicalRiskTypeID == 1 means ordinary complaint (not Notice=2, not Never Event=3).
    # Seasonal-report subcases have no incident_request_case_id → is_complaint stays False.
    is_complaint = False
    incident_id = subcase.get('incident_request_case_id')
    if incident_id:
        from backend.api.db_layer import incident_case as incident_case_db
        incident = incident_case_db.get_incident_case_by_id(incident_id)
        if incident and incident.get('ClinicalRiskTypeID') == 1:
            is_complaint = True

    # Write acceptance text so the field is never left empty
    administrative_subcase_db.update_administration_explanation(
        subcase_id=subcase_id,
        text=ACCEPT_TEXT,
        updated_by_user_id=current_user.user_id
    )

    if is_complaint:
        # Complaint path: hand off to Patient Services for the scientific decision.
        # Action items remain at SUBMITTED_TO_ADMIN — do NOT transition them here.
        administrative_subcase_db.update_subcase_status(
            subcase_id=subcase_id,
            new_status='WAITING_PATIENT_SERVICES_DECISION',
            updated_by_user_id=current_user.user_id
        )
    else:
        # Non-complaint path (Notice / Improvement Opportunity / Seasonal Report):
        # go straight to ADMIN_APPROVED and advance action items as before.
        administrative_subcase_db.update_subcase_status(
            subcase_id=subcase_id,
            new_status='ADMIN_APPROVED',
            updated_by_user_id=current_user.user_id
        )
        # PARALLEL ACTION ITEM TRANSITION: SUBMITTED_TO_ADMIN -> ADMIN_APPROVED
        action_item_subcase_db.bulk_update_action_items_status_by_subcase(
            subcase_id=subcase_id,
            to_status='ADMIN_APPROVED',
            updated_by_user_id=current_user.user_id,
            from_statuses=['SUBMITTED_TO_ADMIN']
        )


def save_patient_services_decision(
    subcase_id: int,
    decision_text: str,
    current_user
) -> None:
    """
    Complaint Supervisor records the Patient Services scientific decision.

    قرار خدمات المرضى بحسب المراجع العلميّة

    Allowed on:
        WAITING_PATIENT_SERVICES_DECISION  (first save)
        PATIENT_SERVICES_DECISION_COMPLETED (re-edit — decision_at preserved)

    Status transition:
        * -> PATIENT_SERVICES_DECISION_COMPLETED

    Action items: untouched.

    Args:
        subcase_id:    Subcase ID
        decision_text: The decision text (must not be blank)
        current_user:  User object with user_id attribute

    Raises:
        Exception: If user is None, decision_text is blank, or status invalid
    """
    if current_user is None:
        raise Exception("current_user cannot be None")
    if not decision_text or not decision_text.strip():
        raise Exception("decision_text cannot be empty")

    subcase = _load_subcase_or_fail(subcase_id)
    _assert_status(subcase, [
        'WAITING_PATIENT_SERVICES_DECISION',
        'PATIENT_SERVICES_DECISION_COMPLETED',  # allow re-edit
    ])

    administrative_subcase_db.save_patient_services_decision(
        subcase_id=subcase_id,
        decision_text=decision_text.strip(),
        user_id=current_user.user_id
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
    _assert_status(subcase, ['DEPT_ACCEPTED_PENDING_ADMIN', 'FORCE_CLOSED_AT_DEPARTMENT'])

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

    Status transition for administrative complaint subcases (ClinicalRiskTypeID=1):
        DEPT_ACCEPTED_PENDING_ADMIN -> WAITING_PATIENT_SERVICES_DECISION
        Action items replaced but left at DRAFT (not transitioned to ADMIN_APPROVED).

    Status transition for all other case types:
        DEPT_ACCEPTED_PENDING_ADMIN -> ADMIN_APPROVED
        Action items replaced and transitioned DRAFT -> ADMIN_APPROVED (existing behaviour).

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
    _assert_status(subcase, ['DEPT_ACCEPTED_PENDING_ADMIN', 'FORCE_CLOSED_AT_DEPARTMENT'])

    # Detect complaint subcase (same logic as approve_administration)
    is_complaint = False
    incident_id = subcase.get('incident_request_case_id')
    if incident_id:
        from backend.api.db_layer import incident_case as incident_case_db
        incident = incident_case_db.get_incident_case_by_id(incident_id)
        if incident and incident.get('ClinicalRiskTypeID') == 1:
            is_complaint = True

    # Replace all action items (always — admin creates their own set)
    _replace_action_items(subcase_id, action_items, current_user)

    # Update administration explanation
    administrative_subcase_db.update_administration_explanation(
        subcase_id=subcase_id,
        text=explanation_text,
        updated_by_user_id=current_user.user_id
    )

    if is_complaint:
        # Complaint path: hand off to Patient Services.
        # Action items stay at DRAFT — do NOT transition them here.
        administrative_subcase_db.update_subcase_status(
            subcase_id=subcase_id,
            new_status='WAITING_PATIENT_SERVICES_DECISION',
            updated_by_user_id=current_user.user_id
        )
    else:
        # Non-complaint path: transition to final approval and advance action items.
        administrative_subcase_db.update_subcase_status(
            subcase_id=subcase_id,
            new_status='ADMIN_APPROVED',
            updated_by_user_id=current_user.user_id
        )
        # PARALLEL ACTION ITEM TRANSITION: DRAFT -> ADMIN_APPROVED
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
) -> str:
    """
    Force close a subcase from any non-terminal state.
    Administrative override that bypasses normal workflow.

    Status transition:
      ANY (except terminal) -> FORCE_CLOSED_DRAFT  (data incomplete)
      ANY (except terminal) -> FORCE_CLOSED_COMPLETE (all explanations present)

    Authorization: COMPLAINT_SUPERVISOR and WORKER only.

    Args:
        subcase_id: Subcase ID
        reason_text: Reason for force closing
        current_user: Current user object (must have user_id attribute)

    Returns:
        The new status string ('FORCE_CLOSED_DRAFT' or 'FORCE_CLOSED_COMPLETE')

    Raises:
        HTTPException(403): If user is not COMPLAINT_SUPERVISOR or WORKER
        Exception: If user is None, subcase not found, or already in terminal state
    """
    if current_user is None:
        raise Exception("current_user cannot be None")

    role_code = (
        current_user.scopes[0].role_code
        if current_user.scopes
        else None
    )
    if role_code not in ('COMPLAINT_SUPERVISOR', 'WORKER'):
        raise HTTPException(
            status_code=403,
            detail="Only COMPLAINT_SUPERVISOR or WORKER can force close cases."
        )

    subcase = _load_subcase_or_fail(subcase_id)

    current_status = subcase.get('status')
    if current_status in _FORCE_CLOSED_STATUSES or current_status == 'CLOSED':
        raise Exception(
            f"Cannot force close subcase {subcase_id}: "
            f"already in terminal state '{current_status}'"
        )

    new_status = (
        'FORCE_CLOSED_COMPLETE'
        if _is_force_close_data_complete(subcase)
        else 'FORCE_CLOSED_DRAFT'
    )

    updated = administrative_subcase_db.force_close_subcase_with_tracking(
        subcase_id=subcase_id,
        force_closed_by_user_id=current_user.user_id,
        force_close_reason=reason_text,
        new_status=new_status
    )
    if not updated:
        raise Exception(
            f"Failed to update subcase {subcase_id} in database. "
            "The record may have been modified by another user."
        )

    # Cancel all non-final action items
    action_item_subcase_db.bulk_update_action_items_status_by_subcase(
        subcase_id=subcase_id,
        to_status='CANCELLED',
        updated_by_user_id=current_user.user_id
    )

    return new_status


def force_close_incident(
    incident_id: int,
    reason_text: str,
    current_user
) -> Dict[str, Any]:
    """
    Force close an incident and ALL its subcases.

    This is the main entry point for the force-close feature.
    Closes:
    1. All subcases (regardless of status) → FORCE_CLOSED_DRAFT or FORCE_CLOSED_COMPLETE
    2. The main incident record (sets force_close tracking)

    Authorization: COMPLAINT_SUPERVISOR and WORKER only.

    Args:
        incident_id: Incident ID
        reason_text: Reason for force closing (min 10 chars)
        current_user: Current user object (must have user_id attribute)

    Returns:
        Dictionary with:
        - success: True
        - incident_id: Incident ID
        - subcases_closed: List of {subcase_id, new_status}
        - total_subcases_closed: Count
        - closed_at: Timestamp
        - closed_by: Username
        - reason: Reason text

    Raises:
        HTTPException(403): If user is not COMPLAINT_SUPERVISOR or WORKER
        Exception: If validation fails
    """
    if current_user is None:
        raise Exception("current_user cannot be None")

    role_code = (
        current_user.scopes[0].role_code
        if current_user.scopes
        else None
    )
    if role_code not in ('COMPLAINT_SUPERVISOR', 'WORKER'):
        raise HTTPException(
            status_code=403,
            detail="Only COMPLAINT_SUPERVISOR or WORKER can force close cases."
        )

    if not reason_text or len(reason_text) < 10:
        raise Exception("Reason is required and must be at least 10 characters")

    subcases = administrative_subcase_db.get_subcases_by_incident(incident_id)

    closed_subcase_ids = []
    for subcase in subcases:
        subcase_id = subcase['subcase_id']
        current_status = subcase.get('status')

        # Idempotent: already in a force-closed state — record it but skip DB write
        if current_status in _FORCE_CLOSED_STATUSES:
            closed_subcase_ids.append({
                "subcase_id": subcase_id,
                "new_status": current_status
            })
            continue

        new_status = (
            'FORCE_CLOSED_COMPLETE'
            if _is_force_close_data_complete(subcase)
            else 'FORCE_CLOSED_DRAFT'
        )

        administrative_subcase_db.force_close_subcase_with_tracking(
            subcase_id=subcase_id,
            force_closed_by_user_id=current_user.user_id,
            force_close_reason=reason_text,
            new_status=new_status
        )

        action_item_subcase_db.bulk_update_action_items_status_by_subcase(
            subcase_id=subcase_id,
            to_status='CANCELLED',
            updated_by_user_id=current_user.user_id
        )
        closed_subcase_ids.append({
            "subcase_id": subcase_id,
            "new_status": new_status
        })

    from backend.api.db_layer import incident_case
    incident_case.update_force_close_tracking(
        incident_id=incident_id,
        force_closed_by_user_id=current_user.user_id,
        force_close_reason=reason_text
    )

    return {
        "success": True,
        "incident_id": incident_id,
        "subcases_closed": closed_subcase_ids,
        "total_subcases_closed": len(closed_subcase_ids),
        "closed_at": datetime.now().isoformat(),
        "closed_by": getattr(current_user, 'username', str(current_user.user_id)),
        "reason": reason_text
    }


# ============================================================
# FORCE CLOSE: DRAFT → COMPLETE TRANSITION
# ============================================================

def complete_force_closed_draft(
    subcase_id: int,
    current_user
) -> None:
    """
    Transition a FORCE_CLOSED_DRAFT subcase to FORCE_CLOSED_COMPLETE.

    Called after all required data has been filled in by COMPLAINT_SUPERVISOR
    or WORKER via the manual intervention API (Session 3).

    Authorization: COMPLAINT_SUPERVISOR and WORKER only.

    Status transition: FORCE_CLOSED_DRAFT -> FORCE_CLOSED_COMPLETE

    Args:
        subcase_id: Subcase ID
        current_user: Current user object (must have user_id attribute)

    Raises:
        HTTPException(403): If user is not COMPLAINT_SUPERVISOR or WORKER
        Exception: If subcase not found, not in FORCE_CLOSED_DRAFT, or data still incomplete
    """
    if current_user is None:
        raise Exception("current_user cannot be None")

    role_code = (
        current_user.scopes[0].role_code
        if current_user.scopes
        else None
    )
    if role_code not in ('COMPLAINT_SUPERVISOR', 'WORKER'):
        raise HTTPException(
            status_code=403,
            detail="Only COMPLAINT_SUPERVISOR or WORKER can complete a force-closed draft."
        )

    subcase = _load_subcase_or_fail(subcase_id)
    _assert_status(subcase, ['FORCE_CLOSED_DRAFT'])

    if not _is_force_close_data_complete(subcase):
        raise Exception(
            f"Subcase {subcase_id} is still missing required explanation data. "
            "All three levels (section, department, administration) must be filled "
            "before transitioning to FORCE_CLOSED_COMPLETE."
        )

    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='FORCE_CLOSED_COMPLETE',
        updated_by_user_id=current_user.user_id
    )


# ============================================================
# MANUAL INTERVENTION: FILL ON BEHALF
# ============================================================

# Statuses where no further data filling is meaningful.
_FILL_BLOCKED_STATUSES = {'FORCE_CLOSED_COMPLETE', 'ADMIN_APPROVED', 'SECTION_DENIED'}

# Maps level name to (entered_for_role, db_fill_fn)
_LEVEL_CONFIG = {
    'section': {
        'entered_for_role': 'SECTION_ADMIN',
        'db_fn': lambda: administrative_subcase_db.fill_section_on_behalf,
    },
    'department': {
        'entered_for_role': 'DEPARTMENT_ADMIN',
        'db_fn': lambda: administrative_subcase_db.fill_department_on_behalf,
    },
    'administration': {
        'entered_for_role': 'ADMINISTRATION_ADMIN',
        'db_fn': lambda: administrative_subcase_db.fill_administration_on_behalf,
    },
}


def fill_explanation_on_behalf(
    subcase_id: int,
    level: str,
    explanation_text: str,
    current_user,
    action_items: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Fill a section/department/administration explanation on behalf of the
    role that normally owns that level.

    Allowed callers: COMPLAINT_SUPERVISOR and WORKER.
    Works on both active subcases and FORCE_CLOSED_DRAFT subcases.
    Overwrite is always permitted for these roles.

    Sequential unlock (data-driven):
      - department: section_explanation_text must already be present
      - administration: department_explanation_text must already be present

    entry_mode returned:
      - 'FORCE_CLOSE_INTERVENTION' when subcase is in a force-closed status
      - 'ON_BEHALF' otherwise

    For ON_BEHALF mode only, the workflow status is advanced after the fill:
      - section fill  → SECTION_ACCEPTED_PENDING_DEPT (+ action_items DRAFT→SUBMITTED_TO_DEPT)
      - department fill → DEPT_ACCEPTED_PENDING_ADMIN  (+ action_items SUBMITTED_TO_DEPT→SUBMITTED_TO_ADMIN)
      - administration fill (ClinicalRiskTypeID == 1, complaint) → WAITING_PATIENT_SERVICES_DECISION
        (action_items left untouched at SUBMITTED_TO_ADMIN, per approve_administration())
      - administration fill (all other case types) → ADMIN_APPROVED (+ action_items SUBMITTED_TO_ADMIN→ADMIN_APPROVED)

    action_items is only used at the section level in ON_BEHALF mode.

    Args:
        subcase_id: Subcase ID
        level: 'section', 'department', or 'administration'
        explanation_text: Text to write
        current_user: Authenticated user
        action_items: Optional list of action items for section fill
            [{"title": str, "description": str, "due_date": str|None, "assigned_to_user_id": int|None}, ...]

    Returns:
        entry_mode string ('ON_BEHALF' or 'FORCE_CLOSE_INTERVENTION')

    Raises:
        HTTPException(403): Wrong role
        HTTPException(404): Subcase not found
        HTTPException(400): Subcase is in a terminal state, sequential unlock
                            violation, unknown level, or DB write failed
    """
    if current_user is None:
        raise HTTPException(status_code=403, detail="Not authenticated")

    role_code = (
        current_user.scopes[0].role_code
        if current_user.scopes
        else None
    )
    if role_code not in ('COMPLAINT_SUPERVISOR', 'WORKER'):
        raise HTTPException(
            status_code=403,
            detail="Only COMPLAINT_SUPERVISOR or WORKER can fill data on behalf of other roles."
        )

    if level not in _LEVEL_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown level '{level}'. Must be 'section', 'department', or 'administration'."
        )

    subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
    if subcase is None:
        raise HTTPException(status_code=404, detail=f"Subcase {subcase_id} not found")

    current_status = subcase.get('status')
    if current_status in _FILL_BLOCKED_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Subcase {subcase_id} is in status '{current_status}' — "
                "no further data can be filled."
            )
        )

    # Sequential unlock
    if level == 'department' and not subcase.get('section_explanation_text'):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot fill department data on subcase {subcase_id}: "
                "section explanation must be filled first."
            )
        )
    if level == 'administration' and not subcase.get('department_explanation_text'):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot fill administration data on subcase {subcase_id}: "
                "department explanation must be filled first."
            )
        )

    entry_mode = (
        'FORCE_CLOSE_INTERVENTION'
        if current_status in _FORCE_CLOSED_STATUSES
        else 'ON_BEHALF'
    )

    cfg = _LEVEL_CONFIG[level]
    db_fn = cfg['db_fn']()
    updated = db_fn(
        subcase_id=subcase_id,
        text=explanation_text,
        entered_by_user_id=current_user.user_id,
        entered_for_role=cfg['entered_for_role'],
        entry_mode=entry_mode
    )
    if not updated:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to update subcase {subcase_id}. The record may not exist."
        )

    # For active workflow cases only: advance status and mirror action item transitions.
    # Guards ensure we only advance forward — never regress an already-progressed case.
    if entry_mode == 'FORCE_CLOSE_INTERVENTION':
        # Force-close fills do not advance workflow status or transition action items.
        # Action items are created at DRAFT and remain there for follow-up tracking.
        if level == 'section' and action_items:
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

    elif entry_mode == 'ON_BEHALF':
        if level == 'section' and current_status in (
            'SUBMITTED_TO_SECTION', 'RETURNED_TO_SECTION_FOR_REVISION'
        ):
            if action_items:
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
            action_item_subcase_db.bulk_update_action_items_status_by_subcase(
                subcase_id=subcase_id,
                to_status='SUBMITTED_TO_DEPT',
                updated_by_user_id=current_user.user_id,
                from_statuses=['DRAFT']
            )
            administrative_subcase_db.update_subcase_status(
                subcase_id=subcase_id,
                new_status='SECTION_ACCEPTED_PENDING_DEPT',
                updated_by_user_id=current_user.user_id
            )
        elif level == 'department' and current_status in (
            'SECTION_ACCEPTED_PENDING_DEPT', 'RETURNED_TO_DEPT_FOR_REVISION'
        ):
            action_item_subcase_db.bulk_update_action_items_status_by_subcase(
                subcase_id=subcase_id,
                to_status='SUBMITTED_TO_ADMIN',
                updated_by_user_id=current_user.user_id,
                from_statuses=['SUBMITTED_TO_DEPT']
            )
            administrative_subcase_db.update_subcase_status(
                subcase_id=subcase_id,
                new_status='DEPT_ACCEPTED_PENDING_ADMIN',
                updated_by_user_id=current_user.user_id
            )
        elif level == 'administration' and current_status == 'DEPT_ACCEPTED_PENDING_ADMIN':
            # Mirror approve_administration()'s complaint-routing rule:
            # ClinicalRiskTypeID == 1 (ordinary complaint) hands off to Patient
            # Services for the scientific decision instead of going terminal.
            is_complaint = False
            incident_id = subcase.get('incident_request_case_id')
            if incident_id:
                from backend.api.db_layer import incident_case as incident_case_db
                incident = incident_case_db.get_incident_case_by_id(incident_id)
                if incident and incident.get('ClinicalRiskTypeID') == 1:
                    is_complaint = True

            if is_complaint:
                # Complaint path: hand off to Patient Services for the scientific
                # decision. Action items remain at SUBMITTED_TO_ADMIN — do NOT
                # transition them here.
                administrative_subcase_db.update_subcase_status(
                    subcase_id=subcase_id,
                    new_status='WAITING_PATIENT_SERVICES_DECISION',
                    updated_by_user_id=current_user.user_id
                )
            else:
                action_item_subcase_db.bulk_update_action_items_status_by_subcase(
                    subcase_id=subcase_id,
                    to_status='ADMIN_APPROVED',
                    updated_by_user_id=current_user.user_id,
                    from_statuses=['SUBMITTED_TO_ADMIN']
                )
                administrative_subcase_db.update_subcase_status(
                    subcase_id=subcase_id,
                    new_status='ADMIN_APPROVED',
                    updated_by_user_id=current_user.user_id
                )

    return entry_mode


# ============================================================
# GIVE MORE TIME WORKFLOW (HCAT Automatic Force Close Policy - Session 5)
# ============================================================

# Permission matrix (Session 5): which roles may restore each force-closed
# level. COMPLAINT_SUPERVISOR and SOFTWARE_ADMIN can give more time at any
# level.
_GIVE_SECTION_MORE_TIME_ROLES = ('DEPARTMENT_ADMIN', 'COMPLAINT_SUPERVISOR', 'SOFTWARE_ADMIN')
_GIVE_DEPARTMENT_MORE_TIME_ROLES = ('ADMINISTRATION_ADMIN', 'COMPLAINT_SUPERVISOR', 'SOFTWARE_ADMIN')
# Administration is the top operational level - no ADMINISTRATION-level role
# sits above it, so only the hospital-wide roles can restore it.
_GIVE_ADMINISTRATION_MORE_TIME_ROLES = ('COMPLAINT_SUPERVISOR', 'SOFTWARE_ADMIN')

# Fallback deadline windows (days). Mirrors
# case_creation_service._DEFAULT_DEADLINE_DAYS (duplicated per the Session
# 3/4 convention of not sharing this small helper across service files).
_GIVE_MORE_TIME_DEFAULT_DEADLINE_DAYS = {
    "section": 10,
    "department": 7,
    "administration": 7,
}


def _get_deadline_days_settings(cursor) -> dict:
    """
    Read section/department/administration_deadline_days from
    APP_SystemSettings using the caller's open cursor, falling back to
    _GIVE_MORE_TIME_DEFAULT_DEADLINE_DAYS for missing/invalid values.
    """
    keys_by_setting = {
        "section_deadline_days": "section",
        "department_deadline_days": "department",
        "administration_deadline_days": "administration",
    }

    result = dict(_GIVE_MORE_TIME_DEFAULT_DEADLINE_DAYS)

    cursor.execute(
        "SELECT SettingKey, SettingValue FROM dbo.APP_SystemSettings WHERE SettingKey IN (?, ?, ?)",
        tuple(keys_by_setting.keys())
    )
    for row in cursor.fetchall():
        level = keys_by_setting.get(row.SettingKey)
        if level is None:
            continue
        try:
            result[level] = int(row.SettingValue)
        except (TypeError, ValueError):
            pass

    return result


def _require_give_more_time_role(current_user, allowed_roles, level_label: str) -> str:
    """
    Role check shared by give_*_more_time(). Returns the caller's primary
    role code.

    Raises:
        HTTPException(403): If current_user has no scopes or an unauthorized role.
    """
    role_code = (
        current_user.scopes[0].role_code
        if current_user and current_user.scopes
        else None
    )
    if role_code not in allowed_roles:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Insufficient permissions. Only "
                f"{', '.join(allowed_roles)} can give {level_label} more time. "
                f"Your role: {role_code}"
            )
        )
    return role_code


def _check_give_more_time_scope(subcase: Dict[str, Any], current_user) -> None:
    """
    Scope check shared by give_*_more_time(). Mirrors reopen_denied_case():
    the subcase's TargetOrgUnitID must be within the granting user's
    allowed_unit_ids (COMPLAINT_SUPERVISOR/SOFTWARE_ADMIN have full access;
    DEPARTMENT_ADMIN/ADMINISTRATION_ADMIN are limited to their org subtree).

    Raises:
        HTTPException(403): If the subcase is outside the user's scope.
    """
    target_org_unit_id = subcase.get('target_org_unit_id')
    allowed_unit_ids = getattr(current_user, 'allowed_unit_ids', None) or set()
    if target_org_unit_id not in allowed_unit_ids:
        raise HTTPException(
            status_code=403,
            detail="Subcase is outside your organizational scope."
        )


def give_section_more_time(subcase_id: int, current_user) -> Dict[str, Any]:
    """
    Restore a Section-force-closed subcase to active Section responsibility,
    giving the Section a fresh deadline.

    Status transition: FORCE_CLOSED_AT_SECTION -> SUBMITTED_TO_SECTION

    - SectionDeadlineAt is restarted to now + section_deadline_days (read
      live from APP_SystemSettings).
    - SectionExtraTimeGrantedAt / SectionExtraTimeGrantedBy record who
      granted the extension and when.
    - SectionForceClosedAt and SectionLateReply are preserved unchanged -
      the force-close history is permanent.
    - The Section's auto-inserted explanation text (if any, tagged
      SectionEntryMode='AUTO_FORCE_CLOSE' by Session 4) remains editable via
      the normal SUBMIT_RESPONSE action once Status is SUBMITTED_TO_SECTION.

    Authorization: DEPARTMENT_ADMIN, COMPLAINT_SUPERVISOR, SOFTWARE_ADMIN
    (subcase must be within the granting user's organizational scope).

    Args:
        subcase_id: Subcase ID
        current_user: Current user object (must have user_id, scopes, allowed_unit_ids)

    Returns:
        Updated per-level deadline state (see
        administrative_subcase_db.get_subcase_deadline_state)

    Raises:
        HTTPException(403): Unauthorized role or out-of-scope subcase
        Exception: Subcase not found, not currently FORCE_CLOSED_AT_SECTION
            (including the case where it was already restored)
    """
    if current_user is None:
        raise Exception("current_user cannot be None")

    _require_give_more_time_role(current_user, _GIVE_SECTION_MORE_TIME_ROLES, "Section")

    subcase = _load_subcase_or_fail(subcase_id)
    _check_give_more_time_scope(subcase, current_user)
    _assert_status(subcase, ['FORCE_CLOSED_AT_SECTION'])

    conn = get_connection()
    cursor = conn.cursor()
    try:
        deadline_days = _get_deadline_days_settings(cursor)['section']
    finally:
        cursor.close()
        conn.close()

    restored = administrative_subcase_db.give_section_more_time(
        subcase_id=subcase_id,
        granted_by_user_id=current_user.user_id,
        deadline_days=deadline_days
    )
    if not restored:
        raise Exception(
            f"Subcase {subcase_id} could not be restored - it is no longer "
            f"force-closed at the Section level."
        )

    return administrative_subcase_db.get_subcase_deadline_state(subcase_id)


def give_department_more_time(subcase_id: int, current_user) -> Dict[str, Any]:
    """
    Restore a Department-force-closed subcase to active Department
    responsibility, giving the Department a fresh deadline.

    Status transition: FORCE_CLOSED_AT_DEPARTMENT -> SECTION_ACCEPTED_PENDING_DEPT

    - DepartmentDeadlineAt is restarted to now + department_deadline_days
      (read live from APP_SystemSettings).
    - DepartmentExtraTimeGrantedAt / DepartmentExtraTimeGrantedBy record who
      granted the extension and when.
    - DepartmentForceClosedAt and DepartmentLateReply are preserved unchanged
      - the force-close history is permanent.
    - The Department's auto-inserted explanation text (if any, tagged
      DepartmentEntryMode='AUTO_FORCE_CLOSE' by Session 4) remains editable
      via the normal APPROVE/REJECT/OVERRIDE actions once Status is
      SECTION_ACCEPTED_PENDING_DEPT.

    Authorization: ADMINISTRATION_ADMIN, COMPLAINT_SUPERVISOR, SOFTWARE_ADMIN
    (subcase must be within the granting user's organizational scope).

    Args:
        subcase_id: Subcase ID
        current_user: Current user object (must have user_id, scopes, allowed_unit_ids)

    Returns:
        Updated per-level deadline state (see
        administrative_subcase_db.get_subcase_deadline_state)

    Raises:
        HTTPException(403): Unauthorized role or out-of-scope subcase
        Exception: Subcase not found, not currently FORCE_CLOSED_AT_DEPARTMENT
            (including the case where it was already restored)
    """
    if current_user is None:
        raise Exception("current_user cannot be None")

    _require_give_more_time_role(current_user, _GIVE_DEPARTMENT_MORE_TIME_ROLES, "Department")

    subcase = _load_subcase_or_fail(subcase_id)
    _check_give_more_time_scope(subcase, current_user)
    _assert_status(subcase, ['FORCE_CLOSED_AT_DEPARTMENT'])

    conn = get_connection()
    cursor = conn.cursor()
    try:
        deadline_days = _get_deadline_days_settings(cursor)['department']
    finally:
        cursor.close()
        conn.close()

    restored = administrative_subcase_db.give_department_more_time(
        subcase_id=subcase_id,
        granted_by_user_id=current_user.user_id,
        deadline_days=deadline_days
    )
    if not restored:
        raise Exception(
            f"Subcase {subcase_id} could not be restored - it is no longer "
            f"force-closed at the Department level."
        )

    return administrative_subcase_db.get_subcase_deadline_state(subcase_id)


def give_administration_more_time(subcase_id: int, current_user) -> Dict[str, Any]:
    """
    Restore an Administration-force-closed subcase to active Administration
    responsibility, giving Administration a fresh deadline.

    Status transition: FORCE_CLOSED_AT_ADMINISTRATION -> DEPT_ACCEPTED_PENDING_ADMIN

    - AdministrationDeadlineAt is restarted to now +
      administration_deadline_days (read live from APP_SystemSettings).
    - AdministrationExtraTimeGrantedAt / AdministrationExtraTimeGrantedBy
      record who granted the extension and when.
    - AdministrationForceClosedAt and AdministrationLateReply are preserved
      unchanged - the force-close history is permanent.
    - The Administration's auto-inserted explanation text (if any, tagged
      AdministrationEntryMode='AUTO_FORCE_CLOSE' by Session 4) remains
      editable via the normal APPROVE/REJECT/OVERRIDE actions once Status is
      DEPT_ACCEPTED_PENDING_ADMIN.

    Authorization: COMPLAINT_SUPERVISOR, SOFTWARE_ADMIN only. Administration
    is the top operational level - no ADMINISTRATION-level role sits above it
    in the normal hierarchy, so only the hospital-wide roles (which the
    Session 5 spec allows to give more time "to any level") can restore it.

    Args:
        subcase_id: Subcase ID
        current_user: Current user object (must have user_id, scopes, allowed_unit_ids)

    Returns:
        Updated per-level deadline state (see
        administrative_subcase_db.get_subcase_deadline_state)

    Raises:
        HTTPException(403): Unauthorized role or out-of-scope subcase
        Exception: Subcase not found, not currently FORCE_CLOSED_AT_ADMINISTRATION
            (including the case where it was already restored)
    """
    if current_user is None:
        raise Exception("current_user cannot be None")

    _require_give_more_time_role(current_user, _GIVE_ADMINISTRATION_MORE_TIME_ROLES, "Administration")

    subcase = _load_subcase_or_fail(subcase_id)
    _check_give_more_time_scope(subcase, current_user)
    _assert_status(subcase, ['FORCE_CLOSED_AT_ADMINISTRATION'])

    conn = get_connection()
    cursor = conn.cursor()
    try:
        deadline_days = _get_deadline_days_settings(cursor)['administration']
    finally:
        cursor.close()
        conn.close()

    restored = administrative_subcase_db.give_administration_more_time(
        subcase_id=subcase_id,
        granted_by_user_id=current_user.user_id,
        deadline_days=deadline_days
    )
    if not restored:
        raise Exception(
            f"Subcase {subcase_id} could not be restored - it is no longer "
            f"force-closed at the Administration level."
        )

    return administrative_subcase_db.get_subcase_deadline_state(subcase_id)




