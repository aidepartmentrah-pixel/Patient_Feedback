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

from typing import List, Dict, Any
from backend.api_v2.db_layer import administrative_subcase_db
from backend.api_v2.db_layer import action_item_subcase_db


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
# SECTION-LEVEL ACTIONS
# ============================================================


def submit_section_response(
    subcase_id: int,
    explanation_text: str,
    action_items: List[Dict[str, Any]],
    current_user
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
    
    Raises:
        Exception: If user is None, subcase not found, or status invalid
    """
    if current_user is None:
        raise Exception("current_user cannot be None")
    
    subcase = _load_subcase_or_fail(subcase_id)
    
    # Allow both initial submission and resubmission after rejection
    _assert_status(subcase, ['SUBMITTED_TO_SECTION', 'RETURNED_TO_SECTION_FOR_REVISION'])
    
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


def reject_responsibility(
    subcase_id: int,
    rejection_text: str,
    current_user
) -> None:
    """
    Section Administrator rejects responsibility for the subcase.
    
    Status transition: SUBMITTED_TO_SECTION -> SECTION_DENIED (TERMINAL)
    
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
    _assert_status(subcase, ['SUBMITTED_TO_SECTION'])
    
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
# DEPARTMENT-LEVEL ACTIONS
# ============================================================

def approve_department(
    subcase_id: int,
    current_user
) -> None:
    """
    Department Administrator approves the section's response.
    
    Status transition: SECTION_ACCEPTED_PENDING_DEPT -> DEPT_ACCEPTED_PENDING_ADMIN
    
    Args:
        subcase_id: Subcase ID
        current_user: Current user object (must have user_id attribute)
    
    Raises:
        Exception: If user is None, subcase not found, or status invalid
    """
    if current_user is None:
        raise Exception("current_user cannot be None")
    
    subcase = _load_subcase_or_fail(subcase_id)
    _assert_status(subcase, ['SECTION_ACCEPTED_PENDING_DEPT'])
    
    # Transition to next workflow stage
    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='DEPT_ACCEPTED_PENDING_ADMIN',
        updated_by_user_id=current_user.user_id
    )


def reject_department(
    subcase_id: int,
    rejection_text: str,
    current_user
) -> None:
    """
    Department Administrator rejects the section's response.
    
    WORKFLOW CONTRACT CHANGE: Rejection is NOT terminal - it returns for revision.
    
    Status transition: SECTION_ACCEPTED_PENDING_DEPT -> RETURNED_TO_SECTION_FOR_REVISION
    
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
    _assert_status(subcase, ['SECTION_ACCEPTED_PENDING_DEPT'])
    
    # Update rejection text
    administrative_subcase_db.update_department_rejection(
        subcase_id=subcase_id,
        text=rejection_text,
        updated_by_user_id=current_user.user_id
    )
    
    # Return to section for revision (NOT terminal)
    # Action items remain untouched - will be replaced via override on resubmission
    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='RETURNED_TO_SECTION_FOR_REVISION',
        updated_by_user_id=current_user.user_id
    )


def override_department(
    subcase_id: int,
    explanation_text: str,
    action_items: List[Dict[str, Any]],
    current_user
) -> None:
    """
    Department Administrator overrides section's action items with their own.
    
    Status transition: SECTION_ACCEPTED_PENDING_DEPT -> DEPT_ACCEPTED_PENDING_ADMIN
    
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
    _assert_status(subcase, ['SECTION_ACCEPTED_PENDING_DEPT'])
    
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
    # Action items remain untouched - will be replaced via override on resubmission
    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='RETURNED_TO_DEPT_FOR_REVISION',
        updated_by_user_id=current_user.user_id
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
    
    # Update rejection text with reason
    administrative_subcase_db.update_administration_rejection(
        subcase_id=subcase_id,
        text=reason_text,
        updated_by_user_id=current_user.user_id
    )
    
    # Transition to force closed state
    administrative_subcase_db.update_subcase_status(
        subcase_id=subcase_id,
        new_status='FORCE_CLOSED',
        updated_by_user_id=current_user.user_id
    )


