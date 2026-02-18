"""
Follow-Up Service — Phase 3 Workflow (API v2)

This service provides access to action items assigned to users.
Action items can come from either incident subcases or seasonal report subcases.

This service is case-type agnostic and applies Phase 2.5 scope filtering.

SCOPE MODEL:
- Users can only see action items for subcases within their allowed_unit_ids
- Action items are linked to subcases via SubcaseID
- Subcases have TargetOrgUnitID which determines scope

PERMISSION MODEL:
- Scope is enforced FIRST (Phase 2.5 allowed_unit_ids)
- Then ownership OR role override is checked
- Allowed to modify if: assigned user OR privileged role (within scope)

Structure:
- Public functions: get_action_items_for_user, start_action_item, complete_action_item, delay_action_item
- Internal helpers: _assert_user_can_modify
"""

from typing import List, Dict, Any
from backend.api_v2.db_layer import action_item_subcase_db, administrative_subcase_db


# =============================================================================
# EXCEPTIONS
# =============================================================================

class Unauthorized(Exception):
    """Raised when user is not authenticated or authorized."""
    pass


class NotFound(Exception):
    """Raised when a requested resource is not found."""
    pass


class Forbidden(Exception):
    """Raised when user is authenticated but not authorized to access a resource."""
    pass


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _assert_user_can_modify(action_item: Dict[str, Any], subcase: Dict[str, Any], current_user):
    """
    Assert that the current user has permission to modify an action item.
    
    Permission model:
    1. SCOPE IS ENFORCED FIRST (Phase 2.5 allowed_unit_ids)
    2. Then check: user is assigned OR user has privileged role
    
    Args:
        action_item: Action item dictionary from DB layer
        subcase: Subcase dictionary from DB layer (for target_org_unit_id)
        current_user: User object with user_id, allowed_unit_ids, and role
        
    Raises:
        NotFound: If action_item is None
        Unauthorized: If user is None or user_id is None
        Forbidden: If user is out of scope or lacks permission
    """
    if action_item is None:
        raise NotFound("Action item not found")
    
    # Authentication check
    if current_user is None or not hasattr(current_user, 'user_id') or current_user.user_id is None:
        raise Unauthorized("User must be authenticated to modify action items")
    
    # PHASE 2.5 SCOPE CHECK FIRST
    target_org_unit_id = subcase.get("target_org_unit_id") if subcase else None
    allowed_unit_ids = getattr(current_user, 'allowed_unit_ids', None)
    
    if not allowed_unit_ids or not target_org_unit_id or target_org_unit_id not in allowed_unit_ids:
        raise Forbidden(f"Action item is outside user's organizational scope")
    
    # PERMISSION CHECK: Assigned user OR privileged role
    assigned_to_user_id = action_item.get("assigned_to_user_id")
    
    # Get role from scopes (CurrentUser stores roles in scopes[0].role_code)
    user_role = None
    if hasattr(current_user, 'scopes') and current_user.scopes:
        user_role = current_user.scopes[0].role_code
    
    # Allow if user is assigned to this action item
    if assigned_to_user_id is not None and assigned_to_user_id == current_user.user_id:
        return
    
    # Allow if user has privileged role (use actual role codes from the system)
    privileged_roles = [
        "COMPLAINT_SUPERVISOR", "SOFTWARE_ADMIN",
        "SECTION_ADMIN", "DEPARTMENT_ADMIN", "ADMINISTRATION_ADMIN"
    ]
    if user_role in privileged_roles:
        return
    
    # Otherwise, forbidden
    raise Forbidden(f"User is not assigned to this action item and does not have a privileged role")


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

def get_action_items_for_user(current_user) -> List[Dict[str, Any]]:
    """
    Get all actionable items within the current user's organizational scope.
    
    This function:
    1. Uses the user's allowed_unit_ids (Phase 2.5 scope) to find action items
       whose subcase TargetOrgUnitID falls within scope
    2. Returns ADMIN_APPROVED and IN_PROGRESS items (actionable statuses)
    3. Also includes any items explicitly assigned to the user regardless of scope
    
    This approach ensures that:
    - Admin/dept/section users see all follow-up items in their org scope
    - Items without AssignedToUserID still appear (common case)
    - Scope filtering is enforced via SQL JOIN for efficiency
    
    Args:
        current_user: User object with user_id and allowed_unit_ids attributes
        
    Returns:
        List of action item dictionaries (filtered by scope)
        
    Raises:
        Unauthorized: If user is None or user_id is None
    """
    # Authentication check
    if current_user is None or not hasattr(current_user, 'user_id') or current_user.user_id is None:
        raise Unauthorized("User must be authenticated to access action items")
    
    # Get allowed unit IDs from Phase 2.5 Scope Engine
    allowed_unit_ids = getattr(current_user, 'allowed_unit_ids', None)
    
    # If no allowed_unit_ids, user has no scope - return empty
    if not allowed_unit_ids:
        return []
    
    # Query action items by scope — single efficient SQL JOIN
    # Only returns ADMIN_APPROVED and IN_PROGRESS items (actionable_only=True)
    scoped_items = action_item_subcase_db.get_action_items_by_scope(
        allowed_unit_ids=list(allowed_unit_ids),
        actionable_only=True
    )
    
    # Also fetch items explicitly assigned to this user (may be out of scope)
    assigned_items = action_item_subcase_db.get_action_items_by_assigned_user(current_user.user_id)
    
    # Merge: use scoped items as base, add any assigned items not already included
    seen_ids = {item["action_item_id"] for item in scoped_items}
    merged = list(scoped_items)
    for item in assigned_items:
        if item["action_item_id"] not in seen_ids:
            merged.append(item)
    
    return merged


def start_action_item(action_item_id: int, current_user) -> bool:
    """
    Mark an action item as started (sets StartedAt timestamp).
    
    This function:
    1. Loads the action item and its associated subcase
    2. Enforces permission using _assert_user_can_modify
    3. Calls DB layer to set the started timestamp
    
    Args:
        action_item_id: ID of the action item to start
        current_user: User object with user_id, allowed_unit_ids, and role attributes
        
    Returns:
        True if successful, False if action item not found in DB
        
    Raises:
        Unauthorized: If user is None or user_id is None
        NotFound: If action item does not exist
        Forbidden: If user lacks permission to modify
    """
    # Load action item from DB
    action_item = action_item_subcase_db.get_action_item_by_id(action_item_id)
    
    if action_item is None:
        raise NotFound(f"Action item {action_item_id} not found")
    
    # Load subcase for scope checking
    subcase_id = action_item["subcase_id"]
    subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
    
    if subcase is None:
        raise NotFound(f"Subcase {subcase_id} not found for action item {action_item_id}")
    
    # WORKFLOW CONTRACT GUARD: Block execution if subcase is in revision state
    subcase_status = subcase.get("status")
    if subcase_status in ['RETURNED_TO_SECTION_FOR_REVISION', 'RETURNED_TO_DEPT_FOR_REVISION']:
        raise Exception(
            f"Cannot start action item: subcase is returned for revision (status: {subcase_status}). "
            "Wait for resubmission with corrected response."
        )
    
    # WORKFLOW CONTRACT GUARD: Action item must be ADMIN_APPROVED before it can be started
    action_item_status = action_item.get("status")
    if action_item_status != 'ADMIN_APPROVED':
        raise Exception(
            f"Cannot start action item: status is '{action_item_status}' but must be 'ADMIN_APPROVED'. "
            "Action items can only be started after administration approval."
        )
    
    # Enforce permission (scope + ownership/role)
    _assert_user_can_modify(action_item, subcase, current_user)
    
    # Execute the mutation via DB layer
    success = action_item_subcase_db.set_action_item_started(action_item_id, current_user.user_id)
    
    return success


def complete_action_item(action_item_id: int, current_user) -> bool:
    """
    Mark an action item as completed (sets CompletedAt timestamp).
    
    This function:
    1. Loads the action item and its associated subcase
    2. Enforces permission using _assert_user_can_modify
    3. Calls DB layer to set the completed timestamp
    
    Args:
        action_item_id: ID of the action item to complete
        current_user: User object with user_id, allowed_unit_ids, and role attributes
        
    Returns:
        True if successful, False if action item not found in DB
        
    Raises:
        Unauthorized: If user is None or user_id is None
        NotFound: If action item does not exist
        Forbidden: If user lacks permission to modify
    """
    # Load action item from DB
    action_item = action_item_subcase_db.get_action_item_by_id(action_item_id)
    
    if action_item is None:
        raise NotFound(f"Action item {action_item_id} not found")
    
    # Load subcase for scope checking
    subcase_id = action_item["subcase_id"]
    subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
    
    if subcase is None:
        raise NotFound(f"Subcase {subcase_id} not found for action item {action_item_id}")
    
    # WORKFLOW CONTRACT GUARD: Block execution if subcase is in revision state
    subcase_status = subcase.get("status")
    if subcase_status in ['RETURNED_TO_SECTION_FOR_REVISION', 'RETURNED_TO_DEPT_FOR_REVISION']:
        raise Exception(
            f"Cannot complete action item: subcase is returned for revision (status: {subcase_status}). "
            "Wait for resubmission with corrected response."
        )
    
    # Enforce permission (scope + ownership/role)
    _assert_user_can_modify(action_item, subcase, current_user)
    
    # Execute the mutation via DB layer
    success = action_item_subcase_db.set_action_item_completed(action_item_id, current_user.user_id)
    
    return success


def delay_action_item(action_item_id: int, delay_days: int, current_user) -> Dict[str, Any]:
    """
    Delay an action item by extending its DueDate by N days.
    
    This function:
    1. Loads the action item and its associated subcase
    2. Enforces permission using _assert_user_can_modify
    3. Extends the DueDate by delay_days from today (or from current DueDate if in the future)
    
    Args:
        action_item_id: ID of the action item to delay
        delay_days: Number of days to extend the due date by (1-90)
        current_user: User object with user_id, allowed_unit_ids, and role attributes
        
    Returns:
        Dict with success status and new due date
        
    Raises:
        Unauthorized: If user is None or user_id is None
        NotFound: If action item does not exist
        Forbidden: If user lacks permission to modify
    """
    from datetime import date, timedelta
    
    # Validate delay_days
    if not isinstance(delay_days, int) or delay_days < 1 or delay_days > 90:
        raise Exception("delay_days must be an integer between 1 and 90")
    
    # Load action item from DB
    action_item = action_item_subcase_db.get_action_item_by_id(action_item_id)
    
    if action_item is None:
        raise NotFound(f"Action item {action_item_id} not found")
    
    # Load subcase for scope checking
    subcase_id = action_item["subcase_id"]
    subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
    
    if subcase is None:
        raise NotFound(f"Subcase {subcase_id} not found for action item {action_item_id}")
    
    # WORKFLOW CONTRACT GUARD: Block execution if subcase is in revision state
    subcase_status = subcase.get("status")
    if subcase_status in ['RETURNED_TO_SECTION_FOR_REVISION', 'RETURNED_TO_DEPT_FOR_REVISION']:
        raise Exception(
            f"Cannot delay action item: subcase is returned for revision (status: {subcase_status}). "
            "Wait for resubmission with corrected response."
        )
    
    # Enforce permission (scope + ownership/role)
    _assert_user_can_modify(action_item, subcase, current_user)
    
    # Compute new due date: extend from current DueDate or from today, whichever is later
    current_due_date = action_item.get("due_date")
    today = date.today()
    
    if current_due_date and isinstance(current_due_date, date) and current_due_date > today:
        base_date = current_due_date
    else:
        base_date = today
    
    new_due_date = base_date + timedelta(days=delay_days)
    
    # Update due date in DB
    action_item_subcase_db.update_action_item_due_date(
        action_item_id=action_item_id,
        new_due_date=new_due_date,
        updated_by_user_id=current_user.user_id
    )
    
    return {
        "success": True,
        "action_item_id": action_item_id,
        "previous_due_date": str(current_due_date) if current_due_date else None,
        "new_due_date": str(new_due_date),
        "delay_days": delay_days
    }
