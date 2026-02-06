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
    user_role = getattr(current_user, 'role', None)
    
    # Allow if user is assigned to this action item
    if assigned_to_user_id == current_user.user_id:
        return
    
    # Allow if user has privileged role
    privileged_roles = ["SUPERVISOR", "SECTION_ADMIN", "DEPT_ADMIN", "ADMIN"]
    if user_role in privileged_roles:
        return
    
    # Otherwise, forbidden
    raise Forbidden(f"User is not assigned to this action item and does not have a privileged role")


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

def get_action_items_for_user(current_user) -> List[Dict[str, Any]]:
    """
    Get all action items assigned to the current user, filtered by scope.
    
    This function:
    1. Fetches action items assigned to the user
    2. Fetches the associated subcases to get TargetOrgUnitID
    3. Filters action items to only include those within user's allowed_unit_ids
    
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
    
    # Get all action items assigned to this user
    action_items = action_item_subcase_db.get_action_items_by_assigned_user(current_user.user_id)
    
    # If no action items, return empty list
    if not action_items:
        return []
    
    # Apply scope filtering using Phase 2.5
    # Get allowed unit IDs from Phase 2.5 Scope Engine
    allowed_unit_ids = getattr(current_user, 'allowed_unit_ids', None)
    
    # If no allowed_unit_ids, user has no scope - return empty
    if not allowed_unit_ids:
        return []
    
    # Build a map of subcase_id -> target_org_unit_id
    # We need to fetch subcases to get their TargetOrgUnitID
    subcase_ids = {item["subcase_id"] for item in action_items if item["subcase_id"]}
    
    if not subcase_ids:
        return []
    
    # Fetch subcases to get their TargetOrgUnitID
    subcase_scope_map = {}
    for subcase_id in subcase_ids:
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        if subcase:
            subcase_scope_map[subcase_id] = subcase.get("target_org_unit_id")
    
    # Filter action items: keep only those where the subcase's TargetOrgUnitID is in allowed scope
    filtered_action_items = []
    for action_item in action_items:
        subcase_id = action_item["subcase_id"]
        target_org_unit_id = subcase_scope_map.get(subcase_id)
        
        # Security check: only include if target is in allowed scope
        if target_org_unit_id and target_org_unit_id in allowed_unit_ids:
            filtered_action_items.append(action_item)
    
    return filtered_action_items


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


def delay_action_item(action_item_id: int, current_user) -> bool:
    """
    Delay (cancel) an action item by setting its status to CANCELLED.
    
    This function:
    1. Loads the action item and its associated subcase
    2. Enforces permission using _assert_user_can_modify
    3. Calls DB layer to update status to CANCELLED
    
    Args:
        action_item_id: ID of the action item to delay
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
            f"Cannot delay/cancel action item: subcase is returned for revision (status: {subcase_status}). "
            "Wait for resubmission with corrected response."
        )
    
    # Enforce permission (scope + ownership/role)
    _assert_user_can_modify(action_item, subcase, current_user)
    
    # Execute the mutation via DB layer (set status to CANCELLED)
    success = action_item_subcase_db.update_action_item_status(
        action_item_id, 
        "CANCELLED", 
        current_user.user_id
    )
    
    return success
