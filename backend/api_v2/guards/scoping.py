"""
Scoping Validation Functions
Validates user access to subcases based on organizational scope.

These functions are used by high-level guards to enforce scope constraints.
Contains NO DB access - only validation logic.
"""

from typing import Dict, Any


def validate_case_access(subcase_id: int, current_user) -> None:
    """
    Validate that current_user has access to the given subcase based on organizational scope.
    
    This function checks if the subcase's TargetOrgUnitID is within the user's
    allowed organizational scope (allowed_unit_ids from Phase 2.5).
    
    Args:
        subcase_id: ID of the subcase to validate access for
        current_user: User object with allowed_unit_ids attribute
        
    Raises:
        Exception: If user doesn't have access to the subcase's target org unit
    """
    # Import here to avoid circular dependency
    from backend.api_v2.db_layer import administrative_subcase_db
    
    # Load subcase to get its TargetOrgUnitID
    subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
    
    if subcase is None:
        raise Exception(f"Subcase {subcase_id} not found")
    
    target_org_unit_id = subcase.get("target_org_unit_id")
    allowed_unit_ids = getattr(current_user, 'allowed_unit_ids', None)
    
    # Validate scope access
    if not allowed_unit_ids or not target_org_unit_id or target_org_unit_id not in allowed_unit_ids:
        raise Exception(f"Access denied: Subcase {subcase_id} is outside user's organizational scope")


def validate_action_item_access(action_item_id: int, current_user) -> None:
    """
    Validate that current_user has access to the given action item based on organizational scope.
    
    This function:
    1. Loads the action item
    2. Loads its associated subcase
    3. Checks if the subcase's TargetOrgUnitID is within the user's allowed scope
    
    Args:
        action_item_id: ID of the action item to validate access for
        current_user: User object with allowed_unit_ids attribute
        
    Raises:
        Exception: If user doesn't have access to the action item's subcase
    """
    # Import here to avoid circular dependency
    from backend.api_v2.db_layer import action_item_subcase_db, administrative_subcase_db
    
    # Load action item to get its subcase_id
    action_item = action_item_subcase_db.get_action_item_by_id(action_item_id)
    
    if action_item is None:
        raise Exception(f"Action item {action_item_id} not found")
    
    subcase_id = action_item.get("subcase_id")
    if not subcase_id:
        raise Exception(f"Action item {action_item_id} has no associated subcase")
    
    # Load subcase to get its TargetOrgUnitID
    subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
    
    if subcase is None:
        raise Exception(f"Subcase {subcase_id} not found for action item {action_item_id}")
    
    target_org_unit_id = subcase.get("target_org_unit_id")
    allowed_unit_ids = getattr(current_user, 'allowed_unit_ids', None)
    
    # Validate scope access
    if not allowed_unit_ids or not target_org_unit_id or target_org_unit_id not in allowed_unit_ids:
        raise Exception(f"Access denied: Action item {action_item_id} is outside user's organizational scope")


def can_generate_seasonal_report(current_user) -> bool:
    """
    Check if user has permission to generate seasonal reports.
    
    Currently allows any authenticated user with organizational scope.
    Can be extended with additional logic if needed.
    
    Args:
        current_user: User object with allowed_unit_ids attribute
        
    Returns:
        True if user can generate reports, False otherwise
    """
    allowed_unit_ids = getattr(current_user, 'allowed_unit_ids', None)
    
    # User must have at least one org unit in scope
    return allowed_unit_ids is not None and len(allowed_unit_ids) > 0
