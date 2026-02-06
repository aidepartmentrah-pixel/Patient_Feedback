"""
Inbox Service — Phase 3 Workflow (API v2)

This service provides read-only inbox views for different administrative roles.
Each role sees subcases filtered by their organizational scope and current status.

ROLE MODEL:
- Section Administrator: Works on SUBMITTED_TO_SECTION
- Department Administrator: Works on SECTION_ACCEPTED_PENDING_DEPT
- Administration Administrator: Works on DEPT_ACCEPTED_PENDING_ADMIN

STATUS WORKFLOW:
SUBMITTED_TO_SECTION → SECTION_ACCEPTED_PENDING_DEPT → DEPT_ACCEPTED_PENDING_ADMIN

This service is READ-ONLY:
- No writes, no updates, no deletes
- No workflow transitions
- No side effects

Structure:
- Public functions: get_inbox, get_section_inbox, get_department_inbox, get_administration_inbox
- Internal helpers: _apply_scope_filter, _compute_allowed_actions
"""

from typing import List, Dict, Any, Optional
from backend.api_v2.db_layer import administrative_subcase_db


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

def get_inbox(current_user) -> List[Dict[str, Any]]:
    """
    Get inbox for current user (role-agnostic delegator).
    
    This is a thin routing function that delegates to the appropriate
    role-specific inbox function based on the user's role.
    
    This function exists solely for API v2 workflow router simplification.
    It contains NO business logic, NO filtering, NO scoping.
    
    Args:
        current_user: User object with role attribute
        
    Returns:
        List of inbox items (dictionaries) with allowed actions
        
    Raises:
        ValueError: If user role is not recognized or not authorized
    """
    # Simple role-based delegation based on scopes
    # Get user's primary role from first scope
    if not current_user.scopes:
        raise ValueError("User has no role scopes")
    
    primary_role = current_user.scopes[0].role_code
    
    if primary_role == 'SECTION_ADMIN':
        return get_section_inbox(current_user)
    elif primary_role == 'DEPARTMENT_ADMIN':
        return get_department_inbox(current_user)
    elif primary_role == 'ADMINISTRATION_ADMIN':
        return get_administration_inbox(current_user)
    else:
        raise ValueError(f"Role '{primary_role}' does not have inbox access")


def get_section_inbox(current_user) -> List[Dict[str, Any]]:
    """
    Get inbox for Section Administrator role.
    
    Returns subcases with status SUBMITTED_TO_SECTION that are within
    the user's section scope.
    
    Args:
        current_user: User object with role, section_id, department_id attributes
        
    Returns:
        List of inbox items (dictionaries) with allowed actions
        
    Raises:
        ValueError: If user is not a Section Administrator
    """
    # Light role assertion using scopes
    if not current_user.scopes or current_user.scopes[0].role_code != 'SECTION_ADMIN':
        raise ValueError("User must be a Section Administrator to access section inbox")
    
    # Get subcases pending for section from DB layer
    subcases = administrative_subcase_db.get_subcases_pending_for_section()
    
    # Apply scope filter
    filtered_subcases = _apply_scope_filter(subcases, current_user)
    
    # Build inbox items with allowed actions
    inbox_items = []
    for subcase in filtered_subcases:
        item = _build_inbox_item(subcase, current_user)
        inbox_items.append(item)
    
    return inbox_items


def get_department_inbox(current_user) -> List[Dict[str, Any]]:
    """
    Get inbox for Department Administrator role.
    
    Returns subcases with status SECTION_ACCEPTED_PENDING_DEPT that are within
    the user's department scope.
    
    Args:
        current_user: User object with role, section_id, department_id attributes
        
    Returns:
        List of inbox items (dictionaries) with allowed actions
        
    Raises:
        ValueError: If user is not a Department Administrator
    """
    # Light role assertion using scopes
    if not current_user.scopes or current_user.scopes[0].role_code != 'DEPARTMENT_ADMIN':
        raise ValueError("User must be a Department Administrator to access department inbox")
    
    # Get subcases pending for department from DB layer
    subcases = administrative_subcase_db.get_subcases_pending_for_department()
    
    # Apply scope filter
    filtered_subcases = _apply_scope_filter(subcases, current_user)
    
    # Build inbox items with allowed actions
    inbox_items = []
    for subcase in filtered_subcases:
        item = _build_inbox_item(subcase, current_user)
        inbox_items.append(item)
    
    return inbox_items


def get_administration_inbox(current_user) -> List[Dict[str, Any]]:
    """
    Get inbox for Administration Administrator role.
    
    Returns subcases with status DEPT_ACCEPTED_PENDING_ADMIN that are within
    the user's administration scope (no filtering - sees all).
    
    Args:
        current_user: User object with role, section_id, department_id attributes
        
    Returns:
        List of inbox items (dictionaries) with allowed actions
        
    Raises:
        ValueError: If user is not an Administration Administrator
    """
    # Light role assertion using scopes
    if not current_user.scopes or current_user.scopes[0].role_code != 'ADMINISTRATION_ADMIN':
        raise ValueError("User must be an Administration Administrator to access administration inbox")
    
    # Get subcases pending for administration from DB layer
    subcases = administrative_subcase_db.get_subcases_pending_for_administration()
    
    # Apply scope filter (administration sees all, so no real filtering)
    filtered_subcases = _apply_scope_filter(subcases, current_user)
    
    # Build inbox items with allowed actions
    inbox_items = []
    for subcase in filtered_subcases:
        item = _build_inbox_item(subcase, current_user)
        inbox_items.append(item)
    
    return inbox_items


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _apply_scope_filter(subcases: List[Any], current_user) -> List[Any]:
    """
    Filter subcases based on user's organizational scope from Phase 2.5 Scope Engine.
    
    SECURITY-CRITICAL: This function enforces the central scope boundary.
    Only subcases where TargetOrgUnitID is in current_user.allowed_unit_ids are returned.
    
    This is the ONLY authority for data access:
    - Role does NOT grant data access
    - Legacy local org IDs are IGNORED
    - current_user.allowed_unit_ids is the single source of truth
    
    Even if role logic, router, or frontend is compromised, this function ensures
    no out-of-scope data is ever returned.
    
    Args:
        subcases: List of subcase dicts from DB layer
        current_user: User object with allowed_unit_ids attribute (set[int])
        
    Returns:
        Filtered list of subcases where target_org_unit_id is in allowed_unit_ids
    """
    if not subcases:
        return []
    
    # Get allowed unit IDs from Phase 2.5 Scope Engine
    # This is the ONLY authority for data access
    allowed_unit_ids = getattr(current_user, 'allowed_unit_ids', None)
    
    # If no allowed_unit_ids, user has no scope - return empty
    if not allowed_unit_ids:
        return []
    
    # Filter subcases: keep only those where target_org_unit_id is in allowed scope
    filtered = []
    for subcase in subcases:
        # Subcases from DB layer are dicts, not objects
        target_org_unit_id = subcase.get('target_org_unit_id')
        
        # Security check: only include if target is in allowed scope
        if target_org_unit_id in allowed_unit_ids:
            filtered.append(subcase)
    
    return filtered


def _compute_allowed_actions(subcase: Dict[str, Any], current_user) -> List[str]:
    """
    Compute allowed actions for a subcase based on its status and user's role.
    
    This is minimal placeholder logic. Business rules will be added later.
    
    Action mapping (placeholder):
    - SUBMITTED_TO_SECTION (Section Admin): ["SUBMIT_RESPONSE", "REJECT"]
    - SECTION_ACCEPTED_PENDING_DEPT (Dept Admin): ["APPROVE", "REJECT"]
    - DEPT_ACCEPTED_PENDING_ADMIN (Admin Admin): ["APPROVE", "REJECT"]
    
    Args:
        subcase: Subcase dict from DB layer
        current_user: CurrentUser object with .scopes[] array
        
    Returns:
        List of action strings per API v2 contract
    """
    # DB layer returns dict with lowercase snake_case keys
    status = subcase.get('status', None)
    
    # CurrentUser has .scopes[] array with role_code
    role_code = current_user.scopes[0].role_code if current_user.scopes else None
    
    actions = []
    
    # Section Administrator actions
    if role_code == 'SECTION_ADMIN' and status == 'SUBMITTED_TO_SECTION':
        actions.extend(["SUBMIT_RESPONSE", "REJECT"])
    
    # Department Administrator actions
    elif role_code == 'DEPARTMENT_ADMIN' and status == 'SECTION_ACCEPTED_PENDING_DEPT':
        actions.extend(["APPROVE", "REJECT"])
    
    # Administration Administrator actions
    elif role_code == 'ADMINISTRATION_ADMIN' and status == 'DEPT_ACCEPTED_PENDING_ADMIN':
        actions.extend(["APPROVE", "REJECT"])
    
    return actions


def _build_inbox_item(subcase: Dict[str, Any], current_user) -> Dict[str, Any]:
    """
    Build an inbox item dictionary from a subcase dict.
    
    Converts DB layer dict to API-ready dictionary format.
    
    Args:
        subcase: Subcase dict from DB layer
        current_user: User object for computing allowed actions
        
    Returns:
        Dictionary with inbox item structure
    """
    # Compute allowed actions for this subcase
    allowed_actions = _compute_allowed_actions(subcase, current_user)
    
    # Build the inbox item (subcase is already a dict from DB layer)
    inbox_item = {
        "subcase_id": subcase.get('subcase_id'),
        "case_type": subcase.get('case_type'),
        "incident_id": subcase.get('incident_request_case_id'),
        "seasonal_report_id": subcase.get('seasonal_report_id'),
        "target_org_unit_id": subcase.get('target_org_unit_id'),
        "status": subcase.get('status'),
        "created_at": subcase.get('created_at'),
        "allowed_actions": allowed_actions
    }
    
    return inbox_item
