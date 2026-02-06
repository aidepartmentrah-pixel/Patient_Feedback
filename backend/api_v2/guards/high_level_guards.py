"""
High-Level Guards for API v2
Combines role checks with scope/ownership validation.

These guards compose:
- Role guards (from role_guards.py)
- Scope validation (from scoping.py)

Used at router level for endpoint protection.

Example usage:
    @router.post("/subcase/{subcase_id}/section-response")
    def submit_section_response(
        subcase_id: int,
        current_user = Depends(require_section_admin_on_subcase)
    ):
        ...
"""

from fastapi import Depends, HTTPException
from backend.api.dependencies.user_context import get_current_user
from backend.api.schemas.auth_models import CurrentUser
from backend.api_v2.guards.scoping import validate_case_access


def require_section_admin_on_subcase(
    subcase_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """
    Requires: SECTION_ADMIN role AND access to subcase's org unit.
    
    Args:
        subcase_id: Subcase ID from path parameter
        current_user: Authenticated user from session
        
    Returns:
        current_user if authorized
        
    Raises:
        HTTPException(403): If role check or scope validation fails
    """
    # First: Check role
    if current_user.role != "SECTION_ADMIN":
        raise HTTPException(status_code=403, detail="Forbidden")
    
    # Then: Validate scope access
    try:
        validate_case_access(subcase_id, current_user)
    except Exception as e:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return current_user


def require_dept_admin_on_subcase(
    subcase_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """
    Requires: DEPARTMENT_ADMIN role AND access to subcase's org unit.
    
    Args:
        subcase_id: Subcase ID from path parameter
        current_user: Authenticated user from session
        
    Returns:
        current_user if authorized
        
    Raises:
        HTTPException(403): If role check or scope validation fails
    """
    # First: Check role
    if current_user.role != "DEPARTMENT_ADMIN":
        raise HTTPException(status_code=403, detail="Forbidden")
    
    # Then: Validate scope access
    try:
        validate_case_access(subcase_id, current_user)
    except Exception as e:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return current_user


def require_admin_on_subcase(
    subcase_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """
    Requires: ADMINISTRATION_ADMIN role AND access to subcase's org unit.
    
    Args:
        subcase_id: Subcase ID from path parameter
        current_user: Authenticated user from session
        
    Returns:
        current_user if authorized
        
    Raises:
        HTTPException(403): If role check or scope validation fails
    """
    # First: Check role
    if current_user.role != "ADMINISTRATION_ADMIN":
        raise HTTPException(status_code=403, detail="Forbidden")
    
    # Then: Validate scope access
    try:
        validate_case_access(subcase_id, current_user)
    except Exception as e:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return current_user


def require_worker_or_supervisor_on_subcase(
    subcase_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """
    Requires: SUPERVISOR or WORKER role AND access to subcase's org unit.
    
    Args:
        subcase_id: Subcase ID from path parameter
        current_user: Authenticated user from session
        
    Returns:
        current_user if authorized
        
    Raises:
        HTTPException(403): If role check or scope validation fails
    """
    # First: Check role
    if current_user.role not in ["SUPERVISOR", "WORKER"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    # Then: Validate scope access
    try:
        validate_case_access(subcase_id, current_user)
    except Exception as e:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return current_user
