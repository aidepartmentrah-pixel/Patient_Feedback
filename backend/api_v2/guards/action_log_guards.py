"""
Action Log Guards for API v2
Phase F guard — Action Log export authorization.

These guards enforce ROLE ONLY for Action Log export functionality:
- No DB access
- No scope logic
- No business logic
- Only checks if user has SOFTWARE_ADMIN or WORKER role

Usage in routers:
    @router.get("/export")
    def export_endpoint(current_user = Depends(require_action_log_role)):
        ...
"""

from fastapi import Depends, HTTPException, status
from backend.api.dependencies.user_context import get_current_user
from backend.api.schemas.auth_models import CurrentUser


# Allowed roles for Action Log export
ALLOWED_ROLES = {"SOFTWARE_ADMIN", "WORKER", "COMPLAINT_SUPERVISOR"}


def require_action_log_role(
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """
    Phase F guard — Action Log export authorization.
    
    Allows: SOFTWARE_ADMIN, WORKER, COMPLAINT_SUPERVISOR
    Rejects: All other roles (SECTION_ADMIN, DEPARTMENT_ADMIN, ADMINISTRATION_ADMIN, etc.)
    
    This guard checks ONLY if the user has at least one of the allowed roles.
    Scope filtering is handled separately by the service layer.
    
    Args:
        current_user: Authenticated user with roles list
        
    Returns:
        CurrentUser: Same user if authorized
        
    Raises:
        HTTPException(403): If user does not have SOFTWARE_ADMIN, WORKER, or COMPLAINT_SUPERVISOR role
    """
    # Get user's roles (Phase 4 added roles field)
    user_roles = set(current_user.roles or [])
    
    # Check if user has any of the allowed roles
    if not user_roles.intersection(ALLOWED_ROLES):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized for Action Log export"
        )
    
    return current_user
