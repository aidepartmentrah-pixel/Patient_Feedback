"""
Drawer Notes Guards for API v2
Phase G guard — Drawer Notes authorization.

These guards enforce ROLE ONLY for Drawer Notes functionality:
- No DB access
- No scope logic
- No business logic
- Only checks if user has SOFTWARE_ADMIN, WORKER, or COMPLAINT_SUPERVISOR role

Usage in routers:
    @router.post("/")
    def create_note(current_user = Depends(require_drawer_notes_role)):
        ...
"""

from fastapi import Depends, HTTPException, status
from backend.api.dependencies.user_context import get_current_user
from backend.api.schemas.auth_models import CurrentUser


# Allowed roles for Drawer Notes
ALLOWED_ROLES = {"SOFTWARE_ADMIN", "WORKER", "COMPLAINT_SUPERVISOR"}


def require_drawer_notes_role(
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """
    Phase G guard — Drawer Notes authorization.
    
    Allows: SOFTWARE_ADMIN, WORKER, COMPLAINT_SUPERVISOR
    Rejects: All other roles (SECTION_ADMIN, DEPARTMENT_ADMIN, ADMINISTRATION_ADMIN, etc.)
    
    This guard checks ONLY if the user has at least one of the allowed roles.
    Drawer notes are globally scoped (no org unit filtering).
    
    Args:
        current_user: Authenticated user with roles list
        
    Returns:
        CurrentUser: Same user if authorized
        
    Raises:
        HTTPException(403): If user does not have SOFTWARE_ADMIN, WORKER, or COMPLAINT_SUPERVISOR role
    """
    # Get user's roles
    user_roles = set(current_user.roles or [])
    
    # Check if user has any of the allowed roles
    if not user_roles.intersection(ALLOWED_ROLES):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized for Drawer Notes"
        )
    
    return current_user
