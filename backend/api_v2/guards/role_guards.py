"""
Role Guards for API v2
Simple, explicit role-based access control guards.

These guards enforce ROLE ONLY:
- No DB access
- No scope logic
- No business logic
- No magic

Usage in routers:
    @router.get("/endpoint")
    def endpoint(current_user = Depends(require_section_admin)):
        ...
"""

from fastapi import Depends, HTTPException
from backend.api.dependencies.user_context import get_current_user
from backend.api.schemas.auth_models import CurrentUser


def require_supervisor_or_worker(
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """
    Allows: SUPERVISOR, WORKER
    """
    if current_user.role not in ["SUPERVISOR", "WORKER"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    return current_user


def require_section_admin(
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """
    Allows: SECTION_ADMIN
    """
    if current_user.role != "SECTION_ADMIN":
        raise HTTPException(status_code=403, detail="Forbidden")
    return current_user


def require_dept_admin(
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """
    Allows: DEPARTMENT_ADMIN
    """
    if current_user.role != "DEPARTMENT_ADMIN":
        raise HTTPException(status_code=403, detail="Forbidden")
    return current_user


def require_admin_level(
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """
    Allows: DEPARTMENT_ADMIN, ADMINISTRATION_ADMIN
    """
    if current_user.role not in ["DEPARTMENT_ADMIN", "ADMINISTRATION_ADMIN"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    return current_user


def require_administrator(
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """
    Allows: ADMINISTRATION_ADMIN
    """
    if current_user.role != "ADMINISTRATION_ADMIN":
        raise HTTPException(status_code=403, detail="Forbidden")
    return current_user
