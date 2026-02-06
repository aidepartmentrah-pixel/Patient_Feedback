"""
Example: Using Authorization Guards
Demonstrates how to use role-based authorization guards in endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import (
    require_software_admin,
    require_worker,
    require_any_admin,
    require_any_supervisor,
    has_role,
    get_user_roles,
)
from core.constants.roles import SOFTWARE_ADMIN, WORKER

router = APIRouter(prefix="/api/guarded", tags=["Example - Authorization Guards"])


@router.get("/public")
async def public_endpoint():
    """
    Public endpoint - No authentication or authorization.
    Anyone can access this endpoint.
    """
    return {
        "message": "This is a public endpoint",
        "authentication": "not required",
        "authorization": "not required"
    }


@router.get("/authenticated-only")
async def authenticated_endpoint(current_user: CurrentUser = Depends(get_current_user)):
    """
    Authenticated endpoint - Requires login but no specific role.
    
    Any logged-in user can access this endpoint regardless of role.
    The get_current_user dependency handles authentication.
    """
    return {
        "message": f"Hello {current_user.username}!",
        "authentication": "required",
        "authorization": "any authenticated user",
        "your_roles": get_user_roles(current_user)
    }


@router.get("/admin-only")
async def admin_only_endpoint(current_user: CurrentUser = Depends(get_current_user)):
    """
    Admin-only endpoint - Requires SOFTWARE_ADMIN role.
    
    Only users with SOFTWARE_ADMIN role can access this endpoint.
    Returns 403 Forbidden for users without the role.
    """
    # Check authorization
    require_software_admin(current_user)
    
    return {
        "message": "Welcome, Software Administrator!",
        "authentication": "required",
        "authorization": "SOFTWARE_ADMIN only",
        "action": "sensitive admin operation performed"
    }


@router.get("/worker-only")
async def worker_only_endpoint(current_user: CurrentUser = Depends(get_current_user)):
    """
    Worker-only endpoint - Requires WORKER role.
    
    Only users with WORKER role can access this endpoint.
    """
    # Check authorization
    require_worker(current_user)
    
    return {
        "message": f"Worker {current_user.username} authorized",
        "authentication": "required",
        "authorization": "WORKER only",
        "action": "worker operation performed"
    }


@router.get("/any-admin")
async def any_admin_endpoint(current_user: CurrentUser = Depends(get_current_user)):
    """
    Any admin endpoint - Requires any admin role.
    
    Allows SOFTWARE_ADMIN, SECTION_ADMIN, DEPARTMENT_ADMIN, or ADMINISTRATION_ADMIN.
    """
    # Check authorization
    require_any_admin(current_user)
    
    return {
        "message": f"Admin {current_user.username} authorized",
        "authentication": "required",
        "authorization": "any admin role",
        "your_roles": get_user_roles(current_user)
    }


@router.get("/any-supervisor")
async def any_supervisor_endpoint(current_user: CurrentUser = Depends(get_current_user)):
    """
    Any supervisor endpoint - Requires supervisor or admin role.
    
    Allows COMPLAINT_SUPERVISOR or any admin role.
    """
    # Check authorization
    require_any_supervisor(current_user)
    
    return {
        "message": f"Supervisor {current_user.username} authorized",
        "authentication": "required",
        "authorization": "supervisor or admin role",
        "your_roles": get_user_roles(current_user)
    }


@router.post("/conditional-access")
async def conditional_access_endpoint(current_user: CurrentUser = Depends(get_current_user)):
    """
    Conditional access - Different behavior based on role.
    
    Demonstrates using has_role() for conditional logic instead of strict guards.
    """
    # Check if user is admin (non-throwing)
    is_admin = has_role(current_user, SOFTWARE_ADMIN)
    
    if is_admin:
        # Admins get full access
        return {
            "message": "Full access granted",
            "data": {
                "sensitive_field_1": "admin data",
                "sensitive_field_2": "more admin data",
                "all_records": ["record1", "record2", "record3"]
            },
            "access_level": "full"
        }
    else:
        # Other users get limited access
        return {
            "message": "Limited access granted",
            "data": {
                "public_field": "basic data"
            },
            "access_level": "limited"
        }


@router.delete("/dangerous-operation/{resource_id}")
async def dangerous_operation(
    resource_id: int,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Dangerous operation - Multiple authorization checks.
    
    Demonstrates checking multiple conditions and returning custom errors.
    """
    # First check: Must be an admin
    require_any_admin(current_user)
    
    # Second check: For extra safety, only SOFTWARE_ADMIN can delete certain resources
    if resource_id <= 10:  # Protected resources
        require_software_admin(current_user)
    
    return {
        "message": f"Resource {resource_id} deleted",
        "deleted_by": current_user.username,
        "roles": get_user_roles(current_user)
    }


@router.get("/my-permissions")
async def get_my_permissions(current_user: CurrentUser = Depends(get_current_user)):
    """
    Get current user's permissions and roles.
    
    Useful for debugging and UI permission checks.
    """
    roles = get_user_roles(current_user)
    
    # Check what the user can do
    permissions = {
        "can_access_admin_panel": has_role(current_user, SOFTWARE_ADMIN),
        "can_manage_workers": any(has_role(current_user, role) for role in 
                                  [SOFTWARE_ADMIN, "SECTION_ADMIN", "DEPARTMENT_ADMIN"]),
        "can_approve_complaints": any(has_role(current_user, role) for role in 
                                      [SOFTWARE_ADMIN, "COMPLAINT_SUPERVISOR"]),
        "can_handle_complaints": has_role(current_user, WORKER),
    }
    
    return {
        "user": current_user.username,
        "user_id": current_user.user_id,
        "roles": roles,
        "permissions": permissions,
        "scopes": [
            {
                "role": scope.role_code,
                "org_unit_id": scope.org_unit_id,
                "org_unit_type": scope.org_unit_type
            }
            for scope in current_user.scopes
        ]
    }
