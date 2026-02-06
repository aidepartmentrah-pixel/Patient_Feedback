"""
Example: Using get_current_user() Dependency
Demonstrates how to protect endpoints with session-based authentication.
"""

from fastapi import APIRouter, Depends
from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser

router = APIRouter(prefix="/api/example", tags=["Example - Protected Endpoints"])


@router.get("/public")
async def public_endpoint():
    """
    Public endpoint - No authentication required.
    Anyone can access this endpoint.
    """
    return {
        "message": "This is a public endpoint",
        "authentication": "not required"
    }


@router.get("/protected")
async def protected_endpoint(current_user: CurrentUser = Depends(get_current_user)):
    """
    Protected endpoint - Authentication required.
    
    This endpoint requires a valid session. Users must login first via:
    POST /api/auth/login
    
    The get_current_user dependency automatically:
    - Validates session exists
    - Loads user from database
    - Injects CurrentUser into endpoint
    - Returns 401 if not authenticated
    """
    return {
        "message": f"Hello {current_user.username}!",
        "user_id": current_user.user_id,
        "is_active": current_user.is_active,
        "roles": [scope.role_code for scope in current_user.scopes],
        "authentication": "required and successful"
    }


@router.get("/user-info")
async def get_user_info(current_user: CurrentUser = Depends(get_current_user)):
    """
    Get detailed information about currently authenticated user.
    
    Demonstrates accessing user scopes and organizational units.
    """
    return {
        "user": {
            "id": current_user.user_id,
            "username": current_user.username,
            "active": current_user.is_active
        },
        "access": {
            "roles": [
                {
                    "role": scope.role_code,
                    "org_unit_id": scope.org_unit_id,
                    "org_unit_type": scope.org_unit_type
                }
                for scope in current_user.scopes
            ]
        }
    }


@router.get("/check-role/{role_code}")
async def check_user_role(
    role_code: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Check if current user has a specific role.
    
    Example: GET /api/example/check-role/SOFTWARE_ADMIN
    """
    has_role = any(scope.role_code == role_code for scope in current_user.scopes)
    
    return {
        "user": current_user.username,
        "role": role_code,
        "has_role": has_role,
        "user_roles": [scope.role_code for scope in current_user.scopes]
    }
