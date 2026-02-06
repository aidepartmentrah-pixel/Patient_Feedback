"""
Settings Users Router
API endpoints for Settings Users admin operations.

⚠️ SOFTWARE_ADMIN ONLY — USER MANAGEMENT

This router provides CRUD operations for managing users in the Settings UI.
All endpoints require SOFTWARE_ADMIN role.

Phase B - User Management Tooling
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any

# Auth dependencies
from ..services.auth_service import get_current_user, CurrentUser
from ..utils.guards import require_software_admin

# Service functions
from ..services.user_management_service import (
    create_user_with_role_scope,
    update_user_identity_service,
    list_users_for_settings_service,
    delete_user_service,
    admin_reset_user_password_service
)

# Schemas
from ..schemas.settings_users_models import (
    CreateUserRequest,
    UpdateUserIdentityRequest,
    UpdateUserPasswordRequest,
    SettingsUserListItemResponse,
    CreateUserResponse
)


# Create router
router = APIRouter(
    prefix="/api/settings/users",
    tags=["settings-users"]
)


@router.get("/", response_model=List[SettingsUserListItemResponse])
def list_users(
    current_user: CurrentUser = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    List all users for Settings Users table (SOFTWARE_ADMIN only).
    
    **Requires:** SOFTWARE_ADMIN role
    
    **Returns:**
    List of users with their role assignments and org unit mappings.
    Each user-role-scope combination returns as a separate row.
    
    **Response:**
    ```json
    [
        {
            "user_id": 5,
            "username": "section_admin",
            "display_name": "Dr. Sarah Johnson",
            "department_display_name": "Emergency Medicine",
            "role_name": "SECTION_ADMIN",
            "org_unit_name": "قسم الطوارئ",
            "is_active": true
        }
    ]
    ```
    
    **Errors:**
    - 403: User does not have SOFTWARE_ADMIN role
    """
    # Authorization guard
    require_software_admin(current_user)
    
    # Call service
    return list_users_for_settings_service()


@router.post("/", response_model=CreateUserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    request: CreateUserRequest,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a new user with role assignment (SOFTWARE_ADMIN only).
    
    **Requires:** SOFTWARE_ADMIN role
    
    **Body:**
    ```json
    {
        "username": "new_admin",
        "password": "SecurePass123!",
        "display_name": "Dr. Ahmed Ali",
        "department_display_name": "Emergency Department",
        "role_id": 6,
        "org_unit_id": 10
    }
    ```
    
    **Process:**
    1. Validates username uniqueness
    2. Hashes password with bcrypt
    3. Creates user record
    4. Assigns role and org unit scope
    5. Commits transaction
    
    **Returns:**
    ```json
    {
        "user_id": 123,
        "username": "new_admin"
    }
    ```
    
    **Errors:**
    - 403: User does not have SOFTWARE_ADMIN role
    - 400: Validation error (duplicate username, invalid IDs, etc.)
    - 500: Database error
    """
    # Authorization guard
    require_software_admin(current_user)
    
    # Call service
    try:
        user_id = create_user_with_role_scope(
            username=request.username,
            password_plain=request.password,
            display_name=request.display_name,
            department_display_name=request.department_display_name,
            role_id=request.role_id,
            org_unit_id=request.org_unit_id
        )
        
        return {
            "user_id": user_id,
            "username": request.username
        }
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.patch("/{user_id}/identity")
def update_user_identity(
    user_id: int,
    request: UpdateUserIdentityRequest,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Update user display identity fields (SOFTWARE_ADMIN only).
    
    **Requires:** SOFTWARE_ADMIN role
    
    **Body:**
    ```json
    {
        "display_name": "Dr. Ahmed Ali (Updated)",
        "department_display_name": "Cardiology"
    }
    ```
    
    **Notes:**
    - Can update one or both fields
    - Fields set to null/omitted remain unchanged
    - Does not modify username or password
    
    **Returns:**
    ```json
    {
        "status": "ok"
    }
    ```
    
    **Errors:**
    - 403: User does not have SOFTWARE_ADMIN role
    - 400: Validation error (both fields null, invalid user_id)
    - 404: User not found
    - 500: Database error
    """
    # Authorization guard
    require_software_admin(current_user)
    
    # Call service
    try:
        update_user_identity_service(
            user_id=user_id,
            display_name=request.display_name,
            department_display_name=request.department_display_name
        )
        
        return {"status": "ok"}
        
    except ValueError as e:
        # Check if it's a "not found" error
        if "not found" in str(e).lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.patch("/{user_id}/password")
def reset_user_password(
    user_id: int,
    request: UpdateUserPasswordRequest,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Admin reset user password (SOFTWARE_ADMIN only).
    
    **Requires:** SOFTWARE_ADMIN role
    
    **Body:**
    ```json
    {
        "new_password": "NewSecurePass123!"
    }
    ```
    
    **Process:**
    1. Validates inputs
    2. Hashes password with bcrypt
    3. Updates user password
    4. Commits transaction
    
    **Notes:**
    - Admin override - does not require old password
    - Password is automatically hashed
    - Password is never logged or returned
    
    **Returns:**
    ```json
    {
        "status": "password_updated"
    }
    ```
    
    **Errors:**
    - 403: User does not have SOFTWARE_ADMIN role
    - 400: Validation error (empty password, invalid user_id)
    - 404: User not found
    - 500: Database error
    """
    # Authorization guard
    require_software_admin(current_user)
    
    # Call service
    try:
        admin_reset_user_password_service(
            user_id=user_id,
            new_password=request.new_password
        )
        
        return {"status": "password_updated"}
        
    except ValueError as e:
        # Check if it's a "not found" error
        if "not found" in str(e).lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/{user_id}")
def delete_user(
    user_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Delete a user and all role assignments (SOFTWARE_ADMIN only).
    
    **Requires:** SOFTWARE_ADMIN role
    
    **Process:**
    1. Validates user exists
    2. Checks protection rules (cannot delete software_admin or SOFTWARE_ADMIN role users)
    3. Deletes role scopes
    4. Deletes user record
    5. Commits transaction
    
    **Safety Rules:**
    - Cannot delete username "software_admin"
    - Cannot delete any user with SOFTWARE_ADMIN role
    
    **Returns:**
    ```json
    {
        "status": "deleted"
    }
    ```
    
    **Errors:**
    - 403: User does not have SOFTWARE_ADMIN role or user is protected
    - 404: User not found
    - 500: Database error
    """
    # Authorization guard
    require_software_admin(current_user)
    
    # Call service
    try:
        delete_user_service(user_id)
        return {"status": "deleted"}
        
    except ValueError as e:
        # Check if it's a "not found" error
        if "not found" in str(e).lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        # Protection rule violation
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
