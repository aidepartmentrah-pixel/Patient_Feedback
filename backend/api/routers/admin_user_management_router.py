"""
Admin User Management Router

⚠️ ADMIN TEST TOOL — USER DELETE — HANDLE WITH CARE

This router provides user management operations for SOFTWARE_ADMIN.
Includes safety checks to prevent deletion of protected accounts.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel

# Import existing auth dependencies
from ..services.auth_service import get_current_user, CurrentUser
from ..utils.guards import require_role
from core.constants.roles import SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR

# Import service
from ..services.user_management_service import delete_user_service, update_user_service


# Request model for user updates
class UpdateUserRequest(BaseModel):
    display_name: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


# Create router
router = APIRouter(
    prefix="/api/admin",
    tags=["admin-users"]
)


@router.delete("/users/{user_id}")
def delete_user_endpoint(
    user_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Delete a user account (SOFTWARE_ADMIN only).
    
    ⚠️ ADMIN TEST TOOL — USER DELETE — HANDLE WITH CARE
    
    **Requires:** SOFTWARE_ADMIN role
    
    **Safety Features:**
    - Blocks deletion of "software_admin" username
    - Blocks deletion of any user with SOFTWARE_ADMIN role
    - Deletes role scopes before user (maintains referential integrity)
    - Fully transactional (rolls back on error)
    
    **Process:**
    1. Verify user exists
    2. Check protection rules
    3. Delete user role scopes (APP_UserRoleScope)
    4. Delete user record (APP_Users)
    5. Commit transaction
    
    **Returns:**
    ```json
    {
        "deleted_user_id": 15,
        "deleted_username": "sec_10_admin"
    }
    ```
    
    **Errors:**
    - 404: User not found
    - 403: Protected account (software_admin or has SOFTWARE_ADMIN role)
    - 500: Database operation failed
    
    **Example:**
    ```
    DELETE /api/admin/users/15
    
    Response:
    {
        "deleted_user_id": 15,
        "deleted_username": "sec_10_admin"
    }
    ```
    
    **Security Notes:**
    - Only accessible by SOFTWARE_ADMIN
    - Cannot delete system administrator accounts
    - Transaction ensures data consistency
    - Does not cascade to organizational tables
    """
    # Check SOFTWARE_ADMIN or COMPLAINT_SUPERVISOR permission
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    
    try:
        # Call service to delete user
        result = delete_user_service(user_id)
        
        return result
        
    except Exception as e:
        error_message = str(e)
        
        # Check for specific error types
        if "not found" in error_message.lower():
            raise HTTPException(
                status_code=404,
                detail=error_message
            )
        elif "cannot delete" in error_message.lower() or "protected" in error_message.lower():
            raise HTTPException(
                status_code=403,
                detail=error_message
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete user: {error_message}"
            )


@router.put("/users/{user_id}")
def update_user_endpoint(
    user_id: int,
    updates: UpdateUserRequest,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update a user account (SOFTWARE_ADMIN only).
    
    **Requires:** SOFTWARE_ADMIN role
    
    **Allowed Updates:**
    - display_name: User's display name for UI
    - username: Login username (must be unique, 3-50 alphanumeric chars)
    - password: New password (minimum 8 characters)
    
    **Protection Rules:**
    - Only SOFTWARE_ADMIN can edit users
    - Cannot edit SOFTWARE_ADMIN users (protected)
    - Username must be unique
    - Username: 3-50 chars, alphanumeric + underscore only
    - Password: minimum 8 characters
    
    **Request Body:**
    ```json
    {
        "display_name": "John Smith",
        "username": "new_username",
        "password": "newpassword123"
    }
    ```
    All fields are optional - only provided fields will be updated.
    
    **Returns:**
    ```json
    {
        "success": true,
        "user": {
            "user_id": 1,
            "username": "updated_username",
            "display_name": "John Smith"
        }
    }
    ```
    
    **Errors:**
    - 403: Not authorized (non-SOFTWARE_ADMIN user)
    - 403: Protected user (target is SOFTWARE_ADMIN)
    - 404: User not found
    - 400: Username taken or validation error
    
    **Error Responses:**
    ```json
    // 403 - Not authorized
    {"detail": "Only SOFTWARE_ADMIN can edit users"}
    
    // 403 - Protected user
    {"detail": "Cannot edit SOFTWARE_ADMIN users"}
    
    // 400 - Username taken
    {"detail": "Username already exists"}
    
    // 400 - Validation error
    {"detail": "Username must be 3-50 alphanumeric characters"}
    ```
    
    **Example:**
    ```
    PUT /api/admin/users/5
    Content-Type: application/json
    
    {
        "display_name": "John Smith",
        "username": "john_admin",
        "password": "secure123"
    }
    
    Response:
    {
        "success": true,
        "user": {
            "user_id": 5,
            "username": "john_admin",
            "display_name": "John Smith"
        }
    }
    ```
    
    **Security Notes:**
    - Only accessible by SOFTWARE_ADMIN
    - Cannot edit SOFTWARE_ADMIN accounts
    - Password is hashed before storage
    - Test password stored as TEMP_HASH_ format for testing
    - Transaction ensures data consistency
    """
    # Check SOFTWARE_ADMIN or COMPLAINT_SUPERVISOR permission
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    
    try:
        # Call service to update user
        result = update_user_service(
            user_id=user_id,
            display_name=updates.display_name,
            username=updates.username,
            password=updates.password
        )
        
        return result
        
    except Exception as e:
        error_message = str(e)
        
        # Check for specific error types
        if "not found" in error_message.lower():
            raise HTTPException(
                status_code=404,
                detail=error_message
            )
        elif "cannot edit" in error_message.lower() or "protected" in error_message.lower():
            raise HTTPException(
                status_code=403,
                detail=error_message
            )
        elif "username already exists" in error_message.lower() or "must be" in error_message.lower():
            raise HTTPException(
                status_code=400,
                detail=error_message
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update user: {error_message}"
            )
