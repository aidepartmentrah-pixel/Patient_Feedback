"""
Admin User Management Router

⚠️ ADMIN TEST TOOL — USER DELETE — HANDLE WITH CARE

This router provides user management operations for SOFTWARE_ADMIN.
Includes safety checks to prevent deletion of protected accounts.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

# Import existing auth dependencies
from ..services.auth_service import get_current_user, CurrentUser
from ..utils.guards import require_software_admin

# Import service
from ..services.user_management_service import delete_user_service


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
    # Check SOFTWARE_ADMIN permission
    require_software_admin(current_user)
    
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
