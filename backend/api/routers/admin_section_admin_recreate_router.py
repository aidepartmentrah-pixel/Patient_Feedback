"""
Admin Section Admin Recreation Router

⚠️ ADMIN TEST TOOL — RECREATE SECTION ADMIN USER

This router provides endpoint to create additional section admin users.
Does not delete or modify existing admins - creates new ones with versioned usernames.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

# Import existing auth dependencies
from ..services.auth_service import get_current_user, CurrentUser
from ..utils.guards import require_software_admin

# Import service
from ..services.section_admin_recreate_service import recreate_section_admin_service

# Import schemas
from ..schemas.section_creation_schemas import SectionRecreateAdminResponse


# Create router
router = APIRouter(
    prefix="/api/admin/sections",
    tags=["admin-sections"]
)


@router.post("/{section_id}/recreate-admin", response_model=SectionRecreateAdminResponse)
def recreate_section_admin_endpoint(
    section_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a new section admin user for an existing section (SOFTWARE_ADMIN only).
    
    ⚠️ ADMIN TEST TOOL — RECREATE SECTION ADMIN USER
    
    **Requires:** SOFTWARE_ADMIN role
    
    **Purpose:**
    - Create additional admin account for a section
    - Does NOT delete or modify existing section admins
    - Generates unique username with version suffix if needed
    
    **Process:**
    1. Verify section exists and Type = 324 (SECTION)
    2. Generate unique username: sec_{id}_admin or sec_{id}_admin_v2, v3, etc.
    3. Create user with TEMP_HASH test password
    4. Assign SECTION_ADMIN role with section scope
    5. Commit transaction
    
    **Returns:**
    ```json
    {
        "section_id": 10,
        "username": "sec_10_admin_v2",
        "password": "Hospital2026!"
    }
    ```
    
    **Errors:**
    - 404: Section not found
    - 400: Organization unit is not a section (Type != 324)
    - 500: Database operation failed
    
    **Example:**
    ```
    POST /api/admin/sections/10/recreate-admin
    
    Response:
    {
        "section_id": 10,
        "username": "sec_10_admin_v2",
        "password": "Hospital2026!"
    }
    ```
    
    **Use Cases:**
    - Lost credentials for existing section admin
    - Need multiple admin accounts for testing
    - Section admin was deleted and needs replacement
    
    **Security Notes:**
    - Only accessible by SOFTWARE_ADMIN
    - Creates new account without affecting existing ones
    - Uses TEMP_HASH password for testing phase
    - Transaction ensures data consistency
    """
    # Check SOFTWARE_ADMIN permission
    require_software_admin(current_user)
    
    try:
        # Call service to recreate section admin
        result = recreate_section_admin_service(section_id)
        
        return {
            "section_id": result["section_id"],
            "username": result["username"],
            "temp_password": result["temp_password"]
        }
        
    except Exception as e:
        error_message = str(e)
        
        # Check for specific error types
        if "not found" in error_message.lower():
            raise HTTPException(
                status_code=404,
                detail=error_message
            )
        elif "not a section" in error_message.lower() or "type is" in error_message.lower():
            raise HTTPException(
                status_code=400,
                detail=error_message
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to recreate section admin: {error_message}"
            )
