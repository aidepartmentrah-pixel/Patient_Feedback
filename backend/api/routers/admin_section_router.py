"""
Admin Section Router
Admin-only endpoints for creating sections with admin users.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Import existing auth dependencies
from ..services.auth_service import get_current_user, CurrentUser
from ..utils.guards import require_software_admin

# Import service
from ..services.section_admin_creator_service import create_section_with_admin


# Create router
router = APIRouter(
    prefix="/api/admin",
    tags=["admin-sections"]
)


# Request model
class CreateSectionRequest(BaseModel):
    section_name: str
    parent_department_id: int


# Response model
class CreateSectionResponse(BaseModel):
    section_id: int
    username: str
    password: str


@router.post("/create-section-with-admin", response_model=CreateSectionResponse)
def create_section_with_admin_endpoint(
    request: CreateSectionRequest,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a new section and automatically create a SECTION_ADMIN user for it.
    
    **Requires:** SOFTWARE_ADMIN role
    
    **Process:**
    1. Creates new section in AdminsrationUnit table (Type=324)
    2. Generates username: sec_{section_id}_admin
    3. Creates user with TEMP_HASH test password
    4. Assigns SECTION_ADMIN role with section scope
    
    **Returns:**
    - section_id: New section's UniqueID
    - username: Generated admin username
    - password: Test password (Hospital2026!)
    
    **Example:**
    ```
    POST /api/admin/create-section-with-admin
    {
        "section_name": "Emergency Department",
        "parent_department_id": 5
    }
    
    Response:
    {
        "section_id": 101,
        "username": "sec_101_admin",
        "password": "Hospital2026!"
    }
    ```
    """
    # Check SOFTWARE_ADMIN permission
    require_software_admin(current_user)
    
    try:
        # Call service to create section + admin
        result = create_section_with_admin(
            section_name=request.section_name,
            parent_department_id=request.parent_department_id
        )
        
        # Return credentials
        return {
            "section_id": result["section_id"],
            "username": result["username"],
            "password": result["temp_password"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create section with admin: {str(e)}"
        )
