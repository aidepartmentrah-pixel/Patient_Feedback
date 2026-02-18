"""
Admin Section Router
Admin-only endpoints for creating sections with admin users.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

# Import existing auth dependencies
from ..services.auth_service import CurrentUser
from ..utils.guards import get_current_software_admin

# Import service
from ..services.section_admin_creator_service import create_section_with_admin

# Import schemas
from ..schemas.section_creation_schemas import (
    SectionCreateRequest,
    SectionCreateResponse
)


# Create router
router = APIRouter(
    prefix="/api/admin",
    tags=["admin-sections"]
)


# CONTRACT FROZEN — Used by Settings UI — Do not change without versioning
@router.post("/create-section-with-admin", response_model=SectionCreateResponse)
def create_section_with_admin_endpoint(
    request: SectionCreateRequest,
    admin: CurrentUser = Depends(get_current_software_admin)
) -> Dict[str, Any]:
    """
    Create a new section org unit and automatically create a SECTION_ADMIN user for it.
    
    ═══════════════════════════════════════════════════════════════════════════
    CONTRACT SPECIFICATION — Phase C B-C7
    ═══════════════════════════════════════════════════════════════════════════
    
    **Purpose:**
    Admin-only endpoint for creating leaf-level section organizational units.
    Parent must be ADMINISTRATION (Type=322) or DEPARTMENT (Type=323).
    Always creates both the section AND its admin user atomically.
    
    **HTTP Method:** POST
    **Path:** /api/admin/create-section-with-admin
    **Auth:** Requires SOFTWARE_ADMIN role (via get_current_software_admin dependency)
    
    **Request Model:** SectionCreateRequest
    **Request JSON Example:**
    ```json
    {
        "section_name": "Emergency Department Section A",
        "parent_unit_id": 5
    }
    ```
    
    **Response Model:** SectionCreateResponse
    **Response JSON Example:**
    ```json
    {
        "section_id": 101,
        "section_name": "Emergency Department Section A",
        "parent_unit_id": 5,
        "username": "sec_101_admin",
        "temp_password": "Hospital2026!"
    }
    ```
    
    **Field Constraints:**
    - section_name: 2-100 characters, non-empty after trim
    - parent_unit_id: Must exist in AdminsrationUnit, Type 322 or 323
    
    **Behavior:**
    1. Creates section in AdminsrationUnit (Type=324, IsActive=1)
    2. Generates username: sec_{section_id}_admin
    3. Creates APP_Users record with TEMP_HASH password
    4. Creates APP_UserRoleScope record (SECTION_ADMIN role, section scope)
    5. Returns complete details for both created entities
    
    **Authorization:**
    - 401 Unauthorized: Missing or invalid session
    - 403 Forbidden: Authenticated but not SOFTWARE_ADMIN role
    - 200 OK: Success with SectionCreateResponse payload
    
    **Error Scenarios:**
    - 422 Validation Error: Invalid request fields
    - 500 Internal Server Error: Database or service failure
    
    ═══════════════════════════════════════════════════════════════════════════
    DO NOT modify path, method, request fields, or response fields without
    creating a new versioned endpoint (e.g., /v2/create-section-with-admin).
    Frontend Settings UI depends on this exact contract.
    ═══════════════════════════════════════════════════════════════════════════
    """
    try:
        # Call service to create section + admin
        result = create_section_with_admin(
            section_name=request.section_name,
            parent_department_id=request.parent_unit_id
        )
        
        # Return complete response with all created entity details
        return {
            "section_id": result["section_id"],
            "section_name": result["section_name"],
            "parent_unit_id": result["parent_unit_id"],
            "username": result["username"],
            "temp_password": result["temp_password"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create section with admin: {str(e)}"
        )
