"""
User Inventory Router
API endpoints for organizational unit and user mapping queries.

Phase 5 - Module 5.1: Inventory & Mapping Engine
Admin-only endpoints for viewing user-org unit assignments.
"""

from fastapi import APIRouter, Depends, status
from typing import List, Dict, Any

from ..services.auth_service import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_software_admin
from ..services.user_inventory_service import (
    get_org_unit_user_inventory_service,
    get_org_units_without_users_service,
    get_inventory_summary_service
)


router = APIRouter(prefix="/api/admin/user-inventory", tags=["Admin - User Inventory"])


@router.get("", response_model=List[Dict[str, Any]], status_code=status.HTTP_200_OK)
def get_user_inventory(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get comprehensive inventory of all organizational units and their assigned users.
    
    **Authorization:** SOFTWARE_ADMIN only
    
    Returns a list showing:
    - All organizational units (Administrations, Departments, Sections)
    - Which users (if any) are assigned to each unit
    - The roles assigned to those users
    - User active status
    
    **Response Format:**
    ```json
    [
        {
            "org_unit_id": 10,
            "org_unit_name": "قسم الطوارئ",
            "org_unit_type": "SECTION",
            "username": "sec_10_admin",
            "role_code": "SECTION_ADMIN",
            "is_active": true
        },
        {
            "org_unit_id": 25,
            "org_unit_name": "قسم الجراحة",
            "org_unit_type": "SECTION",
            "username": null,
            "role_code": null,
            "is_active": null
        }
    ]
    ```
    
    **Note:** Org units without users will have null username/role_code.
    """
    # Check authorization
    require_software_admin(current_user)
    
    # Retrieve inventory
    return get_org_unit_user_inventory_service()


@router.get("/missing", response_model=List[Dict[str, Any]], status_code=status.HTTP_200_OK)
def get_org_units_without_users(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get organizational units that have NO users assigned.
    
    **Authorization:** SOFTWARE_ADMIN only
    
    Useful for identifying which units need user accounts created.
    
    **Response Format:**
    ```json
    [
        {
            "org_unit_id": 25,
            "org_unit_name": "قسم الجراحة",
            "org_unit_type": "SECTION"
        }
    ]
    ```
    """
    # Check authorization
    require_software_admin(current_user)
    
    # Retrieve org units without users
    return get_org_units_without_users_service()


@router.get("/summary", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
def get_inventory_summary(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get summary statistics about user inventory.
    
    **Authorization:** SOFTWARE_ADMIN only
    
    Returns counts of:
    - Total organizational units
    - Total active users
    - Units with users (by type)
    - Units without users
    
    **Response Format:**
    ```json
    {
        "total_org_units": 150,
        "total_users": 45,
        "administrations_with_users": 3,
        "departments_with_users": 12,
        "sections_with_users": 30,
        "org_units_without_users": 105
    }
    ```
    """
    # Check authorization
    require_software_admin(current_user)
    
    # Retrieve summary
    return get_inventory_summary_service()
