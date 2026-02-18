"""
Admin User Credentials Router (TEST ONLY)

⚠️ TEST ONLY ENDPOINT — MUST NOT BE ENABLED IN PRODUCTION

This router exposes user credentials for testing purposes only.
It derives test passwords from TEMP_HASH_ format.

SECURITY WARNING:
- This endpoint MUST be disabled in production
- Only accessible by SOFTWARE_ADMIN role
- Returns derived test passwords (not actual hashes)
- Should be removed or gated before production deployment
"""

from fastapi import APIRouter, Depends
from typing import List, Dict, Any

# Import existing auth dependencies
from ..services.auth_service import get_current_user, CurrentUser
from ..utils.guards import require_role
from core.constants.roles import SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR

# Import service
from ..services.user_credentials_service import get_all_user_credentials_service


# Create router
router = APIRouter(
    prefix="/api/admin/testing",
    tags=["admin-testing"]
)


@router.get("/user-credentials")
def list_all_user_credentials(
    current_user: CurrentUser = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    List all users with their test credentials (TEST ONLY).
    
    ⚠️ TEST ONLY — DO NOT USE IN PRODUCTION
    
    **Requires:** SOFTWARE_ADMIN role
    
    **Purpose:**
    - View all user accounts for testing
    - Get test passwords derived from TEMP_HASH_ format
    - Verify role and scope assignments
    
    **Returns:**
    List of user credentials with:
    - user_id: User's ID
    - username: Username
    - display_name: User's display name (or null)
    - role: Assigned role code (or null if no scope)
    - org_unit: Organization unit name (or null)
    - org_unit_id: Organization unit ID (or null)
    - org_unit_type: Organization unit type (or null)
    - active: Whether account is active
    - test_password: Derived test password (or null if not TEMP_HASH)
    
    **Security Notes:**
    - Never returns actual PasswordHash field
    - Only derives passwords from TEMP_HASH_ prefix
    - Non-test accounts show test_password = null
    
    **Example Response:**
    ```json
    [
        {
            "user_id": 7,
            "username": "sec_10_admin",
            "display_name": "Emergency Department Admin",
            "role": "SECTION_ADMIN",
            "org_unit": "Emergency Department",
            "org_unit_id": 10,
            "org_unit_type": "SECTION",
            "active": true,
            "test_password": "Hospital2026!"
        }
    ]
    ```
    """
    # Check SOFTWARE_ADMIN or COMPLAINT_SUPERVISOR permission
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    
    # Get all user credentials
    credentials = get_all_user_credentials_service()
    
    return credentials
