"""
Authentication Models
Pydantic models for authentication and user context.
"""

from pydantic import BaseModel
from typing import List, Optional, Set


class UserScope(BaseModel):
    """Single role-scope mapping for a user."""
    role_code: str
    org_unit_id: int
    org_unit_type: str


class CurrentUser(BaseModel):
    """
    Current authenticated user with all role scopes.
    
    This model represents the authenticated user context throughout the application.
    It contains the user's identity and all their role-scope assignments.
    
    Attributes:
        user_id: Unique user identifier
        username: Username for display and logging
        display_name: Optional display name for UI greetings (Phase A)
        department_display_name: Optional department name for UI display (Phase A)
        is_active: Whether the user account is active
        scopes: List of role-scope mappings for authorization
        allowed_unit_ids: Set of org unit IDs this user can access (computed from scopes)
        roles: List of unique role codes (derived from scopes) - Phase 4
        primary_unit_id: Primary organizational unit ID (derived from scopes) - Phase 4
        primary_unit_type: Primary organizational unit type (derived from scopes) - Phase 4
    """
    user_id: int
    username: str
    display_name: Optional[str] = None
    department_display_name: Optional[str] = None
    is_active: bool
    scopes: List[UserScope]
    allowed_unit_ids: Set[int] = set()
    
    # Phase 4 additions - derived fields for frontend consumption
    roles: List[str] = []
    primary_unit_id: Optional[int] = None
    primary_unit_type: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 5,
                "username": "section_admin",
                "display_name": "Dr. Sarah Johnson",
                "department_display_name": "Emergency Medicine",
                "is_active": True,
                "scopes": [
                    {
                        "role_code": "SECTION_ADMIN",
                        "org_unit_id": 10,
                        "org_unit_type": "SECTION"
                    }
                ],
                "allowed_unit_ids": [10],
                "roles": ["SECTION_ADMIN"],
                "primary_unit_id": 10,
                "primary_unit_type": "SECTION"
            }
        }
