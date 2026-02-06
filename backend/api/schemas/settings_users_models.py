"""
Settings Users Schemas
Pydantic models for Settings Users admin operations.

Phase B - User Management Tooling
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional


class CreateUserRequest(BaseModel):
    """Request model for creating a new user."""
    username: str = Field(..., min_length=3, description="Username for login")
    password: str = Field(..., min_length=6, description="Initial password")
    display_name: Optional[str] = Field(None, description="Display name for UI")
    department_display_name: Optional[str] = Field(None, description="Department name for UI")
    role_id: int = Field(..., gt=0, description="Role ID to assign")
    org_unit_id: int = Field(..., gt=0, description="Organization unit ID for scope")
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "new_admin",
                "password": "SecurePass123!",
                "display_name": "Dr. Ahmed Ali",
                "department_display_name": "Emergency Department",
                "role_id": 6,
                "org_unit_id": 10
            }
        }


class UpdateUserIdentityRequest(BaseModel):
    """Request model for updating user display identity."""
    display_name: Optional[str] = Field(None, description="New display name (None = no change)")
    department_display_name: Optional[str] = Field(None, description="New department name (None = no change)")
    
    @model_validator(mode='after')
    def check_at_least_one_field(self):
        """Validate that at least one field is provided."""
        if self.display_name is None and self.department_display_name is None:
            raise ValueError("At least one field must be provided (display_name or department_display_name)")
        return self
    
    class Config:
        json_schema_extra = {
            "example": {
                "display_name": "Dr. Ahmed Ali (Updated)",
                "department_display_name": "Cardiology"
            }
        }


class UpdateUserPasswordRequest(BaseModel):
    """Request model for admin password reset."""
    new_password: str = Field(..., min_length=6, description="New password for user")
    
    @field_validator('new_password')
    @classmethod
    def check_not_whitespace(cls, v: str) -> str:
        """Validate password is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Password cannot be empty or whitespace only")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "new_password": "NewSecurePass123!"
            }
        }


class CreateUserResponse(BaseModel):
    """Response model for user creation."""
    user_id: int = Field(..., description="Created user ID")
    username: str = Field(..., description="Username")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "username": "new_admin"
            }
        }


class SettingsUserListItemResponse(BaseModel):
    """Response model for settings users list item."""
    user_id: int
    username: str
    display_name: Optional[str] = None
    department_display_name: Optional[str] = None
    role_name: str
    org_unit_name: str
    is_active: Optional[bool] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 5,
                "username": "section_admin",
                "display_name": "Dr. Sarah Johnson",
                "department_display_name": "Emergency Medicine",
                "role_name": "SECTION_ADMIN",
                "org_unit_name": "قسم الطوارئ",
                "is_active": True
            }
        }
