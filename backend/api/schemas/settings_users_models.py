"""
Settings Users Schemas
Pydantic models for Settings Users admin operations.

Phase B - User Management Tooling
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List


class CreateUserRequest(BaseModel):
    """Request model for creating a new user."""
    username: str = Field(..., min_length=3, description="Username for login")
    password: str = Field(..., min_length=6, description="Initial password")
    display_name: Optional[str] = Field(None, description="Display name for UI")
    department_display_name: Optional[str] = Field(None, description="Department name for UI")
    email: Optional[str] = Field(None, description="Email address for notifications")
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
    email: Optional[str] = Field(None, description="Email address for notifications (None = no change)")
    
    @model_validator(mode='after')
    def check_at_least_one_field(self):
        """Validate that at least one field is provided."""
        if self.display_name is None and self.department_display_name is None and self.email is None:
            raise ValueError("At least one field must be provided (display_name, department_display_name, or email)")
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
    email: Optional[str] = None
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
                "email": "sarah.johnson@hospital.local",
                "role_name": "SECTION_ADMIN",
                "org_unit_name": "قسم الطوارئ",
                "is_active": True
            }
        }


class BulkDeleteUsersRequest(BaseModel):
    """Request model for bulk deleting users."""
    user_ids: List[int] = Field(..., min_length=1, max_length=100, description="List of user IDs to delete (max 100)")
    
    @field_validator('user_ids')
    @classmethod
    def check_user_ids_valid(cls, v: List[int]) -> List[int]:
        """Validate user IDs are positive integers."""
        if not v:
            raise ValueError("user_ids cannot be empty")
        if len(v) > 100:
            raise ValueError("Cannot delete more than 100 users at once")
        for user_id in v:
            if user_id <= 0:
                raise ValueError(f"Invalid user_id: {user_id} (must be > 0)")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_ids": [1, 5, 12, 25, 33]
            }
        }


class DeletedUserResult(BaseModel):
    """Result for a single user deletion attempt."""
    user_id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    status: str = Field(..., description="Status: 'deleted' or 'failed'")
    reason: Optional[str] = Field(None, description="Failure reason (only for failed deletions)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "username": "testuser1",
                "status": "deleted"
            }
        }


class BulkDeleteUsersResponse(BaseModel):
    """Response model for bulk user deletion."""
    success: bool = Field(..., description="True if all deletions succeeded")
    deleted_count: int = Field(..., description="Number of users successfully deleted")
    failed_count: int = Field(..., description="Number of users that failed to delete")
    deleted_users: List[DeletedUserResult] = Field(default_factory=list, description="List of successfully deleted users")
    failed_users: List[DeletedUserResult] = Field(default_factory=list, description="List of failed deletion attempts")
    message: str = Field(..., description="Summary message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "deleted_count": 5,
                "failed_count": 0,
                "deleted_users": [
                    {
                        "user_id": 1,
                        "username": "testuser1",
                        "status": "deleted"
                    },
                    {
                        "user_id": 5,
                        "username": "testuser2",
                        "status": "deleted"
                    }
                ],
                "failed_users": [],
                "message": "Successfully deleted 5 user(s)"
            }
        }
