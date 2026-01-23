"""
System Settings Router
FastAPI endpoints for managing system-wide configuration settings.
"""

from fastapi import APIRouter, HTTPException, Body, Path
from typing import List, Optional
from pydantic import BaseModel, Field

from ..services.system_settings_service import SystemSettingsService


router = APIRouter(prefix="/api/system-settings", tags=["System Settings"])


# ==================== REQUEST/RESPONSE MODELS ====================

class SystemSettingOut(BaseModel):
    """Response model for a system setting."""
    key: str = Field(..., description="Unique setting key")
    value: str = Field(..., description="Raw string value from database")
    type: str = Field(..., description="Setting type: int, bool, string, or json")
    description: Optional[str] = Field(None, description="Human-readable description")
    updated_at: Optional[str] = Field(None, description="Last update timestamp (ISO format)")
    updated_by_user_id: Optional[int] = Field(None, description="ID of user who last updated")
    parsed_value: Optional[any] = Field(None, description="Parsed value in appropriate type")
    parse_error: Optional[str] = Field(None, description="Error message if parsing failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "key": "ComplaintDelayDays",
                "value": "14",
                "type": "int",
                "description": "After this many days, a complaint is considered delayed",
                "updated_at": "2026-01-21T10:30:00",
                "updated_by_user_id": None,
                "parsed_value": 14,
                "parse_error": None
            }
        }


class SystemSettingUpdateRequest(BaseModel):
    """Request model for updating a setting value."""
    value: str = Field(..., description="New value (as string)")
    updated_by_user_id: Optional[int] = Field(None, description="ID of user making the update")
    
    class Config:
        json_schema_extra = {
            "example": {
                "value": "21",
                "updated_by_user_id": 1
            }
        }


class SystemSettingCreateRequest(BaseModel):
    """Request model for creating a new setting."""
    key: str = Field(..., description="Unique setting key")
    value: str = Field(..., description="Setting value (as string)")
    type: str = Field(..., description="Setting type: int, bool, string, or json")
    description: Optional[str] = Field(None, description="Human-readable description")
    updated_by_user_id: Optional[int] = Field(None, description="ID of user creating the setting")
    
    class Config:
        json_schema_extra = {
            "example": {
                "key": "MaxUploadSizeMB",
                "value": "50",
                "type": "int",
                "description": "Maximum file upload size in megabytes",
                "updated_by_user_id": 1
            }
        }


# ==================== ENDPOINTS ====================

@router.get(
    "",
    response_model=List[SystemSettingOut],
    summary="Get All System Settings",
    description="Retrieve all system configuration settings."
)
def get_all_settings():
    """
    Get all system settings.
    
    Returns a list of all settings with their parsed values.
    """
    try:
        settings = SystemSettingsService.get_all_settings()
        return settings
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve settings: {str(e)}"
        )


@router.get(
    "/{key}",
    response_model=SystemSettingOut,
    summary="Get Single System Setting",
    description="Retrieve a specific system setting by its key."
)
def get_setting(
    key: str = Path(..., description="The setting key to retrieve")
):
    """
    Get a single system setting by key.
    
    Args:
        key: The setting key
        
    Returns:
        The setting details with parsed value
        
    Raises:
        404: If setting not found
    """
    try:
        setting = SystemSettingsService.get_setting(key)
        return setting
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve setting: {str(e)}"
        )


@router.put(
    "/{key}",
    response_model=SystemSettingOut,
    summary="Update System Setting",
    description="Update the value of an existing system setting."
)
def update_setting(
    key: str = Path(..., description="The setting key to update"),
    request: SystemSettingUpdateRequest = Body(...)
):
    """
    Update a system setting value.
    
    Args:
        key: The setting key to update
        request: Update request containing new value and optional user ID
        
    Returns:
        The updated setting with parsed value
        
    Raises:
        404: If setting not found
        400: If validation fails
    """
    try:
        updated_setting = SystemSettingsService.update_setting(
            key=key,
            value=request.value,
            updated_by_user_id=request.updated_by_user_id
        )
        return updated_setting
    except ValueError as e:
        # Check if it's a "not found" error or validation error
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update setting: {str(e)}"
        )


@router.post(
    "",
    response_model=SystemSettingOut,
    status_code=201,
    summary="Create System Setting",
    description="Create a new system setting. (Admin use - not typically needed in UI)"
)
def create_setting(
    request: SystemSettingCreateRequest = Body(...)
):
    """
    Create a new system setting.
    
    Args:
        request: Create request with key, value, type, description, and optional user ID
        
    Returns:
        The newly created setting
        
    Raises:
        400: If validation fails or key already exists
    """
    try:
        new_setting = SystemSettingsService.create_setting(
            key=request.key,
            value=request.value,
            setting_type=request.type,
            description=request.description,
            updated_by_user_id=request.updated_by_user_id
        )
        return new_setting
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create setting: {str(e)}"
        )


@router.delete(
    "/{key}",
    status_code=204,
    summary="Delete System Setting",
    description="Delete a system setting. (Admin use - not typically needed in UI)"
)
def delete_setting(
    key: str = Path(..., description="The setting key to delete")
):
    """
    Delete a system setting.
    
    Args:
        key: The setting key to delete
        
    Returns:
        No content on success
        
    Raises:
        404: If setting not found
    """
    try:
        deleted = SystemSettingsService.delete_setting(key)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Setting '{key}' not found"
            )
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete setting: {str(e)}"
        )
