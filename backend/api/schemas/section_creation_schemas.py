"""
Section Creation Schemas
Pydantic models for section org unit creation operations.

Phase C - Section Creation Tooling
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class SectionCreateRequest(BaseModel):
    """
    Request model for creating a new section org unit.
    
    A section is a leaf-level organizational unit (Type=324) that must have
    a parent org unit (either Administration or Department).
    """
    section_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Name of the section org unit"
    )
    parent_unit_id: int = Field(
        ...,
        gt=0,
        description="Parent org unit ID (administration or department)"
    )
    
    @field_validator('section_name')
    @classmethod
    def validate_section_name(cls, v: str) -> str:
        """Validate section name is not empty or whitespace only."""
        # Strip whitespace
        stripped = v.strip()
        
        if not stripped:
            raise ValueError("Section name cannot be empty or whitespace only")
        
        if len(stripped) < 2:
            raise ValueError("Section name must be at least 2 characters long")
        
        if len(stripped) > 100:
            raise ValueError("Section name cannot exceed 100 characters")
        
        return stripped
    
    class Config:
        json_schema_extra = {
            "example": {
                "section_name": "Emergency Department Section A",
                "parent_unit_id": 5
            }
        }


class SectionCreateResponse(BaseModel):
    """
    Response model for section creation.
    
    Returns created section details and credentials for the auto-generated
    section admin user.
    """
    section_id: int = Field(..., description="Created section's UniqueID")
    section_name: str = Field(..., description="Created section name")
    parent_unit_id: int = Field(..., description="Parent org unit ID")
    username: str = Field(..., description="Generated admin username (sec_{id}_admin)")
    temp_password: str = Field(..., description="Temporary test password")
    
    class Config:
        json_schema_extra = {
            "example": {
                "section_id": 101,
                "section_name": "Emergency Department Section A",
                "parent_unit_id": 5,
                "username": "sec_101_admin",
                "temp_password": "Hospital2026!"
            }
        }


class SectionRecreateAdminResponse(BaseModel):
    """
    Response model for section admin recreation.
    
    Used when creating additional admin users for existing sections.
    """
    section_id: int = Field(..., description="Section's UniqueID")
    username: str = Field(..., description="Generated admin username with version suffix")
    temp_password: str = Field(..., description="Temporary test password")
    
    class Config:
        json_schema_extra = {
            "example": {
                "section_id": 101,
                "username": "sec_101_admin_v2",
                "temp_password": "Hospital2026!"
            }
        }
