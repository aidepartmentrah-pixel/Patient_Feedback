"""
Drawer Label Schemas for API v2
Phase G — Request and response models for Drawer Labels endpoints.

These schemas define the contract between frontend and backend for drawer labels API.
"""

from pydantic import BaseModel, Field
from datetime import datetime


class CreateLabelRequest(BaseModel):
    """Request model for creating a new drawer label."""
    label_name: str = Field(..., description="The label name (2-100 characters)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "label_name": "Follow-up Required"
            }
        }


class LabelResponse(BaseModel):
    """Response model for a single drawer label."""
    label_id: int = Field(..., description="Label unique identifier")
    label_name: str = Field(..., description="Label name")
    is_active: bool = Field(True, description="Whether label is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "label_id": 5,
                "label_name": "Follow-up Required",
                "is_active": True,
                "created_at": "2026-02-07T10:00:00"
            }
        }


class CreateLabelResponse(BaseModel):
    """Response model for label creation."""
    label_id: int = Field(..., description="Created label ID")
    success: bool = Field(True, description="Operation success indicator")
    
    class Config:
        json_schema_extra = {
            "example": {
                "label_id": 5,
                "success": True
            }
        }


class ListLabelsResponse(BaseModel):
    """Response model for list of labels."""
    labels: list[LabelResponse] = Field(..., description="List of active labels")
    total: int = Field(..., description="Total number of labels")
    
    class Config:
        json_schema_extra = {
            "example": {
                "labels": [
                    {
                        "label_id": 5,
                        "label_name": "Follow-up Required",
                        "is_active": True,
                        "created_at": "2026-02-07T10:00:00"
                    }
                ],
                "total": 1
            }
        }
