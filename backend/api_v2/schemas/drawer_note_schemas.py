"""
Drawer Note Schemas for API v2
Phase G — Request and response models for Drawer Notes endpoints.

These schemas define the contract between frontend and backend for drawer notes API.
"""

from pydantic import BaseModel, Field
from typing import List
from datetime import datetime


class CreateNoteRequest(BaseModel):
    """Request model for creating a new drawer note."""
    note_text: str = Field(..., description="The note content")
    label_ids: List[int] = Field(..., description="List of label IDs to attach to note")
    
    class Config:
        json_schema_extra = {
            "example": {
                "note_text": "Patient expressed concern about medication timing",
                "label_ids": [1, 3]
            }
        }


class UpdateNoteTextRequest(BaseModel):
    """Request model for updating note text."""
    note_text: str = Field(..., description="The new note content")
    
    class Config:
        json_schema_extra = {
            "example": {
                "note_text": "Updated note: Patient confirmed medication schedule works well"
            }
        }


class UpdateNoteLabelsRequest(BaseModel):
    """Request model for replacing note labels."""
    label_ids: List[int] = Field(..., description="New list of label IDs (replaces existing)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "label_ids": [2, 4, 5]
            }
        }


class NoteResponse(BaseModel):
    """Response model for a single drawer note."""
    note_id: int = Field(..., description="Note unique identifier")
    note_text: str = Field(..., description="Note content")
    created_at: datetime = Field(..., description="Creation timestamp")
    created_by_user_id: int = Field(..., description="Creator user ID")
    created_by_name: str = Field(..., description="Creator username")
    label_ids: List[int] = Field(..., description="Attached label IDs")
    is_deleted: bool = Field(False, description="Whether note is soft-deleted")
    
    class Config:
        json_schema_extra = {
            "example": {
                "note_id": 42,
                "note_text": "Patient needs follow-up on treatment plan",
                "created_at": "2026-02-07T10:30:00",
                "created_by_user_id": 5,
                "created_by_name": "john_doe",
                "label_ids": [1, 3],
                "is_deleted": False
            }
        }


class ListNotesResponse(BaseModel):
    """Response model for list of drawer notes."""
    items: List[NoteResponse] = Field(..., description="List of notes")
    total: int = Field(..., description="Total number of items returned")
    
    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {
                        "note_id": 42,
                        "note_text": "Sample note",
                        "created_at": "2026-02-07T10:30:00",
                        "created_by_user_id": 5,
                        "created_by_name": "john_doe",
                        "label_ids": [1, 3],
                        "is_deleted": False
                    }
                ],
                "total": 1
            }
        }


class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = Field(True, description="Operation success indicator")
    message: str = Field(..., description="Success message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Note updated successfully"
            }
        }


class CreateNoteResponse(BaseModel):
    """Response model for note creation."""
    note_id: int = Field(..., description="Created note ID")
    success: bool = Field(True, description="Operation success indicator")
    
    class Config:
        json_schema_extra = {
            "example": {
                "note_id": 42,
                "success": True
            }
        }
