"""
Drawer Notes Router (API v2)
Phase G — G-B7 — Drawer Notes CRUD endpoints.

This router provides drawer notes management endpoints under /api/v2/drawer-notes.

Endpoints exposed:
- POST /api/v2/drawer-notes - Create new note
- GET /api/v2/drawer-notes - List notes (with optional label filtering)
- GET /api/v2/drawer-notes/{note_id} - Get single note
- PUT /api/v2/drawer-notes/{note_id}/text - Update note text
- PUT /api/v2/drawer-notes/{note_id}/labels - Update note labels
- DELETE /api/v2/drawer-notes/{note_id} - Soft delete note

Security: 
- All endpoints protected by authentication
- Requires SOFTWARE_ADMIN or WORKER role
"""

from fastapi import APIRouter, Query, Path, Depends, HTTPException, status, Response
from typing import List, Optional

from backend.api_v2.guards.drawer_notes_guards import require_drawer_notes_role
from backend.api.schemas.auth_models import CurrentUser
from backend.api_v2.services import drawer_note_service
from backend.api_v2.services.drawer_note_export_service import build_drawer_notes_word_export
from backend.api_v2.schemas.drawer_note_schemas import (
    CreateNoteRequest,
    UpdateNoteTextRequest,
    UpdateNoteLabelsRequest,
    NoteResponse,
    ListNotesResponse,
    SuccessResponse,
    CreateNoteResponse
)


# ============================================================
# ROUTER DEFINITION
# ============================================================
router = APIRouter(prefix="/api/v2/drawer-notes", tags=["Drawer Notes V2"])


# ============================================================
# ENDPOINTS
# ============================================================

@router.post("/", response_model=CreateNoteResponse, status_code=status.HTTP_201_CREATED)
def create_drawer_note(
    request: CreateNoteRequest,
    current_user: CurrentUser = Depends(require_drawer_notes_role)
):
    """
    Create a new drawer note with labels.
    
    Phase G Drawer Notes — Create endpoint.
    
    Business Rules:
    - Note text cannot be empty (trimmed)
    - Must have at least one label
    - All labels must be active
    
    Args:
        request: Note creation request with text and label IDs
        current_user: Authenticated user with SOFTWARE_ADMIN or WORKER role
        
    Returns:
        Created note ID
        
    Security:
        - Requires authentication
        - Requires SOFTWARE_ADMIN or WORKER role
        
    Raises:
        400: Validation error (empty text, empty labels, invalid labels)
        401: Not authenticated
        403: Not authorized (missing required role)
    """
    try:
        note_id = drawer_note_service.create_note_with_labels(
            note_text=request.note_text,
            label_ids=request.label_ids,
            created_by_user_id=current_user.user_id,
            created_by_name=current_user.username
        )
        
        return CreateNoteResponse(note_id=note_id, success=True)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/", response_model=ListNotesResponse)
def list_drawer_notes(
    label_ids: Optional[List[int]] = Query(None, description="Filter by label IDs (AND logic)"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: CurrentUser = Depends(require_drawer_notes_role)
):
    """
    List drawer notes with optional label filtering.
    
    Phase G Drawer Notes — List endpoint.
    
    Label Filtering:
    - If label_ids provided: returns notes with ALL specified labels (AND logic)
    - If no label_ids: returns all active notes
    - Deleted notes are excluded from list
    
    Args:
        label_ids: Optional list of label IDs for filtering
        limit: Maximum number of results (1-500)
        offset: Pagination offset
        current_user: Authenticated user with SOFTWARE_ADMIN or WORKER role
        
    Returns:
        List of notes with metadata
        
    Security:
        - Requires authentication
        - Requires SOFTWARE_ADMIN or WORKER role
        
    Raises:
        401: Not authenticated
        403: Not authorized (missing required role)
    """
    notes = drawer_note_service.list_notes(
        label_ids=label_ids,
        limit=limit,
        offset=offset
    )
    
    # Convert to response models
    items = [
        NoteResponse(
            note_id=note['note_id'],
            note_text=note['note_text'],
            created_at=note['created_at'],
            created_by_user_id=note['created_by_user_id'],
            created_by_name=note['created_by_name'],
            label_ids=note.get('label_ids', []),
            is_deleted=note.get('is_deleted', False)
        )
        for note in notes
    ]
    
    return ListNotesResponse(items=items, total=len(items))


@router.get("/{note_id}", response_model=NoteResponse)
def get_drawer_note(
    note_id: int = Path(..., description="Note ID"),
    current_user: CurrentUser = Depends(require_drawer_notes_role)
):
    """
    Get single drawer note by ID.
    
    Phase G Drawer Notes — Get detail endpoint.
    
    Returns note with all details including attached labels.
    Deleted notes can still be retrieved (soft delete).
    
    Args:
        note_id: Note unique identifier
        current_user: Authenticated user with SOFTWARE_ADMIN or WORKER role
        
    Returns:
        Note details with label IDs
        
    Security:
        - Requires authentication
        - Requires SOFTWARE_ADMIN or WORKER role
        
    Raises:
        404: Note not found
        401: Not authenticated
        403: Not authorized (missing required role)
    """
    note = drawer_note_service.get_note_detail(note_id)
    
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Note {note_id} not found"
        )
    
    return NoteResponse(
        note_id=note['note_id'],
        note_text=note['note_text'],
        created_at=note['created_at'],
        created_by_user_id=note['created_by_user_id'],
        created_by_name=note['created_by_name'],
        label_ids=note.get('label_ids', []),
        is_deleted=note.get('is_deleted', False)
    )


@router.put("/{note_id}/text", response_model=SuccessResponse)
def update_drawer_note_text(
    note_id: int = Path(..., description="Note ID"),
    request: UpdateNoteTextRequest = ...,
    current_user: CurrentUser = Depends(require_drawer_notes_role)
):
    """
    Update drawer note text.
    
    Phase G Drawer Notes — Edit text endpoint.
    
    Business Rules:
    - Text cannot be empty (trimmed)
    - Cannot edit deleted notes
    
    Args:
        note_id: Note unique identifier
        request: Update request with new text
        current_user: Authenticated user with SOFTWARE_ADMIN or WORKER role
        
    Returns:
        Success response
        
    Security:
        - Requires authentication
        - Requires SOFTWARE_ADMIN or WORKER role
        
    Raises:
        400: Validation error (empty text, deleted note)
        404: Note not found
        401: Not authenticated
        403: Not authorized (missing required role)
    """
    try:
        drawer_note_service.edit_note_text(note_id, request.note_text)
        return SuccessResponse(
            success=True,
            message=f"Note {note_id} text updated successfully"
        )
        
    except ValueError as e:
        # Check if it's a "not found" error
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )


@router.put("/{note_id}/labels", response_model=SuccessResponse)
def update_drawer_note_labels(
    note_id: int = Path(..., description="Note ID"),
    request: UpdateNoteLabelsRequest = ...,
    current_user: CurrentUser = Depends(require_drawer_notes_role)
):
    """
    Replace drawer note labels.
    
    Phase G Drawer Notes — Edit labels endpoint.
    
    Business Rules:
    - Must have at least one label
    - All labels must be active
    - Cannot edit deleted notes
    
    Args:
        note_id: Note unique identifier
        request: Update request with new label IDs
        current_user: Authenticated user with SOFTWARE_ADMIN or WORKER role
        
    Returns:
        Success response
        
    Security:
        - Requires authentication
        - Requires SOFTWARE_ADMIN or WORKER role
        
    Raises:
        400: Validation error (empty labels, invalid labels, deleted note)
        404: Note not found
        401: Not authenticated
        403: Not authorized (missing required role)
    """
    try:
        drawer_note_service.edit_note_labels(note_id, request.label_ids)
        return SuccessResponse(
            success=True,
            message=f"Note {note_id} labels updated successfully"
        )
        
    except ValueError as e:
        # Check if it's a "not found" error
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )


@router.delete("/{note_id}", response_model=SuccessResponse)
def delete_drawer_note(
    note_id: int = Path(..., description="Note ID"),
    current_user: CurrentUser = Depends(require_drawer_notes_role)
):
    """
    Soft delete drawer note.
    
    Phase G Drawer Notes — Delete endpoint.
    
    This is a soft delete (sets IsDeleted = 1).
    - Note remains in database
    - Excluded from list endpoint
    - Can still be retrieved by ID
    - Cannot be edited after deletion
    - Label links remain intact
    
    Args:
        note_id: Note unique identifier
        current_user: Authenticated user with SOFTWARE_ADMIN or WORKER role
        
    Returns:
        Success response
        
    Security:
        - Requires authentication
        - Requires SOFTWARE_ADMIN or WORKER role
        
    Raises:
        404: Note not found
        401: Not authenticated
        403: Not authorized (missing required role)
    """
    try:
        drawer_note_service.soft_delete_note(note_id)
        return SuccessResponse(
            success=True,
            message=f"Note {note_id} deleted successfully"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get("/export/word")
def export_drawer_notes_word(
    current_user: CurrentUser = Depends(require_drawer_notes_role)
):
    """
    Export all drawer notes to Word document.
    
    **Phase G — G-B10**
    
    Generates a Word (.docx) document containing all non-deleted drawer notes
    with their labels. Document includes:
    - System/hospital name
    - Document title "Drawer Notes Registry"
    - Generated timestamp
    - For each note: ID, created date, author, labels, text
    - Total note count
    
    The export uses the same Word generation patterns as other reports in the system.
    
    Args:
        current_user: Authenticated user with SOFTWARE_ADMIN or WORKER role
        
    Returns:
        Word document as binary stream with proper headers
        
    Security:
        - Requires authentication
        - Requires SOFTWARE_ADMIN or WORKER role
        
    Raises:
        500: Export generation failed
        401: Not authenticated
        403: Not authorized (missing required role)
    """
    try:
        # Build Word export
        word_bytes = build_drawer_notes_word_export()
        
        # Return as downloadable Word document
        return Response(
            content=word_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": "attachment; filename=\"drawer_notes_export.docx\""
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate export: {str(e)}"
        )
