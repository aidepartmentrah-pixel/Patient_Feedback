"""
Drawer Labels Router (API v2)
Phase G — G-B8 — Drawer Labels management endpoints.

This router provides label management endpoints under /api/v2/drawer-labels.

Endpoints exposed:
- POST /api/v2/drawer-labels - Create new label
- GET /api/v2/drawer-labels - List active labels
- DELETE /api/v2/drawer-labels/{label_id} - Disable label

Security: 
- All endpoints protected by authentication
- Requires SOFTWARE_ADMIN or WORKER role
"""

from fastapi import APIRouter, Path, Depends, HTTPException, status

from backend.api_v2.guards.drawer_notes_guards import require_drawer_notes_role
from backend.api.schemas.auth_models import CurrentUser
from backend.api_v2.services import drawer_label_service
from backend.api_v2.schemas.drawer_label_schemas import (
    CreateLabelRequest,
    LabelResponse,
    CreateLabelResponse,
    ListLabelsResponse
)
from backend.api_v2.schemas.drawer_note_schemas import SuccessResponse


# ============================================================
# ROUTER DEFINITION
# ============================================================
router = APIRouter(prefix="/api/v2/drawer-labels", tags=["Drawer Labels V2"])


# ============================================================
# ENDPOINTS
# ============================================================

@router.post("/", response_model=CreateLabelResponse, status_code=status.HTTP_201_CREATED)
def create_drawer_label(
    request: CreateLabelRequest,
    current_user: CurrentUser = Depends(require_drawer_notes_role)
):
    """
    Create a new drawer label.
    
    Phase G Drawer Labels — Create endpoint.
    
    Business Rules:
    - Label name is trimmed
    - Must be 2-100 characters
    - Must be unique (DB constraint)
    
    Args:
        request: Label creation request with name
        current_user: Authenticated user with SOFTWARE_ADMIN or WORKER role
        
    Returns:
        Created label ID
        
    Security:
        - Requires authentication
        - Requires SOFTWARE_ADMIN or WORKER role
        
    Raises:
        400: Validation error (too short, too long, duplicate)
        401: Not authenticated
        403: Not authorized (missing required role)
    """
    try:
        label_id = drawer_label_service.create_label(request.label_name)
        
        return CreateLabelResponse(label_id=label_id, success=True)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Catch DB constraint violations (duplicate labels)
        error_str = str(e).lower()
        if "unique" in error_str or "duplicate" in error_str or "constraint" in error_str:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Label name already exists"
            )
        else:
            raise


@router.get("/", response_model=ListLabelsResponse)
def list_drawer_labels(
    current_user: CurrentUser = Depends(require_drawer_notes_role)
):
    """
    List all active drawer labels.
    
    Phase G Drawer Labels — List endpoint.
    
    Returns only active labels (IsActive = 1).
    Disabled labels are excluded.
    
    Args:
        current_user: Authenticated user with SOFTWARE_ADMIN or WORKER role
        
    Returns:
        List of active labels
        
    Security:
        - Requires authentication
        - Requires SOFTWARE_ADMIN or WORKER role
        
    Raises:
        401: Not authenticated
        403: Not authorized (missing required role)
    """
    labels = drawer_label_service.list_active_labels()
    
    # Convert to response models
    label_responses = [
        LabelResponse(
            label_id=label['label_id'],
            label_name=label['label_name'],
            is_active=label['is_active'],
            created_at=label['created_at']
        )
        for label in labels
    ]
    
    return ListLabelsResponse(labels=label_responses, total=len(label_responses))


@router.delete("/{label_id}", response_model=SuccessResponse)
def disable_drawer_label(
    label_id: int = Path(..., description="Label ID to disable"),
    current_user: CurrentUser = Depends(require_drawer_notes_role)
):
    """
    Disable a drawer label (soft disable).
    
    Phase G Drawer Labels — Disable endpoint.
    
    This is a soft disable (sets IsActive = 0).
    - Label remains in database
    - Excluded from list endpoint
    - Cannot be used for new notes
    - Existing note-label links remain intact
    
    Args:
        label_id: Label unique identifier
        current_user: Authenticated user with SOFTWARE_ADMIN or WORKER role
        
    Returns:
        Success response
        
    Security:
        - Requires authentication
        - Requires SOFTWARE_ADMIN or WORKER role
        
    Raises:
        401: Not authenticated
        403: Not authorized (missing required role)
    """
    drawer_label_service.disable_label(label_id)
    
    return SuccessResponse(
        success=True,
        message=f"Label {label_id} disabled successfully"
    )
