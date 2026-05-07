"""
Incident Router
Endpoints for reading Incident parent records and their linked Cases.
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends
from typing import Optional

from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_logged_in
from ..db_layer.incident_parent import get_incident_parent, list_incidents


router = APIRouter(prefix="/api/incidents", tags=["Incidents"])


@router.get("")
async def get_incidents(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    search: Optional[str] = Query(None),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    List all incidents with their basic info and linked case IDs.
    Supports pagination and search by patient name or incident number.
    """
    require_logged_in(current_user)
    try:
        result = list_incidents(page=page, page_size=page_size, search=search)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "INTERNAL_ERROR", "message": str(e)})


@router.get("/{incident_id}")
async def get_incident(
    incident_id: int = Path(..., gt=0),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get a single Incident with all its linked Cases.
    """
    require_logged_in(current_user)
    try:
        result = get_incident_parent(incident_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "NOT_FOUND", "message": f"Incident {incident_id} not found"},
            )
        return {"success": True, "incident": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "INTERNAL_ERROR", "message": str(e)})
