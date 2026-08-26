"""
Incident Router
Endpoints for reading Incident parent records and their linked Cases.
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends
from typing import Optional

from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_logged_in, require_role
from ..db_layer.incident_parent import get_incident_parent, list_incidents, add_case_to_incident, delete_case_from_incident
from core.constants.roles import SOFTWARE_ADMIN, WORKER, COMPLAINT_SUPERVISOR
from ..services.table_view_service import get_cases_for_incident


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


@router.get("/{incident_id}/cases")
async def get_incident_cases(
    incident_id: int = Path(..., gt=0),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get all cases for an incident with full field data (same shape as GET /api/records/{id}).
    Used by EditRecord to load the complete incident with all its cases.
    """
    require_logged_in(current_user)
    try:
        cases = get_cases_for_incident(incident_id)
        if not cases:
            raise HTTPException(
                status_code=404,
                detail={"error": "NOT_FOUND", "message": f"No cases found for incident {incident_id}"},
            )
        return {"success": True, "cases": cases}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "INTERNAL_ERROR", "message": str(e)})


@router.post("/{incident_id}/cases")
async def create_case_for_incident(
    incident_id: int = Path(..., gt=0),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Add a new blank Draft case to an existing incident.
    Inherits patient, building, source, and dates from the incident.
    Returns the new case_id.
    """
    require_logged_in(current_user)
    require_role(current_user, [SOFTWARE_ADMIN, WORKER, COMPLAINT_SUPERVISOR])
    try:
        new_case_id = add_case_to_incident(incident_id, current_user.user_id)
        return {"success": True, "case_id": new_case_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail={"error": "NOT_FOUND", "message": str(e)})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "INTERNAL_ERROR", "message": str(e)})


@router.delete("/{incident_id}/cases/{case_id}")
async def delete_case(
    incident_id: int = Path(..., gt=0),
    case_id: int = Path(..., gt=0),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Delete one case from an incident. Always rejected if it's the
    incident's only case. Draft/Ready to Send cases are hard-deleted;
    already-published cases are soft-deleted (kept, closed, subcase(s)
    retired) instead — see delete_case_from_incident() for details.
    """
    require_logged_in(current_user)
    require_role(current_user, [SOFTWARE_ADMIN, WORKER, COMPLAINT_SUPERVISOR])
    try:
        result = delete_case_from_incident(incident_id, case_id, current_user.user_id)
        return {"success": True, **result}
    except ValueError as e:
        raise HTTPException(status_code=409, detail={"error": "CANNOT_DELETE", "message": str(e)})
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
