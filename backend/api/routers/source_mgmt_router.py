"""
Source Management Router
Settings endpoints for managing APP_LOOKUP_SOURCE entries.
Mirrors classification_mgmt_router.py -- Source is a flat lookup, so there
is no "list grouped by parent" here, just a flat list.

Authorized: SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..services.auth_service import get_current_user, CurrentUser
from ..utils.guards import require_role
from core.constants.roles import SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR
from ..services.source_mgmt_service import (
    list_sources,
    add_source,
    update_source,
    freeze_source,
    unfreeze_source,
)

router = APIRouter(
    prefix="/api/settings/sources",
    tags=["settings-sources"],
)


class AddSourceRequest(BaseModel):
    name_ar: str
    name_en: str


class UpdateSourceRequest(BaseModel):
    name_ar: str
    name_en: str


# ==================== ENDPOINTS ====================

@router.get("/")
async def list_all_sources(
    current_user: CurrentUser = Depends(get_current_user),
):
    """List all sources (all states, including frozen)."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = list_sources()
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.post("/")
async def create_source(
    body: AddSourceRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Add a new source."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = add_source(body.name_ar, body.name_en)
    if not result.get("success"):
        status_code = 400 if result.get("error") in ("VALIDATION_ERROR", "DUPLICATE_NAME") else 500
        raise HTTPException(status_code=status_code, detail=result)
    return result


@router.put("/{source_id}")
async def edit_source(
    source_id: int,
    body: UpdateSourceRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Update Arabic and English name of a source."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = update_source(source_id, body.name_ar, body.name_en)
    if not result.get("success"):
        status_code = 404 if result.get("error") == "NOT_FOUND" else 400
        raise HTTPException(status_code=status_code, detail=result)
    return result


@router.put("/{source_id}/freeze")
async def freeze(
    source_id: int,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Freeze a source so it no longer appears in new case/notice forms."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = freeze_source(source_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    return result


@router.put("/{source_id}/unfreeze")
async def unfreeze(
    source_id: int,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Unfreeze a source so it reappears in case/notice forms."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = unfreeze_source(source_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    return result
