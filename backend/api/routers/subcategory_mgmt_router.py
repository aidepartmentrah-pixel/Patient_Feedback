"""
Sub-Category Management Router
Settings endpoints for managing APP_LOOKUP_SUBCATEGORY entries.
Mirrors classification_mgmt_router.py one level up the hierarchy.

Authorized: SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..services.auth_service import get_current_user, CurrentUser
from ..utils.guards import require_role
from core.constants.roles import SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR
from ..services.subcategory_mgmt_service import (
    list_subcategories_grouped,
    add_subcategory,
    update_subcategory,
    freeze_subcategory,
    unfreeze_subcategory,
)

router = APIRouter(
    prefix="/api/settings/subcategories",
    tags=["settings-subcategories"],
)


class AddSubcategoryRequest(BaseModel):
    category_id: int
    name_ar: str
    name_en: Optional[str] = None


class UpdateSubcategoryRequest(BaseModel):
    name_ar: str
    name_en: Optional[str] = None


# ==================== ENDPOINTS ====================

@router.get("/")
async def list_subcategories(
    current_user: CurrentUser = Depends(get_current_user),
):
    """List all subcategories grouped by category (all states, including frozen)."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = list_subcategories_grouped()
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.post("/")
async def create_subcategory(
    body: AddSubcategoryRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Add a new subcategory under an existing category."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = add_subcategory(body.category_id, body.name_ar, body.name_en)
    if not result.get("success"):
        status_code = 400 if result.get("error") in ("VALIDATION_ERROR", "DUPLICATE_NAME", "INVALID_CATEGORY") else 500
        raise HTTPException(status_code=status_code, detail=result)
    return result


@router.put("/{subcategory_id}")
async def edit_subcategory(
    subcategory_id: int,
    body: UpdateSubcategoryRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Update Arabic and/or English name of a subcategory."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = update_subcategory(subcategory_id, body.name_ar, body.name_en)
    if not result.get("success"):
        status_code = 404 if result.get("error") == "NOT_FOUND" else 400
        raise HTTPException(status_code=status_code, detail=result)
    return result


@router.put("/{subcategory_id}/freeze")
async def freeze(
    subcategory_id: int,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Freeze a subcategory so it no longer appears in new case/notice forms."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = freeze_subcategory(subcategory_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    return result


@router.put("/{subcategory_id}/unfreeze")
async def unfreeze(
    subcategory_id: int,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Unfreeze a subcategory so it reappears in case/notice forms."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = unfreeze_subcategory(subcategory_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    return result
