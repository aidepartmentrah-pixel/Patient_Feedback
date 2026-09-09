"""
Category Management Router
Settings endpoints for managing APP_LOOKUP_CATEGORY entries.
Mirrors classification_mgmt_router.py one level up the hierarchy.

Authorized: SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..services.auth_service import get_current_user, CurrentUser
from ..utils.guards import require_role
from core.constants.roles import SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR
from ..services.category_mgmt_service import (
    list_categories_grouped,
    add_category,
    update_category,
    freeze_category,
    unfreeze_category,
)

router = APIRouter(
    prefix="/api/settings/categories",
    tags=["settings-categories"],
)


class AddCategoryRequest(BaseModel):
    domain_id: int
    name_ar: str
    name_en: Optional[str] = None


class UpdateCategoryRequest(BaseModel):
    name_ar: str
    name_en: Optional[str] = None


# ==================== ENDPOINTS ====================

@router.get("/")
async def list_categories(
    current_user: CurrentUser = Depends(get_current_user),
):
    """List all categories grouped by domain (all states, including frozen)."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = list_categories_grouped()
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.post("/")
async def create_category(
    body: AddCategoryRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Add a new category under an existing domain."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = add_category(body.domain_id, body.name_ar, body.name_en)
    if not result.get("success"):
        status_code = 400 if result.get("error") in ("VALIDATION_ERROR", "DUPLICATE_NAME", "INVALID_DOMAIN") else 500
        raise HTTPException(status_code=status_code, detail=result)
    return result


@router.put("/{category_id}")
async def edit_category(
    category_id: int,
    body: UpdateCategoryRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Update Arabic and/or English name of a category."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = update_category(category_id, body.name_ar, body.name_en)
    if not result.get("success"):
        status_code = 404 if result.get("error") == "NOT_FOUND" else 400
        raise HTTPException(status_code=status_code, detail=result)
    return result


@router.put("/{category_id}/freeze")
async def freeze(
    category_id: int,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Freeze a category so it no longer appears in new case/notice forms."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = freeze_category(category_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    return result


@router.put("/{category_id}/unfreeze")
async def unfreeze(
    category_id: int,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Unfreeze a category so it reappears in case/notice forms."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = unfreeze_category(category_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    return result
