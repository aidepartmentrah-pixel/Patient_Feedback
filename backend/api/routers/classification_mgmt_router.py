"""
Classification Management Router
Settings endpoints for managing APP_LOOKUP_CLASSIFICATION entries.

Authorized: SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..services.auth_service import get_current_user, CurrentUser
from ..utils.guards import require_role
from core.constants.roles import SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR
from ..services.classification_mgmt_service import (
    list_classifications_grouped,
    add_classification,
    update_classification,
    freeze_classification,
    unfreeze_classification,
)

router = APIRouter(
    prefix="/api/settings/classifications",
    tags=["settings-classifications"],
)


class AddClassificationRequest(BaseModel):
    subcategory_id: int
    name_ar: str
    name_en: Optional[str] = None


class UpdateClassificationRequest(BaseModel):
    name_ar: str
    name_en: Optional[str] = None


# ==================== ENDPOINTS ====================

@router.get("/")
async def list_classifications(
    current_user: CurrentUser = Depends(get_current_user),
):
    """List all classifications grouped by subcategory (all states, including frozen)."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = list_classifications_grouped()
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.post("/")
async def create_classification(
    body: AddClassificationRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Add a new classification under an existing subcategory."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = add_classification(body.subcategory_id, body.name_ar, body.name_en)
    if not result.get("success"):
        status_code = 400 if result.get("error") in ("VALIDATION_ERROR", "DUPLICATE_NAME", "INVALID_SUBCATEGORY") else 500
        raise HTTPException(status_code=status_code, detail=result)
    return result


@router.put("/{classification_id}")
async def edit_classification(
    classification_id: int,
    body: UpdateClassificationRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Update Arabic and/or English name of a classification."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = update_classification(classification_id, body.name_ar, body.name_en)
    if not result.get("success"):
        status_code = 404 if result.get("error") == "NOT_FOUND" else 400
        raise HTTPException(status_code=status_code, detail=result)
    return result


@router.put("/{classification_id}/freeze")
async def freeze(
    classification_id: int,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Freeze a classification so it no longer appears in new case/notice forms."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = freeze_classification(classification_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    return result


@router.put("/{classification_id}/unfreeze")
async def unfreeze(
    classification_id: int,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Unfreeze a classification so it reappears in case/notice forms."""
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    result = unfreeze_classification(classification_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    return result
