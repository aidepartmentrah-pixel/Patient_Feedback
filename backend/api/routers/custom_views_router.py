"""
Custom Views Router
CRUD API for APP_CUSTOM_VIEWS to control table view columns.
"""

from fastapi import APIRouter, HTTPException, Body, Query, Path, Depends
from pydantic import BaseModel, Field
from typing import Optional

from ..services.custom_views_service import (
    create_view,
    get_view,
    list_views,
    update_view,
    delete_view,
)
from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_software_admin

router = APIRouter(prefix="/api/custom-views", tags=["Custom Views"])


class CustomViewBase(BaseModel):
    view_name: Optional[str] = Field(None, alias="ViewName")
    # Show* flags (optional booleans)
    ShowIncidentNumber: Optional[bool] = False
    ShowIncidentRequestCaseID: Optional[bool] = False
    ShowComplaintText: Optional[bool] = False
    ShowImmediateAction: Optional[bool] = False
    ShowTakenAction: Optional[bool] = False
    ShowFeedbackRecievedDate: Optional[bool] = False
    ShowPatientName: Optional[bool] = False
    ShowIssuingOrgUnitID: Optional[bool] = False
    ShowCreatedAt: Optional[bool] = False
    ShowCreatedByUserID: Optional[bool] = False
    ShowIsInPatient: Optional[bool] = False
    ShowClinicalRiskTypeID: Optional[bool] = False
    ShowFeedbackIntentTypeID: Optional[bool] = False
    ShowBuildingID: Optional[bool] = False
    ShowDomainID: Optional[bool] = False
    ShowCategoryID: Optional[bool] = False
    ShowSubCategoryID: Optional[bool] = False
    ShowClassificationID: Optional[bool] = False
    ShowSeverityID: Optional[bool] = False
    ShowStageID: Optional[bool] = False
    ShowHarmLevelID: Optional[bool] = False
    ShowCaseStatusID: Optional[bool] = False
    ShowSourceID: Optional[bool] = False
    ShowExplanationStatusID: Optional[bool] = False
    ShowSectionAnswer: Optional[bool] = False
    ShowDepartmentAnswer: Optional[bool] = False
    ShowAdministrationAnswer: Optional[bool] = False
    ShowTargetDepartment: Optional[bool] = False
    ShowSatisfactionStatus: Optional[bool] = False
    ShowSatisfactionDate: Optional[bool] = False
    ShowRedFlagIndicator: Optional[bool] = False
    ShowNeverEventIndicator: Optional[bool] = False
    ShowMorbidityIndicator: Optional[bool] = False
    ShowLateIndicator: Optional[bool] = False
    ShowForceClosedIndicator: Optional[bool] = False
    ShowLastEdited: Optional[bool] = False

    class Config:
        populate_by_name = True


class CreateCustomViewRequest(CustomViewBase):
    view_name: str = Field(..., alias="ViewName", min_length=1)


class UpdateCustomViewRequest(CustomViewBase):
    pass


@router.get("")
async def list_custom_views_endpoint(active_only: bool = Query(True)):
    return list_views(active_only=active_only)


@router.get("/{view_id}")
async def get_custom_view_endpoint(view_id: int = Path(..., gt=0)):
    result = get_view(view_id)
    if not result.get("success", False):
        raise HTTPException(status_code=404, detail=result)
    return result


@router.post("")
async def create_custom_view_endpoint(
    request: CreateCustomViewRequest = Body(...),
    current_user: CurrentUser = Depends(get_current_user),
):
    require_software_admin(current_user)
    payload = request.dict(by_alias=True)
    # Promote alias to expected key
    if "ViewName" not in payload and request.view_name:
        payload["ViewName"] = request.view_name

    result = create_view(payload)
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.put("/{view_id}")
async def update_custom_view_endpoint(
    view_id: int = Path(..., gt=0),
    request: UpdateCustomViewRequest = Body(...),
    current_user: CurrentUser = Depends(get_current_user),
):
    require_software_admin(current_user)
    payload = request.dict(by_alias=True, exclude_unset=True)
    if "ViewName" not in payload and request.view_name is not None:
        payload["ViewName"] = request.view_name

    result = update_view(view_id, payload)
    if not result.get("success", False):
        status = 404 if result.get("error") == "NOT_FOUND" else 400
        raise HTTPException(status_code=status, detail=result)
    return result


@router.delete("/{view_id}")
async def delete_custom_view_endpoint(
    view_id: int = Path(..., gt=0),
    hard: bool = Query(False),
    current_user: CurrentUser = Depends(get_current_user),
):
    require_software_admin(current_user)
    result = delete_view(view_id, hard=hard)
    if not result.get("success", False):
        status = 404 if result.get("error") == "NOT_FOUND" else 400
        raise HTTPException(status_code=status, detail=result)
    return result
