"""
AUTO-GENERATED API CONTRACT

Source: Obsidian – Follow-Up Actions Page
Iteration: 1
Status: API skeleton only – no implementation
"""

from datetime import date, datetime
from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Query, Path
from pydantic import BaseModel


router = APIRouter(prefix="/api/follow-up", tags=["follow-up"])


# -----------------------------
# Enums
# -----------------------------

class ActionSourceTypeEnum(str, Enum):
    incident_explanation = "incident_explanation"
    seasonal_explanation = "seasonal_explanation"
    investigation = "investigation"
    manual = "manual"


class ActionPriorityEnum(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class ActionStatusEnum(str, Enum):
    pending = "pending"
    delayed = "delayed"
    completed = "completed"


class ActionGroupCategoryEnum(str, Enum):
    overdue = "overdue"
    next_7_days = "next_7_days"
    next_8_14_days = "next_8_14_days"
    future = "future"


# -----------------------------
# Core Action Models
# -----------------------------

class FollowUpAction(BaseModel):
    id: str
    actionTitle: str
    actionDescription: Optional[str] = None
    sourceType: ActionSourceTypeEnum
    sourceId: str
    department: str
    assignedTo: str
    priority: ActionPriorityEnum
    status: ActionStatusEnum
    dueDate: date
    completedDate: Optional[date] = None
    notes: Optional[str] = None
    createdAt: datetime
    createdBy: str
    lastUpdatedAt: datetime
    lastUpdatedBy: str

    # Derived / convenience fields
    isOverdue: Optional[bool] = None
    daysRemaining: Optional[int] = None
    daysOverdue: Optional[int] = None
    groupCategory: Optional[ActionGroupCategoryEnum] = None


# -----------------------------
# List & Statistics Models
# -----------------------------

class FollowUpStatistics(BaseModel):
    actionsToTake: int
    overdue: int
    completed: int


class FollowUpActionsResponse(BaseModel):
    actions: List[FollowUpAction]
    total: int
    statistics: FollowUpStatistics


# -----------------------------
# Single Action Details Models
# -----------------------------

class ActionSourceDetails(BaseModel):
    type: ActionSourceTypeEnum
    recordId: str
    incidentDate: Optional[date] = None
    department: Optional[str] = None


class ActionUpdateHistoryItem(BaseModel):
    timestamp: datetime
    updatedBy: str
    action: str
    note: Optional[str] = None


class FollowUpActionDetailsResponse(FollowUpAction):
    sourceDetails: Optional[ActionSourceDetails] = None
    updateHistory: Optional[List[ActionUpdateHistoryItem]] = None


# -----------------------------
# Update / Command Models
# -----------------------------

class FollowUpActionUpdateRequest(BaseModel):
    actionTitle: Optional[str] = None
    actionDescription: Optional[str] = None
    department: Optional[str] = None
    assignedTo: Optional[str] = None
    priority: Optional[ActionPriorityEnum] = None
    status: Optional[ActionStatusEnum] = None
    dueDate: Optional[date] = None
    notes: Optional[str] = None


class FollowUpActionUpdateResponse(BaseModel):
    success: bool
    message: str
    action: FollowUpAction


class ActionCompletionRequest(BaseModel):
    completionNotes: Optional[str] = None
    completedDate: Optional[date] = None


class ActionDelayRequest(BaseModel):
    delayDays: int
    reason: Optional[str] = None


class ActionSimpleSuccessResponse(BaseModel):
    success: bool
    message: str
    action_id: Optional[str] = None
    action: Optional[FollowUpAction] = None


# -----------------------------
# API Endpoints
# -----------------------------

@router.get(
    "/actions",
    response_model=FollowUpActionsResponse
)
def get_follow_up_actions(
    status: Optional[ActionStatusEnum] = Query(None),
    priority: Optional[ActionPriorityEnum] = Query(None),
    department: Optional[str] = Query(None),
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    include_completed: Optional[bool] = Query(True),
    source_type: Optional[ActionSourceTypeEnum] = Query(None),
    source_id: Optional[str] = Query(None),
):
    raise NotImplementedError


@router.get(
    "/actions/{action_id}",
    response_model=FollowUpActionDetailsResponse
)
def get_follow_up_action_by_id(
    action_id: str = Path(...)
):
    raise NotImplementedError


@router.patch(
    "/actions/{action_id}",
    response_model=FollowUpActionUpdateResponse
)
def update_follow_up_action(
    action_id: str = Path(...),
    payload: FollowUpActionUpdateRequest = ...
):
    raise NotImplementedError


@router.post(
    "/actions/{action_id}/complete",
    response_model=ActionSimpleSuccessResponse
)
def mark_action_completed(
    action_id: str = Path(...),
    payload: Optional[ActionCompletionRequest] = None
):
    raise NotImplementedError


@router.post(
    "/actions/{action_id}/delay",
    response_model=ActionSimpleSuccessResponse
)
def delay_action(
    action_id: str = Path(...),
    payload: ActionDelayRequest = ...
):
    raise NotImplementedError


@router.delete(
    "/actions/{action_id}",
    response_model=ActionSimpleSuccessResponse
)
def delete_or_close_action(
    action_id: str = Path(...),
    soft_delete: Optional[bool] = Query(True),
):
    raise NotImplementedError
