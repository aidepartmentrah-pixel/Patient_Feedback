"""
AUTO-GENERATED API CONTRACT

Source: Obsidian – Incident & Seasonal Explanations Page
Iteration: 1
Status: API skeleton only – no implementation
"""

from datetime import date
from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Query, Path
from pydantic import BaseModel


router = APIRouter(prefix="/api/department-feedback", tags=["department-feedback"])


# -----------------------------
# Enums
# -----------------------------

class SeverityEnum(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class IncidentStatusEnum(str, Enum):
    OPEN = "OPEN"
    OVERDUE = "OVERDUE"
    CLOSED = "CLOSED"


class ProblemDomainEnum(str, Enum):
    CLINICAL = "CLINICAL"
    MANAGEMENT = "MANAGEMENT"
    RELATIONAL = "RELATIONAL"


class PriorityEnum(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SeasonalViolationStatusEnum(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"


# -----------------------------
# Shared Models
# -----------------------------

class ActionItem(BaseModel):
    description: str
    responsible_person: str
    due_date: date
    priority: Optional[PriorityEnum] = None


# -----------------------------
# Incident Explanation Models
# -----------------------------

class IncidentRecord(BaseModel):
    id: str
    complaintID: str
    dateReceived: date
    patientName: str
    patientFullName: str
    targetDepartment: str
    qism: str
    severity: SeverityEnum
    status: IncidentStatusEnum
    daysSinceReceived: int
    isDelayed: bool
    problemDomain: ProblemDomainEnum
    problemCategory: str
    subCategory: str
    classificationAr: str
    rawContent: str
    immediateAction: str
    isRedFlag: bool


class IncidentListResponse(BaseModel):
    records: List[IncidentRecord]
    total: int


class IncidentDetailsResponse(IncidentRecord):
    pass


class IncidentExplanationRequest(BaseModel):
    explanation_text: str
    corrective_actions: str
    action_items: Optional[List[ActionItem]] = None
    contributing_factors: Optional[str] = None
    lessons_learned: Optional[str] = None
    responsible_person: Optional[str] = None
    expected_completion_date: Optional[date] = None


class IncidentExplanationResponse(BaseModel):
    success: bool
    message: str
    explanation_id: str
    follow_up_actions_created: int
    record_closed: Optional[bool] = None


# -----------------------------
# Seasonal Violation Models
# -----------------------------

class SeasonalViolationRecord(BaseModel):
    id: str
    season: str
    seasonLabel: str
    department: str
    qism: str
    metricType: str
    metricLabel: str
    thresholdValue: float
    actualValue: float
    status: SeasonalViolationStatusEnum
    totalRecords: int
    violationCount: Optional[int] = None


class SeasonalViolationsResponse(BaseModel):
    violations: List[SeasonalViolationRecord]
    total: int


class SeasonalExplanationRequest(BaseModel):
    root_cause_analysis: str
    corrective_actions: str
    action_items: Optional[List[ActionItem]] = None
    responsible_person: Optional[str] = None
    expected_completion_date: Optional[date] = None


class SeasonalExplanationResponse(BaseModel):
    success: bool
    message: str
    explanation_id: str
    follow_up_actions_created: int
    violation_status: SeasonalViolationStatusEnum


# -----------------------------
# API Endpoints – Incident Explanations
# -----------------------------

@router.get(
    "",
    response_model=IncidentListResponse
)
def get_open_incident_records(
    search: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    severity: Optional[SeverityEnum] = Query(None),
    status: Optional[IncidentStatusEnum] = Query(None),
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
):
    raise NotImplementedError


@router.get(
    "/{incident_id}",
    response_model=IncidentDetailsResponse
)
def get_incident_details(
    incident_id: str = Path(...)
):
    raise NotImplementedError


@router.post(
    "/{incident_id}/add",
    response_model=IncidentExplanationResponse
)
def submit_incident_explanation(
    incident_id: str = Path(...),
    payload: IncidentExplanationRequest = ...
):
    raise NotImplementedError


@router.post(
    "/{incident_id}/close",
    response_model=IncidentExplanationResponse
)
def submit_and_close_incident_explanation(
    incident_id: str = Path(...),
    payload: IncidentExplanationRequest = ...
):
    raise NotImplementedError


# -----------------------------
# API Endpoints – Seasonal Explanations
# -----------------------------

@router.get(
    "/seasonal-violations",
    response_model=SeasonalViolationsResponse
)
def get_seasonal_violations(
    season: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    status: Optional[SeasonalViolationStatusEnum] = Query(None),
):
    raise NotImplementedError


@router.post(
    "/seasonal/{violation_id}/submit",
    response_model=SeasonalExplanationResponse
)
def submit_seasonal_explanation(
    violation_id: str = Path(...),
    payload: SeasonalExplanationRequest = ...
):
    raise NotImplementedError
