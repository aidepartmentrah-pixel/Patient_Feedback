"""
AUTO-GENERATED API CONTRACT

Source: Obsidian – Record Search & Edit Page
Iteration: 1
Status: API skeleton only – no implementation
"""

from datetime import date, datetime
from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Query, Path
from pydantic import BaseModel


router = APIRouter(prefix="/api/records", tags=["records"])


# -----------------------------
# Enums
# -----------------------------

class RecordStatusEnum(str, Enum):
    Open = "Open"
    InProgress = "In Progress"
    Closed = "Closed"


class SeverityLevelEnum(str, Enum):
    Low = "Low"
    Medium = "Medium"
    High = "High"


class SourceEnum(str, Enum):
    Phone = "Phone"
    Email = "Email"
    Web = "Web"
    InPerson = "In Person"


class CareStageEnum(str, Enum):
    Admission = "Admission"
    Care = "Care"
    Discharge = "Discharge"


class HarmLevelEnum(str, Enum):
    NoHarm = "No Harm"
    Minor = "Minor"
    Major = "Major"


class ImprovementOpportunityEnum(str, Enum):
    Yes = "Yes"
    No = "No"


# -----------------------------
# Search Models (Lightweight)
# -----------------------------

class RecordSearchItem(BaseModel):
    record_id: str
    feedback_received_date: date
    patient_full_name: str
    issuing_department: str
    target_department: Optional[str] = None
    status: RecordStatusEnum
    severity_level: SeverityLevelEnum


class RecordSearchResponse(BaseModel):
    records: List[RecordSearchItem]
    total: int


# -----------------------------
# Full Record Models
# -----------------------------

class RecordDetailsResponse(BaseModel):
    record_id: str
    complaint_text: str
    immediate_action: str
    taken_action: str
    feedback_received_date: date
    issuing_department: str
    target_department: str
    source_1: SourceEnum
    status: RecordStatusEnum
    patient_full_name: str
    doctor_name: str
    other_entities: Optional[str] = None
    category: str
    sub_category: str
    classification_ar: str
    severity_level: SeverityLevelEnum
    stage: CareStageEnum
    harm_level: HarmLevelEnum
    improvement_opportunity_type: ImprovementOpportunityEnum
    created_at: datetime
    last_updated_at: datetime
    last_updated_by: str


# -----------------------------
# Update Payload Models
# -----------------------------

class RecordUpdateRequest(BaseModel):
    complaint_text: Optional[str] = None
    immediate_action: Optional[str] = None
    taken_action: Optional[str] = None
    feedback_received_date: Optional[date] = None
    issuing_department: Optional[str] = None
    target_department: Optional[str] = None
    source_1: Optional[SourceEnum] = None
    status: Optional[RecordStatusEnum] = None
    patient_full_name: Optional[str] = None
    doctor_name: Optional[str] = None
    other_entities: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    classification_ar: Optional[str] = None
    severity_level: Optional[SeverityLevelEnum] = None
    stage: Optional[CareStageEnum] = None
    harm_level: Optional[HarmLevelEnum] = None
    improvement_opportunity_type: Optional[ImprovementOpportunityEnum] = None


class RecordUpdateSuccessResponse(BaseModel):
    success: bool
    message: str
    record_id: str
    updated_fields: List[str]
    last_updated_at: datetime
    last_updated_by: str


class RecordUpdateErrorResponse(BaseModel):
    success: bool
    error: str
    message: str
    field: Optional[str] = None
    last_updated_at: Optional[datetime] = None
    last_updated_by: Optional[str] = None


# -----------------------------
# API Endpoints
# -----------------------------

@router.get(
    "/search",
    response_model=RecordSearchResponse
)
def search_records(
    query: Optional[str] = Query(None),
    record_id: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    status: Optional[RecordStatusEnum] = Query(None),
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    limit: Optional[int] = Query(50),
):
    raise NotImplementedError


@router.get(
    "/{record_id}",
    response_model=RecordDetailsResponse
)
def get_record_by_id(
    record_id: str = Path(...)
):
    raise NotImplementedError


@router.put(
    "/{record_id}",
    response_model=RecordUpdateSuccessResponse
)
def update_record_full(
    record_id: str = Path(...),
    payload: RecordUpdateRequest = ...
):
    raise NotImplementedError


@router.patch(
    "/{record_id}",
    response_model=RecordUpdateSuccessResponse
)
def update_record_partial(
    record_id: str = Path(...),
    payload: RecordUpdateRequest = ...
):
    raise NotImplementedError


"""
AUTO-GENERATED API CONTRACT

Source: Obsidian – Insert Record Page
Iteration: 1
Status: API skeleton only – no implementation
"""

from datetime import date, datetime
from typing import Optional, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/records", tags=["Records"])


# -------------------------
# Request Models
# -------------------------

class CreateRecordRequest(BaseModel):
    complaint_text: str = Field(..., min_length=1)
    immediate_action: Optional[str] = None
    taken_action: Optional[str] = None

    feedback_received_date: date

    issuing_department_id: Optional[int] = None
    target_department_id: Optional[int] = None

    source_id: Optional[int] = None
    in_out: Optional[str] = None
    worker_type: Optional[str] = None

    patient_name: Optional[str] = None
    doctor_name: Optional[str] = None

    domain_id: int
    category_id: int
    subcategory_id: Optional[int] = None
    classification_id: Optional[int] = None

    severity_id: int
    stage_id: Optional[int] = None
    harm_id: Optional[int] = None
    improvement_type: Optional[int] = None


# -------------------------
# Response Models
# -------------------------

class CreateRecordSuccessResponse(BaseModel):
    success: bool
    message: str
    record_id: str
    id: int
    status_id: int
    created_at: datetime


class ErrorResponse(BaseModel):
    success: bool
    error: str
    message: str
    field: Optional[str] = None


# -------------------------
# Routes
# -------------------------

@router.post(
    "/add",
    response_model=CreateRecordSuccessResponse,
)
def create_record(payload: CreateRecordRequest):
    """
    Create a new incident / feedback record.
    """
    raise NotImplementedError
