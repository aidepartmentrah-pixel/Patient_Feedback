"""
AUTO-GENERATED API CONTRACT

Source: Obsidian – TableView
Iteration: 1
Status: API skeleton only – no implementation
"""

from datetime import date, datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Query, Path, Body
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/complaints", tags=["TableView"])


# ------------------------------------------------------------------
# Response Models – Core Table Row
# ------------------------------------------------------------------

class ComplaintTableRow(BaseModel):
    id: int
    complaint_number: str
    complaint_summary: str

    received_date: date
    received_datetime: datetime
    incident_date: Optional[date]

    patient_mrn: str
    patient_name: str
    patient_age: Optional[int]
    patient_gender: Optional[str]
    patient_gender_ar: Optional[str]

    issuing_dept_id: int
    issuing_dept_name: str
    issuing_dept_name_ar: str
    issuing_dept_code: Optional[str]

    target_dept_id: int
    target_dept_name: str
    target_dept_name_ar: str
    target_dept_code: Optional[str]

    domain_id: int
    domain_name: str
    domain_name_ar: str

    category_id: int
    category_name: str
    category_name_ar: str

    classification_id: int
    classification_name: str
    classification_name_ar: str

    severity_id: int
    severity_name: str
    severity_name_ar: str
    severity_color: Optional[str]

    stage_id: int
    stage_name: str
    stage_name_ar: str

    harm_level: Optional[str]
    harm_level_ar: Optional[str]

    status: str
    status_ar: str
    status_display: Optional[str]

    is_closed: bool
    closure_date: Optional[date]

    is_red_flag: bool
    is_never_event: bool
    is_improvement_opportunity: bool

    priority: Optional[str]
    priority_ar: Optional[str]

    source: str
    source_ar: Optional[str]
    source_detail: Optional[str]

    days_open: Optional[int]
    days_to_closure: Optional[int]

    has_follow_up_actions: bool
    pending_actions_count: int
    delayed_actions_count: int

    has_explanation: bool
    explanation_status: Optional[str]

    can_edit: bool
    can_delete: bool

    updated_at: datetime


class PaginationMeta(BaseModel):
    page: int
    page_size: int
    total_records: int
    total_pages: int


class ComplaintsListResponse(BaseModel):
    complaints: List[ComplaintTableRow]
    pagination: PaginationMeta
    filters_applied: Dict[str, Any]
    view: str


# ------------------------------------------------------------------
# Filter Options
# ------------------------------------------------------------------

class FilterDepartmentOption(BaseModel):
    id: int
    name: str
    name_ar: str
    code: Optional[str]
    count: Optional[int]


class FilterValueOption(BaseModel):
    value: str
    label: str
    label_ar: str
    count: Optional[int]


class FilterSeverityOption(BaseModel):
    id: int
    name: str
    name_ar: str
    color: Optional[str]
    count: Optional[int]


class FilterDomainOption(BaseModel):
    id: int
    name: str
    name_ar: str
    count: Optional[int]


class FilterOptionsResponse(BaseModel):
    issuing_departments: List[FilterDepartmentOption]
    target_departments: List[FilterDepartmentOption]
    sources: List[FilterValueOption]
    statuses: List[FilterValueOption]
    severities: List[FilterSeverityOption]
    domains: List[FilterDomainOption]


# ------------------------------------------------------------------
# Single Record (Navigation to EditRecord)
# ------------------------------------------------------------------

class ComplaintDetailsResponse(BaseModel):
    id: int
    complaint_number: str
    complaint_text: str
    complaint_text_ar: Optional[str]

    received_date: date
    received_datetime: datetime
    incident_date: Optional[date]

    patient_id: int
    patient_mrn: str
    patient_name: str
    patient_age: Optional[int]
    patient_gender: Optional[str]
    patient_gender_ar: Optional[str]

    issuing_dept_id: int
    issuing_dept_name: str
    issuing_dept_name_ar: str

    target_dept_id: int
    target_dept_name: str
    target_dept_name_ar: str

    domain_id: int
    domain_name: str
    domain_name_ar: str

    category_id: int
    category_name: str
    category_name_ar: str

    classification_id: int
    classification_name: str
    classification_name_ar: str

    severity_id: int
    severity_name: str
    severity_name_ar: str

    stage_id: int
    stage_name: str
    stage_name_ar: str

    harm_level: Optional[str]
    harm_level_ar: Optional[str]

    status: str
    status_ar: str

    is_red_flag: bool
    is_never_event: bool

    priority: Optional[str]

    source: str
    source_ar: Optional[str]
    source_detail: Optional[str]
    reporter_name: Optional[str]

    created_at: datetime
    updated_at: datetime
    created_by_user_id: int
    updated_by_user_id: int


# ------------------------------------------------------------------
# Export & Count
# ------------------------------------------------------------------

class ExportRequest(BaseModel):
    format: str = Field(..., description="csv or json")
    filters: Dict[str, Any]
    columns: Optional[List[str]]
    include_patient_identifiers: bool = False
    language: Optional[str] = "en"


class ExportResponse(BaseModel):
    export_id: str
    file_name: str
    file_size_bytes: int
    download_url: str
    record_count: int
    generated_at: datetime
    expires_at: datetime
    audit_logged: bool


class CountResponse(BaseModel):
    total_count: int
    filters_applied: Dict[str, Any]


# ------------------------------------------------------------------
# View Presets
# ------------------------------------------------------------------

class TableViewDefinition(BaseModel):
    view_id: str
    view_name: str
    view_name_ar: str
    columns: List[str]
    default_sort: str
    default_sort_order: str
    preset_filters: Optional[Dict[str, Any]] = None


class TableViewsResponse(BaseModel):
    views: List[TableViewDefinition]
    default_view: str


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.get("", response_model=ComplaintsListResponse)
def fetch_complaints(
    search: Optional[str] = None,
    issuing_dept_id: Optional[int] = None,
    target_dept_id: Optional[int] = None,
    dayra_id: Optional[int] = None,
    source: Optional[str] = None,
    status: Optional[str] = None,
    severity_id: Optional[int] = None,
    domain_id: Optional[int] = None,
    category_id: Optional[int] = None,
    is_red_flag: Optional[bool] = None,
    is_never_event: Optional[bool] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    sort_by: str = "received_date",
    sort_order: str = "desc",
    page: int = 1,
    page_size: int = 50,
    view: str = "complete",
):
    raise NotImplementedError


@router.get("/filter-options", response_model=FilterOptionsResponse)
def fetch_filter_options():
    raise NotImplementedError


@router.get("/{id}", response_model=ComplaintDetailsResponse)
def fetch_single_complaint(
    id: int = Path(..., description="Complaint ID")
):
    raise NotImplementedError


@router.post("/export", response_model=ExportResponse)
def export_complaints(
    payload: ExportRequest = Body(...)
):
    raise NotImplementedError


@router.get("/count", response_model=CountResponse)
def count_complaints(
    search: Optional[str] = None,
    issuing_dept_id: Optional[int] = None,
    target_dept_id: Optional[int] = None,
    source: Optional[str] = None,
    status: Optional[str] = None,
    is_red_flag: Optional[bool] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
):
    raise NotImplementedError


@router.get("/views", response_model=TableViewsResponse)
def fetch_table_views():
    raise NotImplementedError
