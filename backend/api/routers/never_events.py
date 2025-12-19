"""
AUTO-GENERATED API CONTRACT

Source: Obsidian – Never Events Page
Iteration: 1
Status: API skeleton only – no implementation
"""

from datetime import date, datetime
from typing import Optional, List, Dict, Union

from fastapi import APIRouter, Query, Path
from pydantic import BaseModel

router = APIRouter(prefix="/api/never-events", tags=["Never Events"])


# =====================================================
# Core Models
# =====================================================

class NeverEventListItem(BaseModel):
    id: Union[int, str]
    recordID: str
    date: date
    patientName: str
    patientID: Optional[str] = None
    neverEventType: str
    neverEventTypeAr: str
    neverEventCategory: str
    status: str
    severity: str
    department: str
    qism: str
    incidentID: str


class NeverEventListResponse(BaseModel):
    never_events: List[NeverEventListItem]
    total: int
    limit: int
    offset: int


# =====================================================
# KPI / Statistics Models
# =====================================================

class MonthlyCount(BaseModel):
    count: int
    month: str


class NeverEventStatisticsResponse(BaseModel):
    total_never_events: int
    unfinished_count: int
    finished_count: int
    by_status: Dict[str, int]
    by_category: Dict[str, int]
    by_severity: Dict[str, int]
    current_month: Optional[MonthlyCount] = None
    previous_month: Optional[MonthlyCount] = None
    period: Dict[str, date]


# =====================================================
# Trend Models
# =====================================================

class TrendPoint(BaseModel):
    period: str
    count: int


class TrendPointWithBreakdown(BaseModel):
    period: str
    total: int
    breakdown: Dict[str, int]


class NeverEventTrendResponse(BaseModel):
    granularity: str
    group_by: Optional[str] = None
    period: Optional[Dict[str, date]] = None
    data: List[Union[TrendPoint, TrendPointWithBreakdown]]


# =====================================================
# Detail View Models
# =====================================================

class IncidentDetails(BaseModel):
    incidentID: str
    complaintText: Optional[str] = None
    immediateAction: Optional[str] = None
    corrective_actions: Optional[str] = None
    rootCause: Optional[str] = None
    responsiblePerson: Optional[str] = None
    targetDepartment: Optional[str] = None
    feedbackReceivedDate: Optional[date] = None
    classification: Optional[str] = None


class TimelineEvent(BaseModel):
    date: datetime
    event: str
    user: str


class RelatedAction(BaseModel):
    action_id: str
    description: str
    status: str
    due_date: Optional[date] = None


class NeverEventDetailsResponse(BaseModel):
    never_event: NeverEventListItem
    incident_details: IncidentDetails
    timeline: List[TimelineEvent]
    related_actions: List[RelatedAction]


# =====================================================
# Routes
# =====================================================

@router.get(
    "",
    response_model=NeverEventListResponse,
)
def get_never_events(
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    department: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(100, ge=1),
    offset: int = Query(0, ge=0),
):
    """
    Fetch list of never events with optional filtering and search.
    """
    raise NotImplementedError


@router.get(
    "/statistics",
    response_model=NeverEventStatisticsResponse,
)
def get_never_event_statistics(
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
):
    """
    Fetch KPI summary statistics for never events.
    """
    raise NotImplementedError


@router.get(
    "/trends",
    response_model=NeverEventTrendResponse,
)
def get_never_event_trends(
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    granularity: str = Query("monthly"),
    group_by: Optional[str] = Query("none"),
):
    """
    Fetch time-series trend data for never events.
    """
    raise NotImplementedError


@router.get(
    "/{never_event_id}",
    response_model=NeverEventDetailsResponse,
)
def get_never_event_details(
    never_event_id: Union[int, str] = Path(...),
):
    """
    Fetch full details for a single never event.
    """
    raise NotImplementedError
