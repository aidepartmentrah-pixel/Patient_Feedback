"""
AUTO-GENERATED API CONTRACT

Source: Obsidian – Doctor Search & Analytics Page
Iteration: 1
Status: API skeleton only – no implementation
"""

from datetime import date
from enum import Enum
from typing import List, Optional, Union

from fastapi import APIRouter, Query, Path
from pydantic import BaseModel


router = APIRouter(prefix="/api/doctors", tags=["doctors"])


# -----------------------------
# Enums
# -----------------------------

class DoctorStatusEnum(str, Enum):
    active = "active"
    inactive = "inactive"
    suspended = "suspended"


class SeverityEnum(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class IncidentStatusEnum(str, Enum):
    OPEN = "OPEN"
    UNDER_REVIEW = "UNDER_REVIEW"
    CLOSED = "CLOSED"


# -----------------------------
# Doctor Models
# -----------------------------

class DoctorListItem(BaseModel):
    id: Union[int, str]
    employeeId: str
    nameEn: str
    nameAr: str
    department: str
    specialty: str
    hireDate: date
    status: DoctorStatusEnum


class DoctorListResponse(BaseModel):
    doctors: List[DoctorListItem]
    total: int


class DoctorProfileResponse(BaseModel):
    id: Union[int, str]
    employeeId: str
    nameEn: str
    nameAr: str
    department: str
    specialty: str
    hireDate: date
    status: DoctorStatusEnum
    yearsOfService: int
    email: Optional[str] = None
    phone: Optional[str] = None
    licenseNumber: Optional[str] = None


# -----------------------------
# Doctor Statistics Models
# -----------------------------

class DoctorIncidentStatistics(BaseModel):
    total: int
    high: int
    medium: int
    low: int
    redFlags: int


class StatisticsPeriod(BaseModel):
    from_date: date
    to_date: date


class DoctorStatisticsResponse(BaseModel):
    statistics: DoctorIncidentStatistics
    period: StatisticsPeriod


# -----------------------------
# Doctor Analytics Models
# -----------------------------

class CategoryBreakdownItem(BaseModel):
    name: str
    count: int


class MonthlyTrendItem(BaseModel):
    month: str
    count: int


class DoctorAnalyticsResponse(BaseModel):
    categoryBreakdown: List[CategoryBreakdownItem]
    monthlyTrend: List[MonthlyTrendItem]
    period: StatisticsPeriod


# -----------------------------
# Doctor Incidents Models
# -----------------------------

class DoctorIncidentRecord(BaseModel):
    id: Union[int, str]
    date: date
    incidentId: str
    patientId: str
    category: str
    categoryAr: str
    severity: SeverityEnum
    status: IncidentStatusEnum
    isRedFlag: bool


class DoctorIncidentsResponse(BaseModel):
    incidents: List[DoctorIncidentRecord]
    total: int
    limit: int
    offset: int


# -----------------------------
# Combined Full Report Model
# -----------------------------

class DoctorFullReportResponse(BaseModel):
    profile: DoctorProfileResponse
    statistics: DoctorStatisticsResponse
    analytics: DoctorAnalyticsResponse
    incidents: DoctorIncidentsResponse


# -----------------------------
# API Endpoints
# -----------------------------

@router.get(
    "",
    response_model=DoctorListResponse
)
def search_doctors(
    query: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    status: Optional[DoctorStatusEnum] = Query(None),
    limit: Optional[int] = Query(50),
):
    raise NotImplementedError


@router.get(
    "/{doctor_id}/profile",
    response_model=DoctorProfileResponse
)
def get_doctor_profile(
    doctor_id: Union[int, str] = Path(...)
):
    raise NotImplementedError


@router.get(
    "/{doctor_id}/statistics",
    response_model=DoctorStatisticsResponse
)
def get_doctor_statistics(
    doctor_id: Union[int, str] = Path(...),
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
):
    raise NotImplementedError


@router.get(
    "/{doctor_id}/analytics",
    response_model=DoctorAnalyticsResponse
)
def get_doctor_analytics(
    doctor_id: Union[int, str] = Path(...),
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
):
    raise NotImplementedError


@router.get(
    "/{doctor_id}/incidents",
    response_model=DoctorIncidentsResponse
)
def get_doctor_incidents(
    doctor_id: Union[int, str] = Path(...),
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    severity: Optional[SeverityEnum] = Query(None),
    status: Optional[IncidentStatusEnum] = Query(None),
    red_flags_only: Optional[bool] = Query(False),
    limit: Optional[int] = Query(100),
    offset: Optional[int] = Query(0),
):
    raise NotImplementedError


@router.get(
    "/{doctor_id}/full-report",
    response_model=DoctorFullReportResponse
)
def get_doctor_full_report(
    doctor_id: Union[int, str] = Path(...),
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
):
    raise NotImplementedError
