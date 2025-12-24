from fastapi import APIRouter
from backend.api.services.admin_hierarchy_service import get_dashboard_hierarchy

from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel


router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

@router.get("/hierarchy")
def dashboard_hierarchy():
    """
    Provides hierarchy for Dashboard selectors
    """
    return get_dashboard_hierarchy()


# -----------------------------
# Enums
# -----------------------------

class TrendDirection(str, Enum):
    up = "up"
    down = "down"


class SeverityLevel(str, Enum):
    High = "High"
    Medium = "Medium"
    Low = "Low"


class IncidentStatus(str, Enum):
    Open = "Open"
    Pending = "Pending"
    Closed = "Closed"


# -----------------------------
# Core Metric Models
# -----------------------------

class OpenClosedMetrics(BaseModel):
    open: int
    closed: int
    forciblyClosed: int


class SeverityBreakdown(BaseModel):
    high: int
    medium: int
    low: int


class DomainBreakdown(BaseModel):
    clinical: int
    management: int
    relational: int


class DashboardMetrics(BaseModel):
    totalIncidents: int
    uniquePatients: int
    openClosed: OpenClosedMetrics
    severityBreakdown: SeverityBreakdown
    domainBreakdown: DomainBreakdown
    redFlags: int


# -----------------------------
# Trend Models
# -----------------------------

class TrendMetric(BaseModel):
    value: int
    direction: TrendDirection


class DashboardTrends(BaseModel):
    incidentsPatients: TrendMetric
    openClosed: TrendMetric
    severity: TrendMetric
    domain: TrendMetric
    redFlags: TrendMetric


# -----------------------------
# Chart Models
# -----------------------------

class TopClassificationItem(BaseModel):
    classification: str
    count: int


class StageHistogramItem(BaseModel):
    stage: str
    count: int


class IssuingDepartmentItem(BaseModel):
    department: str
    count: int


class DashboardCharts(BaseModel):
    top5Classification: List[TopClassificationItem]
    stageHistogram: List[StageHistogramItem]
    issuingDept: Optional[List[IssuingDepartmentItem]] = None


# -----------------------------
# Recent Activity Models
# -----------------------------

class RecentActivityItem(BaseModel):
    timestamp: datetime
    description: str
    severity: SeverityLevel
    status: IncidentStatus


# -----------------------------
# Dashboard Statistics Response
# -----------------------------

class DashboardStatsResponse(BaseModel):
    metrics: DashboardMetrics
    trends: DashboardTrends
    charts: DashboardCharts
    recentActivity: List[RecentActivityItem]


# -----------------------------
# Organizational Hierarchy Models
# -----------------------------

class OrganizationUnit(BaseModel):
    id: str
    nameAr: str
    nameEn: str


class DashboardHierarchyResponse(BaseModel):
    idarat: List[OrganizationUnit]
    dayrat: Dict[str, List[OrganizationUnit]]
    aqsam: Dict[str, List[OrganizationUnit]]


# -----------------------------
# API Endpoints
# -----------------------------

@router.get(
    "/stats",
    response_model=DashboardStatsResponse
)
def get_dashboard_stats(
    scope: str = Query(..., description="Scope level: hospital, administration, department, section"),
    administration_id: Optional[str] = Query(None),
    department_id: Optional[str] = Query(None),
    section_id: Optional[str] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
):
    raise NotImplementedError


@router.get(
    "/hierarchy",
    response_model=DashboardHierarchyResponse
)
def get_dashboard_hierarchy():
    raise NotImplementedError
