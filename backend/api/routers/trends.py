"""
AUTO-GENERATED API CONTRACT

Source: Obsidian – TrendMonitoringPage
Iteration: 1
Status: API skeleton only – no implementation
"""

from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/trends", tags=["TrendMonitoring"])


# ------------------------------------------------------------------
# Shared Time Range Models
# ------------------------------------------------------------------

class TimeRangeInfo(BaseModel):
    start: str  # YYYY-MM
    end: str    # YYYY-MM
    total_months: int
    start_label: str
    start_label_ar: str
    end_label: str
    end_label_ar: str


class TrendSummary(BaseModel):
    total_incidents: int
    total_domains: Optional[int] = None
    total_categories: Optional[int] = None
    average_monthly_incidents: float
    generated_at: datetime


# ------------------------------------------------------------------
# Domain-Level Trend Models
# ------------------------------------------------------------------

class DomainMonthlyData(BaseModel):
    period: str  # YYYY-MM
    month: int
    year: int
    period_label: str
    period_label_ar: str
    incident_count: int
    percentage_of_total: Optional[float]
    month_over_month_change: Optional[int]
    month_over_month_percentage: Optional[float]


class DomainTrend(BaseModel):
    domain_id: int
    domain_name: str
    domain_name_ar: str
    domain_code: str
    domain_color: str

    total_incidents: int
    average_monthly_incidents: float
    peak_month: Optional[str]
    peak_count: Optional[int]

    trend_direction: Optional[str]
    trend_percentage: Optional[float]

    monthly_data: List[DomainMonthlyData]


class DomainTrendsResponse(BaseModel):
    time_range: TimeRangeInfo
    summary: TrendSummary
    domains: List[DomainTrend]


# ------------------------------------------------------------------
# Category-Level Trend Models
# ------------------------------------------------------------------

class CategoryMonthlyData(BaseModel):
    period: str  # YYYY-MM
    month: int
    year: int
    period_label: str
    period_label_ar: str
    incident_count: int
    percentage_of_domain: Optional[float]
    percentage_of_total: Optional[float]
    month_over_month_change: Optional[int]
    month_over_month_percentage: Optional[float]


class CategoryTrend(BaseModel):
    category_id: int
    category_name: str
    category_name_ar: str

    domain_id: int
    domain_name: str
    domain_name_ar: str

    category_color: str

    total_incidents: int
    average_monthly_incidents: float
    peak_month: Optional[str]
    peak_count: Optional[int]

    trend_direction: Optional[str]
    trend_percentage: Optional[float]

    monthly_data: List[CategoryMonthlyData]


class CategoryTrendsResponse(BaseModel):
    time_range: TimeRangeInfo
    summary: TrendSummary
    categories: List[CategoryTrend]


# ------------------------------------------------------------------
# Period Listing Models
# ------------------------------------------------------------------

class PeriodInfo(BaseModel):
    period: str  # YYYY-MM
    month: int
    year: int
    period_label: str
    period_label_ar: str
    period_start_date: date
    period_end_date: date
    total_incidents: int
    has_data: bool


class PeriodsResponse(BaseModel):
    periods: List[PeriodInfo]
    total_periods: int
    earliest_period: str
    latest_period: str


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.get("/domains", response_model=DomainTrendsResponse)
def fetch_domain_trends(
    start_date: Optional[str] = Query(None, description="YYYY-MM"),
    end_date: Optional[str] = Query(None, description="YYYY-MM"),
    include_zero_months: bool = True,
    include_inactive_domains: bool = False,
    calculate_trends: bool = True,
):
    raise NotImplementedError


@router.get("/categories", response_model=CategoryTrendsResponse)
def fetch_category_trends(
    start_date: Optional[str] = Query(None, description="YYYY-MM"),
    end_date: Optional[str] = Query(None, description="YYYY-MM"),
    domain_id: Optional[int] = None,
    include_zero_months: bool = True,
    include_inactive_categories: bool = False,
    calculate_trends: bool = True,
    top_n: Optional[int] = None,
):
    raise NotImplementedError


@router.get("/periods", response_model=PeriodsResponse)
def fetch_available_periods(
    start_date: Optional[str] = Query(None, description="YYYY-MM"),
    end_date: Optional[str] = Query(None, description="YYYY-MM"),
):
    raise NotImplementedError
