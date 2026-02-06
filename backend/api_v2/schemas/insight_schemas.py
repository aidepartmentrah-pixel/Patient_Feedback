"""
Insight API Response Schemas (Phase 4B)

Pydantic models for API v2 insight endpoint responses.
Pure shape definitions - no validators, no computed fields.
"""

from pydantic import BaseModel
from datetime import datetime


class KPIStatusCount(BaseModel):
    """Status distribution item for KPI summary."""
    status: str
    count: int


class KPIActionItemSummary(BaseModel):
    """Action item statistics for KPI summary."""
    total: int
    open: int
    completed: int
    overdue: int


class KPISummaryResponse(BaseModel):
    """Response model for GET /api/v2/insight/kpi-summary."""
    total_subcases: int
    by_status: list[KPIStatusCount]
    action_items: KPIActionItemSummary


class DistributionItem(BaseModel):
    """Distribution item for dimension-based grouping."""
    key: str | int
    count: int


class TrendItem(BaseModel):
    """Trend item for time-bucketed counts."""
    bucket: str
    count: int


class StuckItem(BaseModel):
    """Stuck case item with stagnation details."""
    subcase_id: int
    status: str
    target_org_unit_id: int
    updated_at: datetime
    days_in_stage: int
