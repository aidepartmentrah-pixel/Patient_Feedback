"""
Insight Router (Phase 4B - B-I12)

Read-only analytics endpoints for API v2.
Provides KPI summary, distribution, trend, and stuck case insights.

All endpoints are read-only and scope-filtered via allowed_unit_ids.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
from backend.api.dependencies.user_context import get_current_user
from backend.api.schemas.auth_models import CurrentUser
from backend.api_v2.services import insight_service


router = APIRouter(
    prefix="/api/v2/insight",
    tags=["api_v2_insight"]
)


# Request Models
class DistributionRequest(BaseModel):
    dimension: str


class TrendRequest(BaseModel):
    bucket: str


@router.get("/kpi-summary")
async def get_kpi_summary_endpoint(
    current_user: CurrentUser = Depends(get_current_user)
):
    """Returns aggregated KPI metrics for scoped subcases and action items."""
    result = insight_service.get_kpi_summary(current_user)
    return result


@router.post("/distribution")
async def get_distribution_endpoint(
    req: DistributionRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Returns grouped distribution counts by requested dimension."""
    result = insight_service.get_distribution(
        current_user,
        req.dimension
    )
    return result


@router.post("/trend")
async def get_trend_endpoint(
    req: TrendRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Returns time-bucketed subcase creation counts."""
    result = insight_service.get_trend(
        current_user,
        req.bucket
    )
    return result


@router.get("/stuck")
async def get_stuck_endpoint(
    days_threshold: int,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Returns subcases whose UpdatedAt exceeds threshold and are not terminal."""
    result = insight_service.get_stuck_cases(
        current_user,
        days_threshold
    )
    return result
