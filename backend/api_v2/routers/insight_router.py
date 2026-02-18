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


@router.get("/user-workload")
async def get_user_workload_endpoint(
    org_unit_id: int = None,
    role: str = None,
    min_items: int = 1,
    sort_by: str = 'pending_count',
    sort_order: str = 'desc',
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Returns user workload statistics based on assigned action items.
    
    Shows person-centric view of pending workload distribution.
    Only authorized for SOFTWARE_ADMIN, WORKER, and COMPLAINT_SUPERVISOR roles.
    
    Query Parameters:
    - org_unit_id: Filter by organizational unit ID
    - role: Filter by user role (e.g., SECTION_ADMIN, DEPARTMENT_ADMIN)
    - min_items: Only show users with >= N pending items (default: 1)
    - sort_by: Sort by 'pending_count', 'oldest_item', or 'user_name' (default: 'pending_count')
    - sort_order: Sort order 'asc' or 'desc' (default: 'desc')
    
    Authorization:
    - SOFTWARE_ADMIN: Full access
    - WORKER: Full access
    - COMPLAINT_SUPERVISOR: Full access (if they have Insight page access)
    
    Returns:
    - List of users with their pending item counts and oldest item age
    """
    # Authorization check: Only specific roles can access user workload
    allowed_roles = {"SOFTWARE_ADMIN", "WORKER", "COMPLAINT_SUPERVISOR"}
    user_roles = set(current_user.roles) if hasattr(current_user, 'roles') else set()
    
    if not user_roles.intersection(allowed_roles):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions. Only admins and workers can view user workload."
        )
    
    # Validate sort_by parameter
    valid_sort_by = {"pending_count", "oldest_item", "user_name"}
    if sort_by not in valid_sort_by:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sort_by value. Must be one of: {', '.join(valid_sort_by)}"
        )
    
    # Validate sort_order parameter
    if sort_order not in ['asc', 'desc']:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="Invalid sort_order value. Must be 'asc' or 'desc'"
        )
    
    result = insight_service.get_user_workload(
        current_user=current_user,
        org_unit_id=org_unit_id,
        role=role,
        min_items=min_items,
        sort_by=sort_by,
        sort_order=sort_order
    )
    return result


@router.get("/grouped-inbox")
async def get_grouped_inbox_endpoint(
    current_user: CurrentUser = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Get subcases grouped by target organizational unit.
    
    Shows supervisor names, pending counts, and case details for each section/dept.
    
    **Purpose:**
    - Provides supervisory/analytical view of workload distribution
    - Shows which sections/departments have pending work
    - Displays supervisor assignments and case details
    
    **Security:**
    - Requires authentication
    - Applies Phase 2.5 scope filtering (only shows accessible org units)
    - Intended for admin/supervisory roles
    
    **Sorting:**
    - Groups sorted by pending_count DESC (busiest sections first)
    - Subcases within each group sorted by waiting_days DESC (oldest first)
    
    **Returns:**
    ```json
    [
      {
        "section_id": 10,
        "section_name": "Emergency Department",
        "org_type": "SECTION",
        "supervisor_name": "Dr. Ahmed",
        "pending_count": 5,
        "subcases": [
          {
            "subcase_id": 123,
            "case_type": "INCIDENT_RESPONSE",
            "status": "SUBMITTED_TO_SECTION",
            "case_description": "Patient fell in corridor...",
            "patient_name": "Mohammed Ali",
            "severity": "HIGH",
            "category": "Falls",
            "waiting_days": 14,
            "created_at": "2026-01-29T10:30:00",
            "target_org_unit_id": 10,
            "org_unit_name": "Emergency Department"
          },
          ...
        ]
      },
      ...
    ]
    ```
    
    **Example Usage:**
    ```
    GET /api/v2/insight/grouped-inbox
    Authorization: Bearer <token>
    ```
    
    **Response Fields:**
    - `section_id`: Target organizational unit ID
    - `section_name`: Name of the section/department
    - `org_type`: Type code of the organizational unit
    - `supervisor_name`: Admin/supervisor assigned to this unit ("Unassigned" if none)
    - `pending_count`: Number of pending subcases for this unit
    - `subcases`: Array of subcases with full details including description, severity, waiting time
    """
    return insight_service.get_grouped_inbox_for_admin(current_user)
