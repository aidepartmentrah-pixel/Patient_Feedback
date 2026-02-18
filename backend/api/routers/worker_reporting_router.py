"""
Worker Reporting Router
API endpoints for worker profile and reporting.
"""

from fastapi import APIRouter, Path, Query, HTTPException, status, Depends
from typing import Optional
from datetime import date

from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..schemas.worker_reporting_schema import WorkerProfileResponse
from ..services.worker_reporting_service import WorkerReportingService


router = APIRouter(prefix="/api/workers", tags=["Worker Reporting"])


# ==================== WORKER PROFILE ENDPOINT ====================

@router.get(
    "/{employee_id}/profile",
    response_model=WorkerProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Get worker profile with performance metrics",
    responses={
        200: {
            "description": "Worker profile retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "worker": {
                            "employee_id": 12345,
                            "full_name": "Ahmed Mohammed Al-Shahrani",
                            "job_title": "Quality Assurance Specialist",
                            "department_id": 42,
                            "section_id": 8,
                            "administration_id": 3,
                            "is_active": True
                        },
                        "metrics": {
                            "total_incidents": 12,
                            "total_action_items": 45,
                            "completed_action_items": 38,
                            "overdue_action_items": 3,
                            "explanation_rejected_count": 2,
                            "explanation_accepted_count": 15
                        },
                        "period_from": "2025-01-01",
                        "period_to": "2025-12-31"
                    }
                }
            }
        },
        404: {
            "description": "Worker not found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Worker not found"
                    }
                }
            }
        },
        401: {
            "description": "Not authenticated",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Not authenticated"
                    }
                }
            }
        }
    }
)
def get_worker_profile(
    employee_id: int = Path(..., gt=0, description="Employee ID from HR system"),
    date_from: Optional[date] = Query(
        None,
        description="Start date for metrics aggregation (optional, ISO format: YYYY-MM-DD)",
        example="2025-01-01"
    ),
    date_to: Optional[date] = Query(
        None,
        description="End date for metrics aggregation (optional, ISO format: YYYY-MM-DD)",
        example="2025-12-31"
    ),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Retrieve worker profile with aggregated performance metrics.
    
    This endpoint returns a complete worker profile including:
    - Identity information from HR employee system
    - Performance metrics aggregated from incidents, action items, and explanations
    - Optional date range filtering for metrics
    
    **Path Parameters:**
    - `employee_id`: Unique employee identifier from HR system (required, must be > 0)
    
    **Query Parameters:**
    - `date_from`: Optional start date for filtering metrics (ISO format: YYYY-MM-DD)
    - `date_to`: Optional end date for filtering metrics (ISO format: YYYY-MM-DD)
    
    **Authentication:**
    - Requires authenticated user session (uses session-based authentication)
    - No specific role or permission required (authentication only)
    
    **Response:**
    Returns `WorkerProfileResponse` with:
    - `worker`: Identity block with employee info and org unit assignments
    - `metrics`: Performance metrics including incidents, actions, explanations
    - `period_from`, `period_to`: Date range used for metrics (null = all-time)
    
    **Metrics Included:**
    - Total incidents involving worker
    - Total action items assigned to worker
    - Completed vs overdue action items
    - Explanation acceptance vs rejection counts
    
    **Examples:**
    
    Get all-time metrics for employee:
    ```
    GET /api/workers/12345/profile
    ```
    
    Get metrics for specific date range:
    ```
    GET /api/workers/12345/profile?date_from=2025-01-01&date_to=2025-12-31
    ```
    
    **Error Cases:**
    - 401: User not authenticated (no active session)
    - 404: Worker/employee not found in HR system
    - 500: Database or internal server error
    """
    try:
        # Call service layer to aggregate worker profile
        profile = WorkerReportingService.get_worker_profile(
            employee_id=employee_id,
            date_from=date_from,
            date_to=date_to
        )
        
        return profile
        
    except ValueError as ve:
        # Worker not found in HR system
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Worker not found"
        )
    except Exception as e:
        # Database or other internal errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve worker profile: {str(e)}"
        )
