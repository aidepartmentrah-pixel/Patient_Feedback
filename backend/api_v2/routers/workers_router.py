"""
Worker Router (API v2)
Phase B — B-B3 — Worker search endpoint V2 wrapper.
Phase B — B-B4 — Worker action items endpoint V2.
Phase B — B-B5 — Worker profile endpoint V2.

This router provides worker-related endpoints under /api/v2/workers.

Endpoints exposed:
- GET /api/v2/workers/search - Search for workers/employees
- GET /api/v2/workers/{employee_id}/actions - Get worker action items
- GET /api/v2/workers/{employee_id}/profile - Get worker profile

Security: All endpoints protected by authentication.
"""

from fastapi import APIRouter, Query, Path, Depends, HTTPException
from fastapi.responses import StreamingResponse, Response
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime, date
import traceback
import io
import logging

from backend.api.dependencies.user_context import get_current_user
from backend.api.schemas.auth_models import CurrentUser
from backend.api.services.search_service import search_employees
from core import hospital_directory_client as directory_client
from backend.api_v2.db_layer import action_item_subcase_db
from backend.api.services.worker_reporting_service import (
    WorkerReportingService,
    get_worker_full_history_service,
    export_worker_history_service
)
from backend.api_v2.schemas.profile_schemas import WorkerProfileV2Response, EntityMeta

logger = logging.getLogger(__name__)


# ============================================================
# ROUTER DEFINITION (Phase B — B-B3)
# ============================================================
router = APIRouter(prefix="/api/v2/workers", tags=["Workers V2"])


# ==================== RESPONSE SCHEMAS ====================
# Stable, frontend-friendly response models

class WorkerSearchItem(BaseModel):
    """Individual worker item in search results."""
    # Union[int, str]: a plain reserve EmployeeID (int) or an opaque external
    # id like "ext__E-5002" (str) for a worker sourced from the Hospital
    # Directory API, not yet materialized into a local reserve row -- see
    # staff_directory_service.search_workers_merged(). This used to be a
    # hard int because nothing upstream ever produced anything else; now
    # that the merged search legitimately returns external string ids, this
    # must accept both instead of rejecting the valid value.
    employee_id: Union[int, str] = Field(..., description="Employee unique identifier (int for reserve, opaque string for external/unmaterialized)")
    id: Union[int, str] = Field(..., description="Employee ID (alias)")
    full_name: str = Field(..., description="Employee full name")
    name: str = Field(..., description="Employee name (alias)")
    job_title: Optional[str] = Field(None, description="Job title/position")
    # str, not int: APP_RESERVE_WORKER.DepartmentID/SectionID/AdministrationID
    # are nvarchar org codes (same shape from the Hospital Directory API),
    # not surrogate ints -- Optional[int] here rejected every real worker
    # row with a non-null value and turned it into a 500.
    department_id: Optional[str] = Field(None, description="Department ID")
    section_id: Optional[str] = Field(None, description="Section ID")
    administration_id: Optional[str] = Field(None, description="Administration ID")
    is_manager: bool = Field(False, description="Whether employee is a manager")
    is_active: bool = Field(True, description="Whether employee is active")

    class Config:
        json_schema_extra = {
            "example": {
                "employee_id": 12345,
                "id": 12345,
                "full_name": "Ahmed Mohammed Al-Shahrani",
                "name": "Ahmed Mohammed Al-Shahrani",
                "job_title": "Quality Assurance Specialist",
                "department_id": 42,
                "section_id": 8,
                "administration_id": 3,
                "is_manager": False,
                "is_active": True
            }
        }


class WorkerSearchResponse(BaseModel):
    """Response model for worker search endpoint."""
    success: bool = Field(True, description="Request success status")
    items: List[WorkerSearchItem] = Field(..., description="List of matching workers")
    total: int = Field(..., description="Total number of results")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "items": [
                    {
                        "employee_id": 12345,
                        "id": 12345,
                        "full_name": "Ahmed Mohammed Al-Shahrani",
                        "name": "Ahmed Mohammed Al-Shahrani",
                        "job_title": "Quality Assurance Specialist",
                        "department_id": 42,
                        "section_id": 8,
                        "administration_id": 3,
                        "is_manager": False,
                        "is_active": True
                    }
                ],
                "total": 1
            }
        }


# ==================== ENDPOINTS ====================

@router.get(
    "/search",
    response_model=WorkerSearchResponse,
    summary="Search for workers/employees",
    responses={
        200: {
            "description": "Search results returned successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "items": [
                            {
                                "employee_id": 12345,
                                "id": 12345,
                                "full_name": "Ahmed Mohammed Al-Shahrani",
                                "name": "Ahmed Mohammed Al-Shahrani",
                                "job_title": "Quality Assurance Specialist",
                                "department_id": 42,
                                "section_id": 8,
                                "administration_id": 3,
                                "is_manager": False,
                                "is_active": True
                            }
                        ],
                        "total": 1
                    }
                }
            }
        },
        401: {"description": "Not authenticated"},
        500: {"description": "Search failed"}
    }
)
async def search_workers(
    q: str = Query(..., min_length=2, description="Search query (employee name or ID, minimum 2 characters)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results (1-100)"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Search for active workers/employees by name or ID.
    
    This endpoint wraps the existing employee search logic and provides
    a stable, normalized response format for V2 API consumers.
    
    **Query Parameters:**
    - `q`: Search text (required, min 2 characters) - searches in employee full name and ID
    - `limit`: Maximum results to return (default: 20, max: 100)
    
    **Search Behavior:**
    - Searches for partial matches in employee full name (Arabic and English) and employee ID
    - Returns only active employees (is_active = True)
    - Excludes doctors (only returns workers/staff)
    - Results ordered alphabetically by full name
    
    **Response Fields:**
    - `success`: Always true for successful requests
    - `items`: Array of matching workers
    - `total`: Number of workers returned
    
    **Example Requests:**
    ```
    GET /api/v2/workers/search?q=Ahmed&limit=10
    GET /api/v2/workers/search?q=محمد&limit=5
    GET /api/v2/workers/search?q=E456&limit=20
    ```
    
    **Example Response:**
    ```json
    {
      "success": true,
      "items": [
        {
          "employee_id": 12345,
          "id": 12345,
          "full_name": "Ahmed Mohammed Al-Shahrani",
          "name": "Ahmed Mohammed Al-Shahrani",
          "job_title": "Quality Assurance Specialist",
          "department_id": 42,
          "section_id": 8,
          "administration_id": 3,
          "is_manager": false,
          "is_active": true
        }
      ],
      "total": 1
    }
    ```
    
    **Security:**
    Requires authentication. User must be logged in to search for workers.
    """
    try:
        # Call existing search service (reusing logic, no duplication)
        result = search_employees(q, limit)
        
        # Check for service-level errors
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "SEARCH_FAILED",
                    "message": result.get("error", "Failed to search workers"),
                    "message_ar": "فشل في البحث عن الموظفين"
                }
            )
        
        # Normalize response format: "employees" -> "items"
        # This provides a stable V2 API contract
        employees = result.get("employees", [])
        
        # Map to WorkerSearchItem schema (validates structure)
        items = []
        for emp in employees:
            employee_id = emp.get("employee_id")
            full_name = emp.get("full_name", "")
            
            items.append(WorkerSearchItem(
                employee_id=employee_id,
                id=employee_id,  # Add id field as alias
                full_name=full_name,
                name=full_name,  # Add name field as alias
                job_title=emp.get("job_title"),
                department_id=emp.get("department_id"),
                section_id=emp.get("section_id"),
                administration_id=emp.get("administration_id"),
                is_manager=emp.get("is_manager", False),
                is_active=emp.get("is_active", True)
            ))
        
        # Return normalized response with success and total fields
        return WorkerSearchResponse(
            success=True,
            items=items,
            total=len(items)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SEARCH_ERROR",
                "message": f"Search operation failed: {str(e)}",
                "message_ar": "فشلت عملية البحث"
            }
        )


# ==================== ACTION ITEM SCHEMAS (Phase B — B-B4) ====================

class WorkerActionItem(BaseModel):
    """Individual action item in worker action list."""
    action_item_id: int = Field(..., alias="action_id", description="Action item unique identifier")
    title: str = Field(..., description="Action item title")
    status: str = Field(..., description="Action item status")
    created_at: datetime = Field(..., description="Creation timestamp")
    due_date: Optional[date] = Field(None, description="Due date")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    incident_case_id: Optional[int] = Field(None, description="Associated incident case ID")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "action_id": 123,
                "title": "Review incident report",
                "status": "IN_PROGRESS",
                "created_at": "2026-02-01T10:00:00",
                "due_date": "2026-02-15",
                "completed_at": None,
                "incident_case_id": 456
            }
        }


class WorkerActionListResponse(BaseModel):
    """Response model for worker action list endpoint."""
    items: List[WorkerActionItem] = Field(..., description="List of action items")
    count: int = Field(..., description="Total count of matching action items")
    limit: int = Field(..., description="Applied limit")
    offset: int = Field(..., description="Applied offset")
    
    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {
                        "action_id": 123,
                        "title": "Review incident report",
                        "status": "IN_PROGRESS",
                        "created_at": "2026-02-01T10:00:00",
                        "due_date": "2026-02-15",
                        "completed_at": None,
                        "incident_case_id": 456
                    }
                ],
                "count": 1,
                "limit": 50,
                "offset": 0
            }
        }


# ==================== ACTION ITEM ENDPOINT (Phase B — B-B4) ====================

@router.get(
    "/{employee_id}/actions",
    response_model=WorkerActionListResponse,
    summary="Get action items for a worker",
    responses={
        200: {"description": "Action items retrieved successfully"},
        401: {"description": "Not authenticated"},
        404: {"description": "Worker not found"},
        500: {"description": "Failed to retrieve action items"}
    }
)
async def get_worker_actions(
    employee_id: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of items (1-200)"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    status: Optional[str] = Query(None, description="Filter by status (e.g., 'IN_PROGRESS', 'COMPLETED')"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get action items assigned to a specific worker (employee).
    
    Phase B — B-B4 — Worker action list endpoint V2.
    
    **Path Parameters:**
    - `employee_id`: Employee unique identifier
    
    **Query Parameters:**
    - `limit`: Maximum results (default: 50, max: 200)
    - `offset`: Pagination offset (default: 0)
    - `status`: Optional status filter (e.g., 'DRAFT', 'IN_PROGRESS', 'COMPLETED')
    
    **Response Fields:**
    - `items`: Array of action items
    - `count`: Total number of matching action items
    - `limit`: Applied limit
    - `offset`: Applied offset
    
    **Example Requests:**
    ```
    GET /api/v2/workers/12345/actions
    GET /api/v2/workers/12345/actions?limit=10&offset=0
    GET /api/v2/workers/12345/actions?status=IN_PROGRESS
    ```
    
    **Example Response:**
    ```json
    {
      "items": [
        {
          "action_id": 123,
          "title": "Review incident report",
          "status": "IN_PROGRESS",
          "created_at": "2026-02-01T10:00:00",
          "due_date": "2026-02-15",
          "completed_at": null,
          "incident_case_id": 456
        }
      ],
      "count": 1,
      "limit": 50,
      "offset": 0
    }
    ```
    
    **Security:**
    Requires authentication. Scope guards enforced by reusing Phase D logic.
    """
    try:
        # An external (never-materialized) worker has no local action items --
        # return a valid empty list instead of querying with a nonexistent int id.
        external_id = directory_client.decode_external_id(employee_id)
        if external_id:
            return WorkerActionListResponse(items=[], count=0, limit=limit, offset=offset)

        # Phase B — Worker action list endpoint V2
        # Reuse Phase D aggregation DB functions
        result = action_item_subcase_db.get_worker_action_items(
            employee_id=int(employee_id),
            limit=limit,
            offset=offset,
            status=status
        )
        
        # Map database results to Pydantic schema
        items = []
        for item in result["items"]:
            items.append(WorkerActionItem(
                action_id=item["action_item_id"],
                title=item["title"],
                status=item["status"],
                created_at=item["created_at"],
                due_date=item["due_date"],
                completed_at=item["completed_at"],
                incident_case_id=item["incident_case_id"]
            ))
        
        return WorkerActionListResponse(
            items=items,
            count=result["total_count"],
            limit=result["limit"],
            offset=result["offset"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "FETCH_FAILED",
                "message": f"Failed to retrieve worker action items: {str(e)}",
                "message_ar": "فشل في استرجاع عناصر العمل للموظف"
            }
        )


# ==================== WORKER PROFILE ENDPOINT (Phase B — B-B5) ====================

@router.get(
    "/{employee_id}/profile",
    response_model=WorkerProfileV2Response,
    summary="Get worker profile with metrics",
    responses={
        200: {"description": "Worker profile retrieved successfully"},
        401: {"description": "Not authenticated"},
        404: {"description": "Worker not found"},
        500: {"description": "Failed to retrieve worker profile"}
    }
)
async def get_worker_profile(
    employee_id: str = Path(..., min_length=1, description="Employee unique identifier (int for reserve, or external id like ext__E-5002)"),
    date_from: Optional[date] = Query(
        None,
        description="Start date for metrics filtering (YYYY-MM-DD)"
    ),
    date_to: Optional[date] = Query(
        None,
        description="End date for metrics filtering (YYYY-MM-DD)"
    ),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get worker profile with aggregated performance metrics.
    
    Phase B — V2 profile contract normalized.
    
    **Path Parameters:**
    - `employee_id`: Employee unique identifier (required)
    
    **Query Parameters:**
    - `date_from`: Optional start date for metrics filtering
    - `date_to`: Optional end date for metrics filtering
    
    **Response:**
    Returns standardized V2 profile response with:
    - `profile`: Worker identity and organizational assignment
    - `metrics`: Aggregated performance metrics
    - `items`: Empty array (use /actions endpoint for action items)
    - `meta`: Entity metadata including entity_type, entity_id, period
    
    **Example Requests:**
    ```
    GET /api/v2/workers/12345/profile
    GET /api/v2/workers/12345/profile?date_from=2025-01-01&date_to=2025-12-31
    ```
    
    **Security:**
    Requires authentication. User must be logged in.
    """
    try:
        # Phase B — V2 profile contract normalized
        # Reuse existing worker reporting service
        result = WorkerReportingService.get_worker_profile(
            employee_id=employee_id,
            date_from=date_from,
            date_to=date_to
        )
        
        # Normalize to V2 contract: profile, metrics, items, meta
        # Map from WorkerProfileResponse to standardized structure
        profile_data = {
            "employee_id": result.worker.employee_id,
            "full_name": result.worker.full_name,
            "job_title": result.worker.job_title,
            "department_id": result.worker.department_id,
            "section_id": result.worker.section_id,
            "administration_id": result.worker.administration_id,
            "is_active": result.worker.is_active
        }
        
        metrics_data = {
            "total_incidents": result.metrics.total_incidents,
            "total_action_items": result.metrics.total_action_items,
            "completed_action_items": result.metrics.completed_action_items,
            "overdue_action_items": result.metrics.overdue_action_items,
            "explanation_rejected_count": result.metrics.explanation_rejected_count,
            "explanation_accepted_count": result.metrics.explanation_accepted_count
        }
        
        return WorkerProfileV2Response(
            profile=profile_data,
            metrics=metrics_data,
            items=[],  # Use /actions endpoint for action items
            meta=EntityMeta(
                entity_type="worker",
                entity_id=employee_id,
                period_from=result.period_from,
                period_to=result.period_to
            )
        )
        
    except ValueError as ve:
        # Worker not found
        raise HTTPException(
            status_code=404,
            detail={
                "error": "WORKER_NOT_FOUND",
                "message": str(ve),
                "message_ar": "الموظف غير موجود"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PROFILE_FETCH_FAILED",
                "message": f"Failed to retrieve worker profile: {str(e)}",
                "message_ar": "فشل في استرجاع ملف الموظف"
            }
        )


# ==================== FULL HISTORY (mirrors doctor/patient full-history) ====================

@router.get(
    "/{employee_id}/full-history",
    summary="Get worker full history",
    description="Get worker profile with incident history in a unified response. Mirrors /api/v2/doctors/{id}/full-history.",
    response_description="Full worker history with profile, metrics, and incidents"
)
async def get_worker_full_history(
    employee_id: str = Path(..., min_length=1, description="Employee ID (int for reserve, or external id like ext__E-5002)"),
    date_from: Optional[date] = Query(None, description="Start date filter"),
    date_to: Optional[date] = Query(None, description="End date filter"),
    limit: int = Query(100, ge=1, le=500, description="Max incidents to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Return worker profile + metrics + incidents in V2 unified schema."""
    try:
        result = get_worker_full_history_service(
            employee_id=employee_id,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset
        )
        return result

    except ValueError as ve:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "WORKER_NOT_FOUND",
                "message": str(ve),
                "message_ar": "الموظف غير موجود"
            }
        )
    except Exception as e:
        logger.error(f"Worker full-history error for {employee_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "FULL_HISTORY_FAILED",
                "message": f"Failed to retrieve worker full history: {str(e)}",
                "message_ar": "فشل في استرجاع السجل الكامل للموظف"
            }
        )


# ==================== EXPORT (mirrors doctor/patient export) ====================

@router.get(
    "/{employee_id}/export",
    summary="Export worker history",
    description="Export worker incident history as CSV, JSON, or Word document. Mirrors /api/v2/doctors/{id}/export.",
    response_description="Exported worker history file"
)
async def export_worker_history(
    employee_id: str = Path(..., min_length=1, description="Employee ID (int for reserve, or external id like ext__E-5002)"),
    format: str = Query("json", pattern="^(csv|json|word)$", description="Export format: csv, json, or word"),
    date_from: Optional[date] = Query(None, description="Start date filter"),
    date_to: Optional[date] = Query(None, description="End date filter"),
    include_profile: bool = Query(True, description="Include worker profile in export"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """Export worker history in specified format."""
    try:
        result = export_worker_history_service(
            employee_id=employee_id,
            format_type=format,
            date_from=date_from,
            date_to=date_to,
            include_profile=include_profile
        )

        if format == "csv":
            return StreamingResponse(
                io.StringIO(result["content"]),
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{result["filename"]}"'}
            )
        elif format == "word":
            return Response(
                content=result["content"],
                media_type=result["content_type"],
                headers={"Content-Disposition": f'attachment; filename="{result["filename"]}"'}
            )
        else:
            return result

    except ValueError as ve:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "WORKER_NOT_FOUND",
                "message": str(ve),
                "message_ar": "الموظف غير موجود"
            }
        )
    except Exception as e:
        logger.error(f"Worker export error for {employee_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "EXPORT_FAILED",
                "message": f"Failed to export worker history: {str(e)}",
                "message_ar": "فشل في تصدير سجل الموظف"
            }
        )

