"""
Red Flags Router
API endpoints for Red Flags (Critical Issues) page.
Red flags are high-risk incidents requiring immediate attention and governance follow-up.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from ..services.red_flags_service import (
    get_red_flags_list,
    get_red_flags_statistics,
    get_red_flags_trends,
    get_red_flag_details,
    get_red_flags_category_breakdown,
    get_red_flags_department_breakdown
)


router = APIRouter(prefix="/api/red-flags", tags=["Red Flags"])


# ==================== ENDPOINTS ====================

@router.get("")
async def get_red_flags(
    search: Optional[str] = Query(None, description="Search by record ID or patient name"),
    status: Optional[str] = Query(None, description="Filter by status: OPEN, UNDER_REVIEW, FINISHED, all"),
    from_date: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)"),
    department: Optional[str] = Query(None, description="Filter by department"),
    category: Optional[str] = Query(None, description="Filter by red flag category"),
    severity: Optional[str] = Query(None, description="Filter by severity: HIGH, CRITICAL"),
    is_never_event: Optional[bool] = Query(None, description="Filter red flags that are also Never Events"),
    sort_by: str = Query("date", description="Sort by: date, severity, department, status, created_at, patient_name"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    limit: int = Query(100, ge=1, le=500, description="Max results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """
    Fetch list of red flag incidents with optional filtering and search.
    
    **Red Flags** are high-risk incidents (ClinicalRiskTypeID = 2) requiring immediate attention.
    
    **Example Request:**
    ```
    GET /api/red-flags?status=FINISHED&from_date=2024-01-01&sort_by=severity&sort_order=desc&limit=50
    ```
    
    **Query Parameters:**
    - `search`: Search across record ID and patient name
    - `status`: Filter by OPEN, UNDER_REVIEW, FINISHED, or "all"
    - `from_date`, `to_date`: Date range filters
    - `department`: Filter by department name
    - `category`: Filter by red flag category (Domain)
    - `severity`: Filter by HIGH or CRITICAL
    - `is_never_event`: Show only red flags that are also Never Events
    - `sort_by`: Sort by date (default), severity, department, status, created_at, or patient_name
    - `sort_order`: asc or desc (default: desc for newest first)
    - `limit`: Results per page (default: 100, max: 500)
    - `offset`: Pagination offset (default: 0)
    
    **Returns:**
    - List of red flag records with full patient information, complaint details, and classifications
    - Total count for pagination
    - Sorted by date descending (most recent first) by default
    - Includes fields: patient_full_name, complaint_text, severity, department, classification, etc.
    """
    
    try:
        result = get_red_flags_list(
            search=search,
            status=status,
            from_date=from_date,
            to_date=to_date,
            department=department,
            category=category,
            severity=severity,
            is_never_event=is_never_event,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "QUERY_FAILED",
                "message": f"An error occurred while fetching red flags: {str(e)}",
                "message_ar": f"حدث خطأ أثناء جلب الأعلام الحمراء: {str(e)}"
            }
        )


@router.get("/statistics")
async def get_statistics(
    from_date: Optional[str] = Query(None, description="Statistics from date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="Statistics to date (YYYY-MM-DD)")
):
    """
    Fetch summary statistics for Red Flags KPI cards and Never Event cross-reference.
    
    **Example Request:**
    ```
    GET /api/red-flags/statistics?from_date=2024-01-01&to_date=2024-12-31
    ```
    
    **Query Parameters:**
    - `from_date`: Statistics from this date (optional, default: all time)
    - `to_date`: Statistics to this date (optional, default: today)
    
    **Returns:**
    - `total_red_flags`: Total count of red flags
    - `unfinished`: Count of red flags not yet finished
    - `finished`: Count of completed red flags
    - `by_severity`: Nested object with counts by severity (CRITICAL, HIGH, MEDIUM, LOW)
    - `by_status`: Nested object with counts by status (OPEN, IN_PROGRESS, RESOLVED, CLOSED)
    - `current_month`: Object with count, month name, start_date, and end_date
    - `previous_month`: Object with count and month name
    - `never_event_overlap`: Statistics on overlap between red flags and never events
    - `average_resolution_days`: Average days to resolve red flags
    - `period`: Date range applied (from_date, to_date)
    
    **Never Event Overlap:**
    - `total_never_events`: Total count of Never Events in system
    - `red_flags_also_never_events`: Red flags that are also Never Events
    - `never_events_only`: Never Events that are NOT red flags
    - `red_flags_only`: Red flags that are NOT Never Events
    """
    
    try:
        result = get_red_flags_statistics(
            from_date=from_date,
            to_date=to_date
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "STATISTICS_FAILED",
                "message": f"An error occurred while calculating statistics: {str(e)}",
                "message_ar": f"حدث خطأ أثناء حساب الإحصائيات: {str(e)}"
            }
        )


@router.get("/trends")
async def get_trends(
    from_date: Optional[str] = Query(None, description="Trend from date (default: last 12 months)"),
    to_date: Optional[str] = Query(None, description="Trend to date (default: today)"),
    granularity: str = Query("monthly", description="monthly, quarterly, or weekly"),
    group_by: str = Query("none", description="category, severity, department, or none")
):
    """
    Fetch time-series trend data for Red Flags visualization.
    
    **Example Request:**
    ```
    GET /api/red-flags/trends?granularity=monthly&group_by=category
    ```
    
    **Query Parameters:**
    - `from_date`: Start date for trend (default: 12 months ago)
    - `to_date`: End date for trend (default: today)
    - `granularity`: Time period grouping (monthly, quarterly, weekly)
    - `group_by`: Group by category, severity, department, or none
    
    **Returns:**
    - Time-series data for trend chart
    - Period labels (e.g., "Jan 2024", "Q1 2024")
    - Counts per period
    - Optional breakdown by selected grouping
    
    **Notes:**
    - Periods with zero red flags are included for chart continuity
    - Default shows last 12 months
    - Grouping allows stacked or grouped trend charts
    """
    
    try:
        result = get_red_flags_trends(
            from_date=from_date,
            to_date=to_date,
            granularity=granularity,
            group_by=group_by
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "TRENDS_FAILED",
                "message": f"An error occurred while generating trends: {str(e)}",
                "message_ar": f"حدث خطأ أثناء إنشاء الاتجاهات: {str(e)}"
            }
        )


# ==================== ANALYTICS ENDPOINTS ====================

@router.get("/category-breakdown")
async def get_category_breakdown(
    from_date: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)")
):
    """
    Fetch category breakdown for red flags analytics cards.
    
    **Purpose:** Distribution of red flags across categories with severity breakdown.
    
    **Example Request:**
    ```
    GET /api/red-flags/category-breakdown?from_date=2024-01-01&to_date=2024-12-31
    ```
    
    **Query Parameters:**
    - `from_date`: Start date for filtering (optional)
    - `to_date`: End date for filtering (optional)
    
    **Returns:**
    - Total count of red flags
    - Date range applied
    - Array of categories with:
      - Category name (English and Arabic)
      - Count of red flags in category
      - Percentage of total
      - Severity breakdown (CRITICAL vs HIGH)
    
    **Notes:**
    - Results sorted by count descending
    - Used for dashboard category distribution cards
    - Percentages sum to 100%
    """
    
    try:
        result = get_red_flags_category_breakdown(
            from_date=from_date,
            to_date=to_date
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ANALYTICS_FAILED",
                "message": f"An error occurred while fetching category breakdown: {str(e)}",
                "message_ar": f"حدث خطأ أثناء جلب تفاصيل التصنيف: {str(e)}"
            }
        )


@router.get("/department-breakdown")
async def get_department_breakdown(
    from_date: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)"),
    limit: int = Query(10, ge=1, le=50, description="Max departments to return")
):
    """
    Fetch department breakdown for red flags analytics cards.
    
    **Purpose:** Distribution of red flags across departments with status breakdown.
    
    **Example Request:**
    ```
    GET /api/red-flags/department-breakdown?from_date=2024-01-01&limit=10
    ```
    
    **Query Parameters:**
    - `from_date`: Start date for filtering (optional)
    - `to_date`: End date for filtering (optional)
    - `limit`: Max number of departments (default: 10, max: 50)
    
    **Returns:**
    - Total count of red flags
    - Date range applied
    - Array of departments with:
      - Department name (Arabic and English)
      - Count of red flags in department
      - Percentage of total
      - Status breakdown (OPEN, UNDER_REVIEW, FINISHED)
    
    **Notes:**
    - Results sorted by count descending
    - Shows top N departments by red flag count
    - Used for dashboard department distribution cards
    """
    
    try:
        result = get_red_flags_department_breakdown(
            from_date=from_date,
            to_date=to_date,
            limit=limit
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ANALYTICS_FAILED",
                "message": f"An error occurred while fetching department breakdown: {str(e)}",
                "message_ar": f"حدث خطأ أثناء جلب تفاصيل الأقسام: {str(e)}"
            }
        )


@router.get("/{red_flag_id}")
async def get_single_red_flag(red_flag_id: int):
    """
    Fetch comprehensive details for a specific red flag (for modal view).
    
    **Example Request:**
    ```
    GET /api/red-flags/1
    ```
    
    **Path Parameters:**
    - `red_flag_id`: Unique identifier for the red flag
    
    **Returns:**
    - Full red flag record
    - Linked incident details (complaint text, actions, root cause)
    - Timeline of status changes and events
    - Related follow-up actions
    
    **Errors:**
    - 404: Red flag not found
    - 500: Server error
    """
    
    try:
        result = get_red_flag_details(red_flag_id)
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "RED_FLAG_NOT_FOUND",
                    "message": f"Red flag with ID {red_flag_id} not found",
                    "message_ar": f"لم يتم العثور على العلم الأحمر ذو المعرف {red_flag_id}"
                }
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "QUERY_FAILED",
                "message": f"An error occurred while fetching red flag: {str(e)}",
                "message_ar": f"حدث خطأ أثناء جلب العلم الأحمر: {str(e)}"
            }
        )


@router.post("/{red_flag_id}/export-pdf")
async def export_red_flag_pdf(red_flag_id: int):
    """
    Generate PDF report for a specific red flag (for governance documentation).
    
    **Note:** PDF export functionality is not yet implemented.
    This endpoint is reserved for future implementation.
    
    **Example Request:**
    ```
    POST /api/red-flags/1/export-pdf
    ```
    
    **Path Parameters:**
    - `red_flag_id`: Unique identifier for the red flag
    
    **Returns:**
    - PDF file (when implemented)
    - Or job ID for async generation
    """
    
    raise HTTPException(
        status_code=501,
        detail={
            "error": "NOT_IMPLEMENTED",
            "message": "PDF export functionality is not yet implemented",
            "message_ar": "وظيفة تصدير PDF غير مطبقة بعد"
        }
    )


@router.post("/export-batch")
async def export_batch_red_flags():
    """
    Export multiple red flags based on filters (for reporting/auditing).
    
    **Note:** Batch export functionality is not yet implemented.
    This endpoint is reserved for future implementation.
    
    **Example Request:**
    ```
    POST /api/red-flags/export-batch
    {
      "filters": {
        "status": "FINISHED",
        "from_date": "2024-01-01",
        "to_date": "2024-12-31"
      },
      "format": "pdf"
    }
    ```
    
    **Returns:**
    - Job ID for async export
    - Download URL when ready
    """
    
    raise HTTPException(
        status_code=501,
        detail={
            "error": "NOT_IMPLEMENTED",
            "message": "Batch export functionality is not yet implemented",
            "message_ar": "وظيفة التصدير الجماعي غير مطبقة بعد"
        }
    )


@router.get("/test")
async def test_red_flags():
    """
    Test endpoint to verify Red Flags service is working.
    """
    
    return {
        "status": "operational",
        "service": "red-flags",
        "message": "Red Flags API is operational",
        "endpoints": [
            "GET /api/red-flags",
            "GET /api/red-flags/statistics",
            "GET /api/red-flags/trends",
            "GET /api/red-flags/category-breakdown",
            "GET /api/red-flags/department-breakdown",
            "GET /api/red-flags/{id}",
            "POST /api/red-flags/{id}/export-pdf (not implemented)",
            "POST /api/red-flags/export-batch (not implemented)"
        ]
    }

