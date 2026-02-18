"""
Never Events Router
API endpoints for Never Events page.
Never events are zero-tolerance incidents requiring immediate reporting and investigation.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from ..services.never_events_service import (
    get_never_events_list,
    get_never_events_statistics,
    get_never_events_trends,
    get_never_event_details,
    get_never_events_category_breakdown,
    get_never_events_timeline_comparison
)


router = APIRouter(prefix="/api/never-events", tags=["Never Events"])


# ==================== ENDPOINTS ====================

@router.get("")
async def get_never_events(
    search: Optional[str] = Query(None, description="Search by record ID, patient name, or event type"),
    status: Optional[str] = Query(None, description="Filter by status: OPEN, UNDER_INVESTIGATION, RESOLVED, CLOSED, all"),
    from_date: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)"),
    department: Optional[str] = Query(None, description="Filter by department"),
    category: Optional[str] = Query(None, description="Filter by never event category"),
    sort_by: str = Query("date", description="Sort by: date, incident_date, severity, department, status, category, patient_name"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    limit: int = Query(100, ge=1, le=500, description="Max results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """
    Fetch list of never event incidents with optional filtering and search.
    
    **Never Events** are zero-tolerance incidents (ClinicalRiskTypeID = 3) that should never occur.
    
    **Example Request:**
    ```
    GET /api/never-events?status=UNDER_INVESTIGATION&from_date=2024-01-01&sort_by=severity&sort_order=desc&limit=50
    ```
    
    **Query Parameters:**
    - `search`: Search across record ID, patient name, and event type
    - `status`: Filter by OPEN, UNDER_INVESTIGATION, RESOLVED, CLOSED, or "all"
    - `from_date`, `to_date`: Date range filters
    - `department`: Filter by department name
    - `category`: Filter by never event category
    - `sort_by`: Sort by date (default), incident_date, severity, department, status, category, or patient_name
    - `sort_order`: asc or desc (default: desc for newest first)
    - `limit`: Results per page (default: 100, max: 500)
    - `offset`: Pagination offset (default: 0)
    
    **Returns:**
    - List of never event records with full patient information, investigation details, and tracking
    - Total count for pagination
    - Goal (always 0 - zero tolerance)
    - Message about zero tolerance target
    - Sorted by incident date descending (most recent first) by default
    - Includes fields: patient_full_name, incident_description, investigation_status, root_cause, etc.
    """
    
    try:
        result = get_never_events_list(
            search=search,
            status=status,
            from_date=from_date,
            to_date=to_date,
            department=department,
            category=category,
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
                "message": f"An error occurred while fetching never events: {str(e)}",
                "message_ar": f"حدث خطأ أثناء جلب أحداث لا يجب أن تحدث: {str(e)}"
            }
        )


@router.get("/statistics")
async def get_statistics(
    from_date: Optional[str] = Query(None, description="Statistics from date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="Statistics to date (YYYY-MM-DD)")
):
    """
    Fetch summary statistics for Never Events KPI cards.
    
    **Example Request:**
    ```
    GET /api/never-events/statistics?from_date=2024-01-01&to_date=2024-12-31
    ```
    
    **Query Parameters:**
    - `from_date`: Statistics from this date (optional, default: all time)
    - `to_date`: Statistics to this date (optional, default: today)
    
    **Returns:**
    - `total_never_events`: Total count of never events
    - `goal`: Target (always 0 - zero tolerance)
    - `variance`: Difference from goal
    - `ytd_total`: Year-to-date total
    - `unfinished_count`: Count of never events not yet resolved
    - `finished_count`: Count of completed never events
    - `by_status`: Nested object with counts by status (OPEN, UNDER_INVESTIGATION, RCA_IN_PROGRESS, PENDING_REVIEW, RESOLVED, CLOSED)
    - `by_severity`: Nested object with counts by severity (CRITICAL, HIGH, MEDIUM, LOW)
    - `by_category`: Detailed category breakdown with count and percentage
    - `by_harm_level`: Distribution of harm levels
    - `current_month`: Object with count, month name, start_date, end_date, goal, and status
    - `previous_month`: Object with count, month name, and comparison percentage
    - `rca_statistics`: Root cause analysis statistics (completed, in_progress, completion_rate)
    - `performance_indicators`: Time to investigation, time to resolution, recurrence rate
    - `period`: Date range applied (from_date, to_date)
    
    **Notes:**
    - Never events are zero-tolerance incidents - goal is always 0
    - Investigation status tracking included
    - RCA completion metrics provided
    - Current month status shows CRITICAL if count > 0
    """
    
    try:
        result = get_never_events_statistics(
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
    group_by: str = Query("none", description="category, department, or none")
):
    """
    Fetch time-series trend data for Never Events visualization.
    
    **Example Request:**
    ```
    GET /api/never-events/trends?granularity=monthly&group_by=category
    ```
    
    **Query Parameters:**
    - `from_date`: Start date for trend (default: 12 months ago)
    - `to_date`: End date for trend (default: today)
    - `granularity`: Time period grouping (monthly, quarterly, weekly)
    - `group_by`: Group by category, department, or none
    
    **Returns:**
    - Time-series data for trend chart
    - Period labels (e.g., "Jan 2024", "Q1 2024")
    - Counts per period
    - Optional breakdown by selected grouping
    
    **Notes:**
    - Periods with zero never events are included for chart continuity
    - Default shows last 12 months
    - Grouping allows stacked or grouped trend charts
    - Goal is zero never events per period
    """
    
    try:
        result = get_never_events_trends(
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
    Fetch category breakdown for never events analytics cards.
    
    **Purpose:** Distribution of never events across categories with specific event types.
    
    **Example Request:**
    ```
    GET /api/never-events/category-breakdown?from_date=2024-01-01&to_date=2024-12-31
    ```
    
    **Query Parameters:**
    - `from_date`: Start date for filtering (optional)
    - `to_date`: End date for filtering (optional)
    
    **Returns:**
    - Total count of never events
    - Goal (always 0)
    - Date range applied
    - Array of categories with:
      - Category name (English and Arabic)
      - Count of never events in category
      - Percentage of total
      - Array of specific event types within category
    
    **Notes:**
    - Results sorted by count descending
    - Used for dashboard category distribution cards
    - Drill-down view showing specific event types
    - Goal = 0 (zero tolerance)
    """
    
    try:
        result = get_never_events_category_breakdown(
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


@router.get("/timeline-comparison")
async def get_timeline_comparison(
    period: str = Query("month", description="Time period: month, quarter, or year")
):
    """
    Fetch timeline comparison for never events progress tracking.
    
    **Purpose:** Compare current vs previous period to track progress toward zero.
    
    **Example Request:**
    ```
    GET /api/never-events/timeline-comparison?period=month
    ```
    
    **Query Parameters:**
    - `period`: Time period for comparison (month, quarter, year) - default: month
    
    **Returns:**
    - Goal (always 0)
    - Current period data (count, dates, period name)
    - Previous period data (count, period name)
    - Change analysis (absolute, percentage, trend)
    - Year-to-date statistics
    
    **Notes:**
    - Trend = "improving" when count decreases
    - Trend = "worsening" when count increases
    - Trend = "stable" when count stays the same
    - Used for dashboard comparison cards
    - Goal = 0 (zero tolerance)
    """
    
    try:
        result = get_never_events_timeline_comparison(period=period)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ANALYTICS_FAILED",
                "message": f"An error occurred while fetching timeline comparison: {str(e)}",
                "message_ar": f"حدث خطأ أثناء جلب مقارنة الجدول الزمني: {str(e)}"
            }
        )


@router.get("/{never_event_id}")
async def get_single_never_event(never_event_id: int):
    """
    Fetch comprehensive details for a specific never event (for modal view).
    
    **Example Request:**
    ```
    GET /api/never-events/1
    ```
    
    **Path Parameters:**
    - `never_event_id`: Unique identifier for the never event
    
    **Returns:**
    - Full never event record
    - Linked incident details (complaint text, actions, root cause)
    - Timeline of status changes and events
    - Related follow-up actions
    
    **Notes:**
    - Never events are always linked to an underlying incident
    - Timeline provides audit trail
    - Related actions track follow-up and corrective measures
    
    **Errors:**
    - 404: Never event not found
    - 500: Server error
    """
    
    try:
        result = get_never_event_details(never_event_id)
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NEVER_EVENT_NOT_FOUND",
                    "message": f"Never event with ID {never_event_id} not found",
                    "message_ar": f"لم يتم العثور على الحدث الذي لا يجب أن يحدث بالمعرف {never_event_id}"
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
                "message": f"An error occurred while fetching never event: {str(e)}",
                "message_ar": f"حدث خطأ أثناء جلب الحدث: {str(e)}"
            }
        )


@router.get("/test")
async def test_never_events():
    """
    Test endpoint to verify Never Events service is working.
    """
    
    return {
        "status": "operational",
        "service": "never-events",
        "message": "Never Events API is operational",
        "endpoints": [
            "GET /api/never-events",
            "GET /api/never-events/statistics",
            "GET /api/never-events/trends",
            "GET /api/never-events/category-breakdown",
            "GET /api/never-events/timeline-comparison",
            "GET /api/never-events/{id}"
        ]
    }

