from datetime import date, timedelta
from fastapi import APIRouter, Query, HTTPException, Depends
import traceback
from dateutil.relativedelta import relativedelta
from ..services.trend_service import (
    get_domain_trends,
    get_category_trends,
    get_time_periods,
)
from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_unit_in_scope

router = APIRouter(
    prefix="/api/trends",
    tags=["Trends"],
)


# =========================================================
# DOMAIN TRENDS ENDPOINT
# =========================================================

@router.get("/domains")
def fetch_domain_trends(
    current_user: CurrentUser = Depends(get_current_user),
    start_date: str | None = Query(
        None,
        description="Start month in YYYY-MM format (defaults to 12 months ago)",
        pattern=r"^\d{4}-\d{2}$"
    ),
    end_date: str | None = Query(
        None,
        description="End month in YYYY-MM format (defaults to current month)",
        pattern=r"^\d{4}-\d{2}$"
    ),
    include_zero_months: bool = Query(
        True,
        description="Include months with zero incidents for continuous timeline"
    ),
    include_inactive_domains: bool = Query(
        False,
        description="Include domains with no incidents in the time period"
    ),
    calculate_trends: bool = Query(
        True,
        description="Calculate trend_direction and trend_percentage fields"
    ),
):
    """
    Retrieve aggregated monthly incident counts grouped by clinical domain (HCAT).
    
    Returns trend data suitable for line charts, area charts, and trend tables.
    
    **Default Behavior:**
    - Time range: Last 12 months
    - Zero months: Included (ensures continuous timeline)
    - Inactive domains: Excluded
    - Trend calculations: Enabled
    
    **Response includes:**
    - Monthly incident counts per domain
    - Trend direction (increasing/stable/decreasing)
    - Peak months and counts
    - Month-over-month change percentages
    - Domain metadata (names, codes, colors)
    """
    
    print("=" * 80)
    print("DOMAIN TRENDS REQUEST RECEIVED:")
    print(f"  start_date: {start_date}")
    print(f"  end_date: {end_date}")
    print(f"  include_zero_months: {include_zero_months}")
    print(f"  include_inactive_domains: {include_inactive_domains}")
    print(f"  calculate_trends: {calculate_trends}")
    print("=" * 80)
    
    # -------------------------
    # Parse and validate dates
    # -------------------------
    try:
        start_date_obj = None
        end_date_obj = None
        
        if start_date:
            year, month = map(int, start_date.split("-"))
            start_date_obj = date(year, month, 1)
        
        if end_date:
            year, month = map(int, end_date.split("-"))
            end_date_obj = date(year, month, 1)
        
        # Validate date range
        if start_date_obj and end_date_obj:
            if start_date_obj > end_date_obj:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_date_range",
                        "message": "start_date must be before end_date",
                        "message_ar": "يجب أن يكون تاريخ البداية قبل تاريخ النهاية"
                    }
                )
            
            # Check maximum range (36 months = 3 years)
            months_diff = (end_date_obj.year - start_date_obj.year) * 12 + (end_date_obj.month - start_date_obj.month)
            if months_diff > 36:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "date_range_too_large",
                        "message": "Maximum allowed time range is 36 months (3 years)",
                        "message_ar": "الحد الأقصى للفترة الزمنية هو 36 شهرًا (3 سنوات)"
                    }
                )
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_date_format",
                "message": "Date must be in YYYY-MM format (e.g., 2024-01)",
                "message_ar": "يجب أن يكون التاريخ بصيغة YYYY-MM (مثال: 2024-01)"
            }
        )
    
    # -------------------------
    # Call service
    # -------------------------
    try:
        return get_domain_trends(
            current_user=current_user,
            start_date=start_date_obj,
            end_date=end_date_obj,
            include_zero_months=include_zero_months,
            include_inactive_domains=include_inactive_domains,
            calculate_trends=calculate_trends,
        )
    
    except Exception as e:
        print(f"Domain trends error:")
        print(f"  start_date: {start_date}")
        print(f"  end_date: {end_date}")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# =========================================================
# CATEGORY TRENDS ENDPOINT
# =========================================================

@router.get("/categories")
def fetch_category_trends(
    current_user: CurrentUser = Depends(get_current_user),
    start_date: str | None = Query(
        None,
        description="Start month in YYYY-MM format (defaults to 12 months ago)",
        pattern=r"^\d{4}-\d{2}$"
    ),
    end_date: str | None = Query(
        None,
        description="End month in YYYY-MM format (defaults to current month)",
        pattern=r"^\d{4}-\d{2}$"
    ),
    domain_id: int | None = Query(
        None,
        description="Filter categories by parent domain ID"
    ),
    include_zero_months: bool = Query(
        True,
        description="Include months with zero incidents for continuous timeline"
    ),
    include_inactive_categories: bool = Query(
        False,
        description="Include categories with no incidents in the time period"
    ),
    calculate_trends: bool = Query(
        True,
        description="Calculate trend_direction and trend_percentage fields"
    ),
    top_n: int | None = Query(
        None,
        ge=1,
        le=100,
        description="Return only top N categories by total incidents (max: 100)"
    ),
):
    """
    Retrieve aggregated monthly incident counts grouped by category.
    
    Returns trend data suitable for line charts, area charts, and trend tables.
    Supports filtering by parent domain and limiting to top N categories.
    
    **Default Behavior:**
    - Time range: Last 12 months
    - Zero months: Included (ensures continuous timeline)
    - Inactive categories: Excluded
    - Trend calculations: Enabled
    - Top N filter: Disabled (returns all)
    
    **Response includes:**
    - Monthly incident counts per category
    - Trend direction (increasing/stable/decreasing)
    - Peak months and counts
    - Percentage of domain and total incidents
    - Month-over-month change percentages
    - Category and parent domain metadata
    """
    
    print("=" * 80)
    print("CATEGORY TRENDS REQUEST RECEIVED:")
    print(f"  start_date: {start_date}")
    print(f"  end_date: {end_date}")
    print(f"  domain_id: {domain_id}")
    print(f"  include_zero_months: {include_zero_months}")
    print(f"  include_inactive_categories: {include_inactive_categories}")
    print(f"  calculate_trends: {calculate_trends}")
    print(f"  top_n: {top_n}")
    print("=" * 80)
    
    # -------------------------
    # Parse and validate dates
    # -------------------------
    try:
        start_date_obj = None
        end_date_obj = None
        
        if start_date:
            year, month = map(int, start_date.split("-"))
            start_date_obj = date(year, month, 1)
        
        if end_date:
            year, month = map(int, end_date.split("-"))
            end_date_obj = date(year, month, 1)
        
        # Validate date range
        if start_date_obj and end_date_obj:
            if start_date_obj > end_date_obj:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_date_range",
                        "message": "start_date must be before end_date",
                        "message_ar": "يجب أن يكون تاريخ البداية قبل تاريخ النهاية"
                    }
                )
            
            # Check maximum range (36 months = 3 years)
            months_diff = (end_date_obj.year - start_date_obj.year) * 12 + (end_date_obj.month - start_date_obj.month)
            if months_diff > 36:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "date_range_too_large",
                        "message": "Maximum allowed time range is 36 months (3 years)",
                        "message_ar": "الحد الأقصى للفترة الزمنية هو 36 شهرًا (3 سنوات)"
                    }
                )
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_date_format",
                "message": "Date must be in YYYY-MM format (e.g., 2024-01)",
                "message_ar": "يجب أن يكون التاريخ بصيغة YYYY-MM (مثال: 2024-01)"
            }
        )
    
    # -------------------------
    # Call service
    # -------------------------
    try:
        return get_category_trends(
            current_user=current_user,
            start_date=start_date_obj,
            end_date=end_date_obj,
            domain_id=domain_id,
            include_zero_months=include_zero_months,
            include_inactive_categories=include_inactive_categories,
            calculate_trends=calculate_trends,
            top_n=top_n,
        )
    
    except Exception as e:
        print(f"Category trends error:")
        print(f"  start_date: {start_date}")
        print(f"  end_date: {end_date}")
        print(f"  domain_id: {domain_id}")
        print(f"  top_n: {top_n}")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# =========================================================
# TIME PERIODS ENDPOINT
# =========================================================

@router.get("/periods")
def fetch_time_periods(
    current_user: CurrentUser = Depends(get_current_user),
    start_date: str | None = Query(
        None,
        description="Filter periods from this month (YYYY-MM format)",
        pattern=r"^\d{4}-\d{2}$"
    ),
    end_date: str | None = Query(
        None,
        description="Filter periods to this month (YYYY-MM format)",
        pattern=r"^\d{4}-\d{2}$"
    ),
):
    """
    Retrieve list of available time periods (months) with incident data.
    
    Useful for:
    - Timeline selection dropdowns
    - Validating available date ranges
    - Displaying period metadata (start/end dates, labels)
    
    **Default Behavior:**
    - Time range: Last 12 months
    - Returns all months in range (even those with zero incidents)
    
    **Response includes:**
    - Period identifiers (YYYY-MM format)
    - Human-readable labels (English and Arabic)
    - Start and end dates for each period
    - Incident counts per period
    - has_data flag (indicates if period has incidents)
    """
    
    print("=" * 80)
    print("TIME PERIODS REQUEST RECEIVED:")
    print(f"  start_date: {start_date}")
    print(f"  end_date: {end_date}")
    print("=" * 80)
    
    # -------------------------
    # Parse and validate dates
    # -------------------------
    try:
        start_date_obj = None
        end_date_obj = None
        
        if start_date:
            year, month = map(int, start_date.split("-"))
            start_date_obj = date(year, month, 1)
        
        if end_date:
            year, month = map(int, end_date.split("-"))
            end_date_obj = date(year, month, 1)
        
        # Validate date range
        if start_date_obj and end_date_obj:
            if start_date_obj > end_date_obj:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_date_range",
                        "message": "start_date must be before end_date",
                        "message_ar": "يجب أن يكون تاريخ البداية قبل تاريخ النهاية"
                    }
                )
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_date_format",
                "message": "Date must be in YYYY-MM format (e.g., 2024-01)",
                "message_ar": "يجب أن يكون التاريخ بصيغة YYYY-MM (مثال: 2024-01)"
            }
        )
    
    # -------------------------
    # Call service
    # -------------------------
    try:
        return get_time_periods(
            current_user=current_user,
            start_date=start_date_obj,
            end_date=end_date_obj,
        )
    
    except Exception as e:
        print(f"Time periods error:")
        print(f"  start_date: {start_date}")
        print(f"  end_date: {end_date}")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# =========================================================
# UNIFIED TRENDS ENDPOINT (with scope selection)
# =========================================================

@router.get("/analysis")
def fetch_trends_analysis(
    current_user: CurrentUser = Depends(get_current_user),
    scope: str = Query("hospital", description="hospital | administration | department | section"),
    administration_id: int | None = Query(None),
    department_id: int | None = Query(None),
    section_id: int | None = Query(None),
    start_date: str | None = Query(None, description="Start month in YYYY-MM format"),
    end_date: str | None = Query(None, description="End month in YYYY-MM format"),
    include_zero_months: bool = Query(True),
):
    """
    Unified trends endpoint with scope selection.
    Returns domain, category, and classification trends for selected scope.
    
    **Scope options:**
    - hospital: All incidents across the hospital
    - administration: Filtered by administration unit
    - department: Filtered by department unit
    - section: Filtered by section unit
    
    **Response includes:**
    - Domain trends (clinical, management, relational)
    - Category trends
    - Classification trends
    - Each with: table data, trend direction, monthly breakdown
    """
    
    print("=" * 80)
    print("TRENDS ANALYSIS REQUEST RECEIVED:")
    print(f"  scope: {scope}")
    print(f"  administration_id: {administration_id}")
    print(f"  department_id: {department_id}")
    print(f"  section_id: {section_id}")
    print(f"  start_date: {start_date}")
    print(f"  end_date: {end_date}")
    print("=" * 80)
    
    # Validate scope
    if scope not in {"hospital", "administration", "department", "section"}:
        raise HTTPException(status_code=400, detail="Invalid scope")
    
    if scope == "administration" and administration_id is None:
        raise HTTPException(status_code=400, detail="administration_id required for administration scope")
    
    if scope == "department" and department_id is None:
        raise HTTPException(status_code=400, detail="department_id required for department scope")
    
    if scope == "section" and section_id is None:
        raise HTTPException(status_code=400, detail="section_id required for section scope")
    
    # -------------------------
    # Enforce organizational scope (Phase 2.5)
    # Validate client-provided org unit IDs against user's allowed scope
    # -------------------------
    if scope == "administration" and administration_id is not None:
        require_unit_in_scope(current_user, administration_id)
    
    if scope == "department" and department_id is not None:
        require_unit_in_scope(current_user, department_id)
    
    if scope == "section" and section_id is not None:
        require_unit_in_scope(current_user, section_id)
    
    # Parse dates
    try:
        start_date_obj = None
        end_date_obj = None
        
        if start_date:
            year, month = map(int, start_date.split("-"))
            start_date_obj = date(year, month, 1)
        
        if end_date:
            year, month = map(int, end_date.split("-"))
            end_date_obj = date(year, month, 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM format")
    
    try:
        from ..services.trend_service import get_trends_analysis
        
        return get_trends_analysis(
            current_user=current_user,
            scope=scope,
            administration_id=administration_id,
            department_id=department_id,
            section_id=section_id,
            start_date=start_date_obj,
            end_date=end_date_obj,
            include_zero_months=include_zero_months,
        )
    
    except Exception as e:
        print(f"Trends analysis error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")