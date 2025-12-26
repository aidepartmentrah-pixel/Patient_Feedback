from datetime import date, timedelta
from fastapi import APIRouter, Query, HTTPException
import traceback
from dateutil.relativedelta import relativedelta
from ..services.trend_service import (
    get_domain_trends,
    get_category_trends,
    get_time_periods,
)

router = APIRouter(
    prefix="/api/trends",
    tags=["Trends"],
)


# =========================================================
# DOMAIN TRENDS ENDPOINT
# =========================================================

@router.get("/domains")
def fetch_domain_trends(
    start_date: str | None = Query(
        None,
        description="Start month in YYYY-MM format (defaults to 12 months ago)",
        regex=r"^\d{4}-\d{2}$"
    ),
    end_date: str | None = Query(
        None,
        description="End month in YYYY-MM format (defaults to current month)",
        regex=r"^\d{4}-\d{2}$"
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
    start_date: str | None = Query(
        None,
        description="Start month in YYYY-MM format (defaults to 12 months ago)",
        regex=r"^\d{4}-\d{2}$"
    ),
    end_date: str | None = Query(
        None,
        description="End month in YYYY-MM format (defaults to current month)",
        regex=r"^\d{4}-\d{2}$"
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
    start_date: str | None = Query(
        None,
        description="Filter periods from this month (YYYY-MM format)",
        regex=r"^\d{4}-\d{2}$"
    ),
    end_date: str | None = Query(
        None,
        description="Filter periods to this month (YYYY-MM format)",
        regex=r"^\d{4}-\d{2}$"
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
