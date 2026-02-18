from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import defaultdict, Counter
from typing import Literal
import calendar
from core.database import get_connection
from ..db_layer import lookups, incident_case
from ..schemas.auth_models import CurrentUser
from . import org_tree_service


# =========================================================
# CONSTANTS
# =========================================================

ARABIC_MONTHS = {
    1: "يناير",
    2: "فبراير",
    3: "مارس",
    4: "أبريل",
    5: "مايو",
    6: "يونيو",
    7: "يوليو",
    8: "أغسطس",
    9: "سبتمبر",
    10: "أكتوبر",
    11: "نوفمبر",
    12: "ديسمبر",
}

ENGLISH_MONTHS = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

# Domain color mapping (HCAT standard colors)
DOMAIN_COLORS = {
    1: "#2196f3",  # Clinical - Blue
    2: "#4caf50",  # Management - Green
    3: "#ff9800",  # Relational - Orange
    4: "#9c27b0",  # Purple
    5: "#f44336",  # Red
    6: "#00bcd4",  # Cyan
    7: "#ffeb3b",  # Yellow
    8: "#795548",  # Brown
}

CATEGORY_COLORS = {
    # You can expand this based on your categories
    # For now, we'll generate colors dynamically
}

TREND_THRESHOLD = 0.10  # 10% threshold for stable vs increasing/decreasing


# =========================================================
# PUBLIC SERVICE FUNCTIONS
# =========================================================

def get_trends_analysis(
    *,
    current_user: CurrentUser,
    scope: str,
    administration_id: int | None = None,
    department_id: int | None = None,
    section_id: int | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    include_zero_months: bool = True,
) -> dict:
    """
    Get unified trends analysis for domain, category, and classification
    with scope filtering (hospital, administration, department, section).
    
    Trend Scope Filtering (Fixed):
    - Respects client-requested scope parameters (section/department/administration/hospital)
    - Expands hierarchical scopes using org_tree_service.get_descendants()
    - Intersects requested scope with current_user.allowed_unit_ids for RBAC safety
    - Router guards have already validated that requested IDs are accessible
    
    Returns all three trend views with table and chart data.
    """
    
    # Date range defaults
    if end_date is None:
        end_date = date.today()
    
    if start_date is None:
        start_date = end_date - relativedelta(months=12)
    
    # Normalize to first and last day of month for filtering
    start_date_normalized = date(start_date.year, start_date.month, 1)
    end_date_normalized = date(end_date.year, end_date.month, 1) + relativedelta(months=1) - timedelta(days=1)
    
    # -------------------------
    # Determine Requested Scope
    # -------------------------
    if scope == "section" and section_id is not None:
        requested_unit_ids = {section_id}
    elif scope == "department" and department_id is not None:
        requested_unit_ids = org_tree_service.get_descendants(department_id)
    elif scope == "administration" and administration_id is not None:
        requested_unit_ids = org_tree_service.get_descendants(administration_id)
    else:
        requested_unit_ids = current_user.allowed_unit_ids
    
    # RBAC Safety: Intersect with allowed scope
    scope_unit_ids = list(requested_unit_ids & current_user.allowed_unit_ids)
    
    # Fetch incidents using database-level filtering
    incidents = incident_case.list_incident_cases_filtered(
        unit_ids=scope_unit_ids,
        start_date=start_date_normalized,
        end_date=end_date_normalized
    )
    
    # Build trends for each view (use normalized dates for grouping)
    domain_data = _build_domain_trends(incidents, start_date_normalized, end_date_normalized, include_zero_months)
    category_data = _build_category_trends(incidents, start_date_normalized, end_date_normalized, include_zero_months)
    classification_data = _build_classification_trends(incidents, start_date_normalized, end_date_normalized, include_zero_months)
    
    return {
        "scope": scope,
        "time_range": {
            "start": start_date_normalized.isoformat(),
            "end": end_date_normalized.isoformat(),
        },
        "domain": domain_data,
        "category": category_data,
        "classification": classification_data,
    }


def get_domain_trends(
    *,
    current_user: CurrentUser,
    start_date: date | None = None,
    end_date: date | None = None,
    include_zero_months: bool = True,
    include_inactive_domains: bool = False,
    calculate_trends: bool = True,
) -> dict:
    """
    Get monthly incident trends aggregated by domain.
    
    Uses current_user.allowed_unit_ids for filtering (no client scope params in this function).
    
    Args:
        current_user: Authenticated user with allowed org units
        start_date: Start month (YYYY-MM format or date object)
        end_date: End month (YYYY-MM format or date object)
        include_zero_months: Include months with zero incidents
        include_inactive_domains: Include domains with no incidents in period
        calculate_trends: Calculate trend_direction and trend_percentage
        
    Returns:
        Dictionary with time_range, summary, and domains array
    """
    # -------------------------
    # Date range defaults
    # -------------------------
    if end_date is None:
        end_date = date.today()
    
    if start_date is None:
        start_date = end_date - relativedelta(months=12)
    
    # Normalize to first day of month
    start_date = date(start_date.year, start_date.month, 1)
    end_date = date(end_date.year, end_date.month, 1)
    
    # -------------------------
    # Use full allowed scope (no client params in this function)
    # -------------------------
    scope_unit_ids = list(current_user.allowed_unit_ids)
    raw_data = _fetch_incidents_by_domain_and_month(start_date, end_date, scope_unit_ids)
    
    # -------------------------
    # Get domain metadata
    # -------------------------
    domains_lookup = {d["DomainID"]: d for d in lookups.get_domains()}
    
    # -------------------------
    # Build monthly data structure
    # -------------------------
    all_months = _generate_month_range(start_date, end_date)
    
    domain_aggregations = defaultdict(lambda: {
        "monthly_counts": defaultdict(int),
        "total": 0,
    })
    
    for row in raw_data:
        domain_id = row["DomainID"]
        period = row["Period"]
        count = row["IncidentCount"]
        
        domain_aggregations[domain_id]["monthly_counts"][period] = count
        domain_aggregations[domain_id]["total"] += count
    
    # -------------------------
    # Filter inactive domains
    # -------------------------
    if not include_inactive_domains:
        domain_aggregations = {
            did: data for did, data in domain_aggregations.items()
            if data["total"] > 0
        }
    
    # -------------------------
    # Build response structure
    # -------------------------
    domains_result = []
    total_incidents_all_domains = 0
    
    for domain_id, aggregation in domain_aggregations.items():
        domain_info = domains_lookup.get(domain_id)
        
        if not domain_info:
            continue  # Skip if domain doesn't exist in lookup
        
        monthly_data = []
        monthly_counts_list = []
        
        for month_info in all_months:
            period = month_info["period"]
            count = aggregation["monthly_counts"].get(period, 0)
            
            if count == 0 and not include_zero_months:
                continue
            
            monthly_counts_list.append(count)
            
            # Calculate month-over-month change
            prev_period = _get_previous_month(period)
            prev_count = aggregation["monthly_counts"].get(prev_period)
            
            month_over_month_change = None
            month_over_month_percentage = None
            
            if prev_count is not None:
                month_over_month_change = count - prev_count
                if prev_count > 0:
                    month_over_month_percentage = round((month_over_month_change / prev_count) * 100, 1)
            
            monthly_data.append({
                "period": period,
                "month": month_info["month"],
                "year": month_info["year"],
                "period_label": month_info["period_label"],
                "period_label_ar": month_info["period_label_ar"],
                "incident_count": count,
                "percentage_of_total": None,  # Will calculate after we have totals
                "month_over_month_change": month_over_month_change,
                "month_over_month_percentage": month_over_month_percentage,
            })
        
        # -------------------------
        # Calculate trend metrics
        # -------------------------
        trend_direction = None
        trend_percentage = None
        peak_month = None
        peak_count = 0
        
        if calculate_trends and len(monthly_counts_list) > 1:
            trend_direction, trend_percentage = _calculate_trend(monthly_counts_list)
            
            # Find peak month
            for month_data in monthly_data:
                if month_data["incident_count"] > peak_count:
                    peak_count = month_data["incident_count"]
                    peak_month = month_data["period"]
        
        total_incidents = aggregation["total"]
        total_incidents_all_domains += total_incidents
        
        avg_monthly = round(total_incidents / len(all_months), 1) if len(all_months) > 0 else 0
        
        domains_result.append({
            "domain_id": domain_id,
            "domain_name": domain_info["DomainName"],
            "domain_name_ar": domain_info.get("DomainName"),  # Add Arabic if available
            "domain_code": domain_info["DomainCode"],
            "domain_color": DOMAIN_COLORS.get(domain_id, "#607d8b"),
            "total_incidents": total_incidents,
            "average_monthly_incidents": avg_monthly,
            "peak_month": peak_month,
            "peak_count": peak_count,
            "trend_direction": trend_direction,
            "trend_percentage": trend_percentage,
            "monthly_data": monthly_data,
        })
    
    # -------------------------
    # Calculate percentage_of_total for each month
    # -------------------------
    # First, get total incidents per month across all domains
    month_totals = defaultdict(int)
    for domain in domains_result:
        for month_data in domain["monthly_data"]:
            month_totals[month_data["period"]] += month_data["incident_count"]
    
    # Then calculate percentages
    for domain in domains_result:
        for month_data in domain["monthly_data"]:
            period = month_data["period"]
            total_for_month = month_totals[period]
            if total_for_month > 0:
                month_data["percentage_of_total"] = round(
                    (month_data["incident_count"] / total_for_month) * 100, 1
                )
            else:
                month_data["percentage_of_total"] = 0.0
    
    # -------------------------
    # Sort by total incidents (descending)
    # -------------------------
    domains_result.sort(key=lambda x: x["total_incidents"], reverse=True)
    
    # -------------------------
    # Build response
    # -------------------------
    time_range_info = _build_time_range_info(all_months)
    
    return {
        "time_range": time_range_info,
        "summary": {
            "total_incidents": total_incidents_all_domains,
            "total_domains": len(domains_result),
            "average_monthly_incidents": round(
                total_incidents_all_domains / len(all_months), 1
            ) if len(all_months) > 0 else 0,
            "generated_at": datetime.now().isoformat() + "Z",
        },
        "domains": domains_result,
    }


def get_category_trends(
    *,
    current_user: CurrentUser,
    start_date: date | None = None,
    end_date: date | None = None,
    domain_id: int | None = None,
    include_zero_months: bool = True,
    include_inactive_categories: bool = False,
    calculate_trends: bool = True,
    top_n: int | None = None,
) -> dict:
    """
    Get monthly incident trends aggregated by category.
    
    Uses current_user.allowed_unit_ids for filtering (no client scope params in this function).
    
    Args:
        current_user: Authenticated user with allowed org units
        start_date: Start month (YYYY-MM format or date object)
        end_date: End month (YYYY-MM format or date object)
        domain_id: Filter categories by parent domain
        include_zero_months: Include months with zero incidents
        include_inactive_categories: Include categories with no incidents in period
        calculate_trends: Calculate trend_direction and trend_percentage
        top_n: Return only top N categories by total incidents
        
    Returns:
        Dictionary with time_range, summary, and categories array
    """
    # -------------------------
    # Date range defaults
    # -------------------------
    if end_date is None:
        end_date = date.today()
    
    if start_date is None:
        start_date = end_date - relativedelta(months=12)
    
    # Normalize to first day of month
    start_date = date(start_date.year, start_date.month, 1)
    end_date = date(end_date.year, end_date.month, 1)
    
    # -------------------------
    # Use full allowed scope (no client params in this function)
    # -------------------------
    scope_unit_ids = list(current_user.allowed_unit_ids)
    raw_data = _fetch_incidents_by_category_and_month(start_date, end_date, domain_id, scope_unit_ids)
    
    # -------------------------
    # Get category and domain metadata
    # -------------------------
    categories_lookup = {c["CategoryID"]: c for c in lookups.get_categories()}
    domains_lookup = {d["DomainID"]: d for d in lookups.get_domains()}
    
    # -------------------------
    # Build monthly data structure
    # -------------------------
    all_months = _generate_month_range(start_date, end_date)
    
    category_aggregations = defaultdict(lambda: {
        "monthly_counts": defaultdict(int),
        "domain_monthly_counts": defaultdict(int),
        "total": 0,
    })
    
    for row in raw_data:
        category_id = row["CategoryID"]
        period = row["Period"]
        count = row["IncidentCount"]
        
        category_aggregations[category_id]["monthly_counts"][period] = count
        category_aggregations[category_id]["total"] += count
    
    # -------------------------
    # Filter inactive categories
    # -------------------------
    if not include_inactive_categories:
        category_aggregations = {
            cid: data for cid, data in category_aggregations.items()
            if data["total"] > 0
        }
    
    # -------------------------
    # Build response structure
    # -------------------------
    categories_result = []
    total_incidents_all_categories = 0
    
    for category_id, aggregation in category_aggregations.items():
        category_info = categories_lookup.get(category_id)
        
        if not category_info:
            continue  # Skip if category doesn't exist in lookup
        
        domain_info = domains_lookup.get(category_info["DomainID"])
        
        monthly_data = []
        monthly_counts_list = []
        
        for month_info in all_months:
            period = month_info["period"]
            count = aggregation["monthly_counts"].get(period, 0)
            
            if count == 0 and not include_zero_months:
                continue
            
            monthly_counts_list.append(count)
            
            # Calculate month-over-month change
            prev_period = _get_previous_month(period)
            prev_count = aggregation["monthly_counts"].get(prev_period)
            
            month_over_month_change = None
            month_over_month_percentage = None
            
            if prev_count is not None:
                month_over_month_change = count - prev_count
                if prev_count > 0:
                    month_over_month_percentage = round((month_over_month_change / prev_count) * 100, 1)
            
            monthly_data.append({
                "period": period,
                "month": month_info["month"],
                "year": month_info["year"],
                "period_label": month_info["period_label"],
                "period_label_ar": month_info["period_label_ar"],
                "incident_count": count,
                "percentage_of_domain": None,  # Will calculate later
                "percentage_of_total": None,  # Will calculate later
                "month_over_month_change": month_over_month_change,
                "month_over_month_percentage": month_over_month_percentage,
            })
        
        # -------------------------
        # Calculate trend metrics
        # -------------------------
        trend_direction = None
        trend_percentage = None
        peak_month = None
        peak_count = 0
        
        if calculate_trends and len(monthly_counts_list) > 1:
            trend_direction, trend_percentage = _calculate_trend(monthly_counts_list)
            
            # Find peak month
            for month_data in monthly_data:
                if month_data["incident_count"] > peak_count:
                    peak_count = month_data["incident_count"]
                    peak_month = month_data["period"]
        
        total_incidents = aggregation["total"]
        total_incidents_all_categories += total_incidents
        
        avg_monthly = round(total_incidents / len(all_months), 1) if len(all_months) > 0 else 0
        
        category_color = _generate_category_color(category_id)
        
        categories_result.append({
            "category_id": category_id,
            "category_name": category_info["CategoryName"],
            "category_name_ar": category_info.get("CategoryName"),  # Add Arabic if available
            "domain_id": category_info["DomainID"],
            "domain_name": domain_info["DomainName"] if domain_info else None,
            "domain_name_ar": domain_info.get("DomainName") if domain_info else None,
            "category_color": category_color,
            "total_incidents": total_incidents,
            "average_monthly_incidents": avg_monthly,
            "peak_month": peak_month,
            "peak_count": peak_count,
            "trend_direction": trend_direction,
            "trend_percentage": trend_percentage,
            "monthly_data": monthly_data,
        })
    
    # -------------------------
    # Calculate percentage_of_total and percentage_of_domain for each month
    # -------------------------
    # First, get total incidents per month across all categories
    month_totals = defaultdict(int)
    domain_month_totals = defaultdict(lambda: defaultdict(int))
    
    for category in categories_result:
        for month_data in category["monthly_data"]:
            period = month_data["period"]
            count = month_data["incident_count"]
            month_totals[period] += count
            domain_month_totals[category["domain_id"]][period] += count
    
    # Then calculate percentages
    for category in categories_result:
        for month_data in category["monthly_data"]:
            period = month_data["period"]
            count = month_data["incident_count"]
            
            total_for_month = month_totals[period]
            if total_for_month > 0:
                month_data["percentage_of_total"] = round((count / total_for_month) * 100, 1)
            else:
                month_data["percentage_of_total"] = 0.0
            
            domain_total_for_month = domain_month_totals[category["domain_id"]][period]
            if domain_total_for_month > 0:
                month_data["percentage_of_domain"] = round((count / domain_total_for_month) * 100, 1)
            else:
                month_data["percentage_of_domain"] = 0.0
    
    # -------------------------
    # Sort by total incidents (descending)
    # -------------------------
    categories_result.sort(key=lambda x: x["total_incidents"], reverse=True)
    
    # -------------------------
    # Apply top_n filter
    # -------------------------
    if top_n is not None and top_n > 0:
        categories_result = categories_result[:top_n]
    
    # -------------------------
    # Build response
    # -------------------------
    time_range_info = _build_time_range_info(all_months)
    
    return {
        "time_range": time_range_info,
        "summary": {
            "total_incidents": total_incidents_all_categories,
            "total_categories": len(categories_result),
            "average_monthly_incidents": round(
                total_incidents_all_categories / len(all_months), 1
            ) if len(all_months) > 0 else 0,
            "generated_at": datetime.now().isoformat() + "Z",
        },
        "categories": categories_result,
    }


def get_time_periods(
    *,
    current_user: CurrentUser,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict:
    """
    Get list of available time periods (months) with incident data.
    
    Uses current_user.allowed_unit_ids for filtering (no client scope params in this function).
    
    Args:
        current_user: Authenticated user with allowed org units
        start_date: Filter periods from this month
        end_date: Filter periods to this month
        
    Returns:
        Dictionary with periods array and metadata
    """
    # -------------------------
    # Date range defaults
    # -------------------------
    if end_date is None:
        end_date = date.today()
    
    if start_date is None:
        start_date = end_date - relativedelta(months=12)
    
    # Normalize to first day of month
    start_date = date(start_date.year, start_date.month, 1)
    end_date = date(end_date.year, end_date.month, 1)
    
    # -------------------------
    # Use full allowed scope (no client params in this function)
    # -------------------------
    scope_unit_ids = list(current_user.allowed_unit_ids)
    raw_data = _fetch_incidents_per_month(start_date, end_date, scope_unit_ids)
    
    month_counts = {row["Period"]: row["IncidentCount"] for row in raw_data}
    
    # -------------------------
    # Build periods list
    # -------------------------
    all_months = _generate_month_range(start_date, end_date)
    
    periods_result = []
    
    for month_info in all_months:
        period = month_info["period"]
        year = month_info["year"]
        month = month_info["month"]
        
        # Get last day of month
        last_day = calendar.monthrange(year, month)[1]
        
        total_incidents = month_counts.get(period, 0)
        
        periods_result.append({
            "period": period,
            "month": month,
            "year": year,
            "period_label": month_info["period_label"],
            "period_label_ar": month_info["period_label_ar"],
            "period_start_date": f"{year:04d}-{month:02d}-01",
            "period_end_date": f"{year:04d}-{month:02d}-{last_day:02d}",
            "total_incidents": total_incidents,
            "has_data": total_incidents > 0,
        })
    
    periods_with_data = len([p for p in periods_result if p["has_data"]])
    
    return {
        "periods": periods_result,
        "summary": {
            "total_periods": len(periods_result),
            "periods_with_data": periods_with_data,
            "earliest_period": periods_result[0]["period"] if periods_result else None,
            "latest_period": periods_result[-1]["period"] if periods_result else None,
        }
    }


# =========================================================
# PRIVATE HELPER FUNCTIONS
# =========================================================

def _fetch_incidents_by_domain_and_month(start_date: date, end_date: date, org_unit_ids: list[int]) -> list[dict]:
    """
    Fetch raw incident counts grouped by domain and month.
    
    Phase 2.5: Filtered by organizational units.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Build org unit filter
    if not org_unit_ids:
        # No allowed units - return empty result
        conn.close()
        return []
    
    placeholders = ','.join('?' * len(org_unit_ids))
    
    query = f"""
    SELECT 
        DomainID,
        FORMAT(FeedbackRecievedDate, 'yyyy-MM') AS Period,
        COUNT(*) AS IncidentCount
    FROM dbo.APP_IncidentCase
    WHERE FeedbackRecievedDate >= ?
      AND FeedbackRecievedDate < DATEADD(MONTH, 1, ?)
      AND DomainID IS NOT NULL
      AND IssuingOrgUnitID IN ({placeholders})
    GROUP BY DomainID, FORMAT(FeedbackRecievedDate, 'yyyy-MM')
    ORDER BY DomainID, Period
    """
    
    params = [start_date, end_date] + org_unit_ids
    cursor.execute(query, params)
    
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    
    conn.close()
    
    return [dict(zip(columns, row)) for row in rows]


def _fetch_incidents_by_category_and_month(
    start_date: date,
    end_date: date,
    domain_id: int | None = None,
    org_unit_ids: list[int] = None
) -> list[dict]:
    """
    Fetch raw incident counts grouped by category and month.
    
    Phase 2.5: Filtered by organizational units.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Build org unit filter
    if not org_unit_ids:
        # No allowed units - return empty result
        conn.close()
        return []
    
    placeholders = ','.join('?' * len(org_unit_ids))
    
    query = f"""
    SELECT 
        CategoryID,
        FORMAT(FeedbackRecievedDate, 'yyyy-MM') AS Period,
        COUNT(*) AS IncidentCount
    FROM dbo.APP_IncidentCase
    WHERE FeedbackRecievedDate >= ?
      AND FeedbackRecievedDate < DATEADD(MONTH, 1, ?)
      AND CategoryID IS NOT NULL
      AND IssuingOrgUnitID IN ({placeholders})
    """
    
    params = [start_date, end_date] + org_unit_ids
    
    if domain_id is not None:
        query += " AND DomainID = ?"
        params.append(domain_id)
    
    query += """
    GROUP BY CategoryID, FORMAT(FeedbackRecievedDate, 'yyyy-MM')
    ORDER BY CategoryID, Period
    """
    
    cursor.execute(query, params)
    
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    
    conn.close()
    
    return [dict(zip(columns, row)) for row in rows]


def _fetch_incidents_per_month(start_date: date, end_date: date, org_unit_ids: list[int]) -> list[dict]:
    """
    Fetch total incident counts per month.
    
    Phase 2.5: Filtered by organizational units.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Build org unit filter
    if not org_unit_ids:
        # No allowed units - return empty result
        conn.close()
        return []
    
    placeholders = ','.join('?' * len(org_unit_ids))
    
    query = f"""
    SELECT 
        FORMAT(FeedbackRecievedDate, 'yyyy-MM') AS Period,
        COUNT(*) AS IncidentCount
    FROM dbo.APP_IncidentCase
    WHERE FeedbackRecievedDate >= ?
      AND FeedbackRecievedDate < DATEADD(MONTH, 1, ?)
      AND IssuingOrgUnitID IN ({placeholders})
    GROUP BY FORMAT(FeedbackRecievedDate, 'yyyy-MM')
    ORDER BY Period
    """
    
    params = [start_date, end_date] + org_unit_ids
    cursor.execute(query, params)
    
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    
    conn.close()
    
    return [dict(zip(columns, row)) for row in rows]


def _generate_month_range(start_date: date, end_date: date) -> list[dict]:
    """
    Generate list of all months in the date range.
    
    Returns:
        List of dicts with period, month, year, labels
    """
    months = []
    current = start_date
    
    while current <= end_date:
        month = current.month
        year = current.year
        
        months.append({
            "period": f"{year:04d}-{month:02d}",
            "month": month,
            "year": year,
            "period_label": f"{ENGLISH_MONTHS[month]} {year}",
            "period_label_ar": f"{ARABIC_MONTHS[month]} {year}",
        })
        
        current = current + relativedelta(months=1)
    
    return months


def _get_previous_month(period: str) -> str:
    """
    Get previous month period string.
    
    Args:
        period: "YYYY-MM" format
        
    Returns:
        Previous month in "YYYY-MM" format
    """
    year, month = map(int, period.split("-"))
    dt = date(year, month, 1)
    prev = dt - relativedelta(months=1)
    return f"{prev.year:04d}-{prev.month:02d}"


def _calculate_trend(monthly_counts: list[int]) -> tuple[str | None, float | None]:
    """
    Calculate trend direction and percentage change.
    
    Args:
        monthly_counts: List of monthly incident counts in chronological order
        
    Returns:
        Tuple of (trend_direction, trend_percentage)
        - trend_direction: "increasing", "stable", or "decreasing"
        - trend_percentage: Percentage change from first to last month
    """
    if len(monthly_counts) < 2:
        return None, None
    
    first_count = monthly_counts[0]
    last_count = monthly_counts[-1]
    
    # Avoid division by zero
    if first_count == 0:
        if last_count == 0:
            return "stable", 0.0
        else:
            return "increasing", None  # Can't calculate percentage from zero
    
    # Calculate percentage change
    percentage_change = ((last_count - first_count) / first_count) * 100
    
    # Determine trend direction based on threshold
    if percentage_change > (TREND_THRESHOLD * 100):
        trend_direction = "increasing"
    elif percentage_change < -(TREND_THRESHOLD * 100):
        trend_direction = "decreasing"
    else:
        trend_direction = "stable"
    
    return trend_direction, round(percentage_change, 1)


def _build_time_range_info(all_months: list[dict]) -> dict:
    """
    Build time range metadata for response.
    """
    if not all_months:
        return {
            "start": None,
            "end": None,
            "total_months": 0,
            "start_label": None,
            "start_label_ar": None,
            "end_label": None,
            "end_label_ar": None,
        }
    
    first_month = all_months[0]
    last_month = all_months[-1]
    
    return {
        "start": first_month["period"],
        "end": last_month["period"],
        "total_months": len(all_months),
        "start_label": first_month["period_label"],
        "start_label_ar": first_month["period_label_ar"],
        "end_label": last_month["period_label"],
        "end_label_ar": last_month["period_label_ar"],
    }


def _generate_category_color(category_id: int) -> str:
    """
    Generate a consistent color for a category based on its ID.
    Uses a predefined color palette.
    """
    color_palette = [
        "#2196f3", "#4caf50", "#ff9800", "#9c27b0", "#f44336",
        "#00bcd4", "#ffeb3b", "#795548", "#607d8b", "#e91e63",
        "#3f51b5", "#009688", "#ff5722", "#673ab7", "#cddc39",
        "#ffc107", "#8bc34a", "#03a9f4", "#ff6f00", "#1976d2",
    ]
    
    return color_palette[category_id % len(color_palette)]


def _generate_color_from_id(item_id):
    """
    Generate a consistent color for an item based on its ID.
    Uses a predefined color palette.
    """
    color_palette = [
        "#2196f3", "#4caf50", "#ff9800", "#9c27b0", "#f44336",
        "#00bcd4", "#ffeb3b", "#795548", "#607d8b", "#e91e63",
        "#3f51b5", "#009688", "#ff5722", "#673ab7", "#cddc39",
        "#ffc107", "#8bc34a", "#03a9f4", "#ff6f00", "#1976d2",
    ]
    return color_palette[item_id % len(color_palette)]


def _generate_months(start_date, end_date, include_zero_months):
    """Generate list of months between start and end dates"""
    months = []
    current = start_date
    while current <= end_date:
        months.append(current)
        current += relativedelta(months=1)
    return months

# =========================================================
# HELPER FUNCTIONS FOR UNIFIED TRENDS ANALYSIS
# =========================================================

def _build_domain_trends(incidents, start_date, end_date, include_zero_months):
    """Build domain trends for scoped incidents"""
    domain_names = {1: "Clinical", 2: "Management", 3: "Relational"}
    
    # Generate months in range
    months = _generate_months(start_date, end_date, include_zero_months)
    
    # Group incidents by domain and month
    domain_monthly = defaultdict(lambda: defaultdict(int))
    for incident in incidents:
        if incident.get("DomainID"):
            incident_month = incident["CreatedAt"].date()
            incident_month = date(incident_month.year, incident_month.month, 1)
            domain_id = incident["DomainID"]
            domain_monthly[domain_id][incident_month] += 1
    
    # Build table data
    table_data = []
    for domain_id in [1, 2, 3]:
        domain_name = domain_names.get(domain_id, f"Unknown ({domain_id})")
        monthly_counts = [domain_monthly[domain_id].get(m, 0) for m in months]
        total = sum(monthly_counts)
        
        if total > 0 or True:  # Include all domains
            trend_direction, trend_pct = _calculate_trend(monthly_counts)
            table_data.append({
                "id": domain_id,
                "name": domain_name,
                "total": total,
                "color": DOMAIN_COLORS.get(domain_id, "#999999"),
                "trend_direction": trend_direction,
                "trend_percentage": trend_pct,
                "monthly": [
                    {
                        "month": m.strftime("%Y-%m"),
                        "month_ar": f"{ARABIC_MONTHS[m.month]} {m.year}",
                        "count": domain_monthly[domain_id].get(m, 0)
                    }
                    for m in months
                ]
            })
    
    return {
        "table": table_data,
        "chart_data": [
            {
                "name": item["name"],
                "data": [m["count"] for m in item["monthly"]],
                "color": item["color"]
            }
            for item in table_data
        ],
        "chart_labels": [m.strftime("%b %Y") for m in months]
    }


def _build_category_trends(incidents, start_date, end_date, include_zero_months):
    """Build category trends for scoped incidents"""
    # Get all categories (no domain filter)
    try:
        categories = lookups.get_categories()  # No domain_id = get all
    except TypeError:
        # If get_categories requires domain_id, get them all manually
        categories = []
        for domain_id in [1, 2, 3]:
            try:
                cats = lookups.get_categories(domain_id=domain_id)
                categories.extend(cats)
            except:
                pass
    
    if not categories:
        categories = []
    
    category_map = {c["CategoryID"]: c.get("Category_AR", c.get("Category_EN", f"Category {c['CategoryID']}")) 
                   for c in categories if c.get("CategoryID")}
    
    # Generate months
    months = _generate_months(start_date, end_date, include_zero_months)
    
    # Group incidents by category and month
    category_monthly = defaultdict(lambda: defaultdict(int))
    for incident in incidents:
        if incident.get("CategoryID"):
            incident_month = incident["CreatedAt"].date()
            incident_month = date(incident_month.year, incident_month.month, 1)
            category_id = incident["CategoryID"]
            category_monthly[category_id][incident_month] += 1
    
    # Build table data
    table_data = []
    for category_id, monthly_data in category_monthly.items():
        category_name = category_map.get(category_id, f"Unknown ({category_id})")
        monthly_counts = [category_monthly[category_id].get(m, 0) for m in months]
        total = sum(monthly_counts)
        
        trend_direction, trend_pct = _calculate_trend(monthly_counts)
        table_data.append({
            "id": category_id,
            "name": category_name,
            "total": total,
            "color": _generate_color_from_id(category_id),
            "trend_direction": trend_direction,
            "trend_percentage": trend_pct,
            "monthly": [
                {
                    "month": m.strftime("%Y-%m"),
                    "month_ar": f"{ARABIC_MONTHS[m.month]} {m.year}",
                    "count": category_monthly[category_id].get(m, 0)
                }
                for m in months
            ]
        })
    
    # Sort by total descending
    table_data.sort(key=lambda x: x["total"], reverse=True)
    
    return {
        "table": table_data,
        "chart_data": [
            {
                "name": item["name"],
                "data": [m["count"] for m in item["monthly"]],
                "color": item["color"]
            }
            for item in table_data
        ],
        "chart_labels": [m.strftime("%b %Y") for m in months]
    }


def _build_classification_trends(incidents, start_date, end_date, include_zero_months):
    """Build classification trends for scoped incidents"""
    classifications = lookups.get_classifications() or []
    classification_map = {c["ClassificationID"]: c.get("Classification_AR", c.get("Classification_EN", f"Classification {c['ClassificationID']}")) 
                         for c in classifications if c.get("ClassificationID")}
    
    # Generate months
    months = _generate_months(start_date, end_date, include_zero_months)
    
    # Group incidents by classification and month
    classification_monthly = defaultdict(lambda: defaultdict(int))
    for incident in incidents:
        if incident.get("ClassificationID"):
            incident_month = incident["CreatedAt"].date()
            incident_month = date(incident_month.year, incident_month.month, 1)
            classification_id = incident["ClassificationID"]
            classification_monthly[classification_id][incident_month] += 1
    
    # Build table data
    table_data = []
    for classification_id, monthly_data in classification_monthly.items():
        classification_name = classification_map.get(classification_id, f"Unknown ({classification_id})")
        monthly_counts = [classification_monthly[classification_id].get(m, 0) for m in months]
        total = sum(monthly_counts)
        
        trend_direction, trend_pct = _calculate_trend(monthly_counts)
        table_data.append({
            "id": classification_id,
            "name": classification_name,
            "total": total,
            "color": _generate_color_from_id(classification_id),
            "trend_direction": trend_direction,
            "trend_percentage": trend_pct,
            "monthly": [
                {
                    "month": m.strftime("%Y-%m"),
                    "month_ar": f"{ARABIC_MONTHS[m.month]} {m.year}",
                    "count": classification_monthly[classification_id].get(m, 0)
                }
                for m in months
            ]
        })
    
    # Sort by total descending
    table_data.sort(key=lambda x: x["total"], reverse=True)
    
    return {
        "table": table_data,
        "chart_data": [
            {
                "name": item["name"],
                "data": [m["count"] for m in item["monthly"]],
                "color": item["color"]
            }
            for item in table_data
        ],
        "chart_labels": [m.strftime("%b %Y") for m in months]
    }
