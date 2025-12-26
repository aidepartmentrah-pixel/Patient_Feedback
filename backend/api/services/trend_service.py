from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import defaultdict, Counter
from typing import Literal
import calendar
from ..db_layer.database import get_connection
from ..db_layer import lookups


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

def get_domain_trends(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    include_zero_months: bool = True,
    include_inactive_domains: bool = False,
    calculate_trends: bool = True,
) -> dict:
    """
    Get monthly incident trends aggregated by domain.
    
    Args:
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
    # Fetch raw incident data
    # -------------------------
    raw_data = _fetch_incidents_by_domain_and_month(start_date, end_date)
    
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
    
    Args:
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
    # Fetch raw incident data
    # -------------------------
    raw_data = _fetch_incidents_by_category_and_month(start_date, end_date, domain_id)
    
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
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict:
    """
    Get list of available time periods (months) with incident data.
    
    Args:
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
    # Fetch incident counts per month
    # -------------------------
    raw_data = _fetch_incidents_per_month(start_date, end_date)
    
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
    
    return {
        "periods": periods_result,
        "total_periods": len(periods_result),
        "earliest_period": periods_result[0]["period"] if periods_result else None,
        "latest_period": periods_result[-1]["period"] if periods_result else None,
    }


# =========================================================
# PRIVATE HELPER FUNCTIONS
# =========================================================

def _fetch_incidents_by_domain_and_month(start_date: date, end_date: date) -> list[dict]:
    """
    Fetch raw incident counts grouped by domain and month.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
    SELECT 
        DomainID,
        FORMAT(FeedbackRecievedDate, 'yyyy-MM') AS Period,
        COUNT(*) AS IncidentCount
    FROM dbo.APP_IncidentCase
    WHERE FeedbackRecievedDate >= ?
      AND FeedbackRecievedDate < DATEADD(MONTH, 1, ?)
      AND DomainID IS NOT NULL
    GROUP BY DomainID, FORMAT(FeedbackRecievedDate, 'yyyy-MM')
    ORDER BY DomainID, Period
    """
    
    cursor.execute(query, start_date, end_date)
    
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    
    conn.close()
    
    return [dict(zip(columns, row)) for row in rows]


def _fetch_incidents_by_category_and_month(
    start_date: date,
    end_date: date,
    domain_id: int | None = None
) -> list[dict]:
    """
    Fetch raw incident counts grouped by category and month.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
    SELECT 
        CategoryID,
        FORMAT(FeedbackRecievedDate, 'yyyy-MM') AS Period,
        COUNT(*) AS IncidentCount
    FROM dbo.APP_IncidentCase
    WHERE FeedbackRecievedDate >= ?
      AND FeedbackRecievedDate < DATEADD(MONTH, 1, ?)
      AND CategoryID IS NOT NULL
    """
    
    params = [start_date, end_date]
    
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


def _fetch_incidents_per_month(start_date: date, end_date: date) -> list[dict]:
    """
    Fetch total incident counts per month.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
    SELECT 
        FORMAT(FeedbackRecievedDate, 'yyyy-MM') AS Period,
        COUNT(*) AS IncidentCount
    FROM dbo.APP_IncidentCase
    WHERE FeedbackRecievedDate >= ?
      AND FeedbackRecievedDate < DATEADD(MONTH, 1, ?)
    GROUP BY FORMAT(FeedbackRecievedDate, 'yyyy-MM')
    ORDER BY Period
    """
    
    cursor.execute(query, start_date, end_date)
    
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
