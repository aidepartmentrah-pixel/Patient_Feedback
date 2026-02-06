"""
Red Flags Service
Handles data retrieval and filtering for Red Flags (Critical Issues) page.
Red flags are high-risk incidents requiring immediate attention and governance follow-up.
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, date
import pyodbc
from dateutil.relativedelta import relativedelta

from ..db_layer.database import get_connection


# ==================== MAIN ENDPOINTS ====================

def get_red_flags_list(
    search: Optional[str] = None,
    status: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    department: Optional[str] = None,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    is_never_event: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Fetch list of red flag incidents with optional filtering.
    
    Red flags are critical incidents with ClinicalRiskTypeID = 2 (REDFLAG).
    
    Returns:
        Dictionary with red_flags array, total count, and pagination info.
    """
    
    # Build WHERE clause
    where_conditions = ["c.ClinicalRiskTypeID = 2"]  # Red flags only
    params = []
    
    # Free-text search
    if search:
        search_condition = """(
            CAST(c.IncidentRequestCaseID AS VARCHAR) LIKE ? 
            OR c.PatientName LIKE ?
        )"""
        where_conditions.append(search_condition)
        search_param = f"%{search}%"
        params.extend([search_param, search_param])
    
    # Status filter
    if status and status.upper() != "ALL":
        where_conditions.append("status.Name = ?")
        params.append(status.upper())
    
    # Date range filters
    if from_date:
        where_conditions.append("c.FeedbackRecievedDate >= ?")
        params.append(from_date)
    
    if to_date:
        where_conditions.append("c.FeedbackRecievedDate <= ?")
        params.append(to_date)
    
    # Department filter
    if department:
        where_conditions.append("org_unit.Name = ?")
        params.append(department)
    
    # Category filter (using Domain as category)
    if category:
        where_conditions.append("domain.DomainName = ?")
        params.append(category)
    
    # Severity filter
    if severity:
        where_conditions.append("severity.SeverityName = ?")
        params.append(severity)
    
    # Never Event overlap filter
    if is_never_event is not None:
        if is_never_event:
            where_conditions.append("c.ClinicalRiskTypeID = 3")  # Never Events
        else:
            where_conditions.append("c.ClinicalRiskTypeID != 3")
    
    where_clause = "WHERE " + " AND ".join(where_conditions)
    
    # Get total count
    count_query = f"""
        SELECT COUNT(*) as total
        FROM dbo.APP_IncidentCase c
        LEFT JOIN AdminsrationUnit org_unit ON c.IssuingOrgUnitID = org_unit.UniqueID
        LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
        LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
        LEFT JOIN APP_LOOKUP_CASE_STATUS status ON c.CaseStatusID = status.CaseStatusID
        {where_clause}
    """
    
    # Get paginated records
    list_query = f"""
        SELECT 
            c.IncidentRequestCaseID as id,
            CONCAT('RF-', YEAR(c.FeedbackRecievedDate), '-', 
                   RIGHT('000' + CAST(c.IncidentRequestCaseID AS VARCHAR), 3)) as case_id,
            CASE 
                WHEN c.ClinicalRiskTypeID = 2 THEN risk_type.Name
                ELSE 'High Risk Incident'
            END as title,
            c.ComplaintText as description,
            severity.SeverityName as severity,
            status.Name as status,
            org_unit.Name as department,
            domain.DomainName as category,
            c.FeedbackRecievedDate as date,
            CASE WHEN c.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END as is_never_event,
            c.CreatedByUser as assigned_to,
            c.CreatedAt as created_at,
            c.UpdatedAt as updated_at
        FROM dbo.APP_IncidentCase c
        LEFT JOIN AdminsrationUnit org_unit ON c.IssuingOrgUnitID = org_unit.UniqueID
        LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
        LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
        LEFT JOIN APP_LOOKUP_CASE_STATUS status ON c.CaseStatusID = status.CaseStatusID
        LEFT JOIN APP_LOOKUP_CLINICAL_RISK_TYPE risk_type ON c.ClinicalRiskTypeID = risk_type.ClinicalRiskTypeID
        {where_clause}
        ORDER BY c.FeedbackRecievedDate DESC
        OFFSET ? ROWS
        FETCH NEXT ? ROWS ONLY
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Get total count
        cursor.execute(count_query, params)
        total = cursor.fetchone().total
        
        # Get records
        cursor.execute(list_query, params + [offset, limit])
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        
        red_flags = []
        for row in rows:
            record = {}
            for idx, col_name in enumerate(columns):
                value = row[idx]
                if isinstance(value, (datetime, date)):
                    record[col_name] = value.strftime('%Y-%m-%d')
                elif col_name == 'is_never_event':
                    record[col_name] = bool(value)
                else:
                    record[col_name] = value if value is not None else ""
            red_flags.append(record)
        
        return {
            "red_flags": red_flags,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    finally:
        cursor.close()
        conn.close()


def get_red_flags_statistics(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch summary statistics for Red Flags KPI cards.
    
    Returns counts, breakdowns by status/category/severity, and Never Event overlap.
    """
    
    # Build date filter
    date_conditions = ["c.ClinicalRiskTypeID = 2"]  # Red flags only
    params = []
    
    if from_date:
        date_conditions.append("c.FeedbackRecievedDate >= ?")
        params.append(from_date)
    
    if to_date:
        date_conditions.append("c.FeedbackRecievedDate <= ?")
        params.append(to_date)
    
    date_filter = " AND ".join(date_conditions)
    
    query = f"""
        -- Total and status counts
        SELECT 
            COUNT(*) as total_red_flags,
            SUM(CASE WHEN status.Name != 'FINISHED' THEN 1 ELSE 0 END) as unfinished_count,
            SUM(CASE WHEN status.Name = 'FINISHED' THEN 1 ELSE 0 END) as finished_count,
            SUM(CASE WHEN status.Name = 'OPEN' THEN 1 ELSE 0 END) as open_count,
            SUM(CASE WHEN status.Name = 'UNDER_REVIEW' THEN 1 ELSE 0 END) as under_review_count,
            SUM(CASE WHEN c.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END) as never_event_overlap
        FROM dbo.APP_IncidentCase c
        LEFT JOIN APP_LOOKUP_CASE_STATUS status ON c.CaseStatusID = status.CaseStatusID
        WHERE {date_filter};
        
        -- By category (Domain)
        SELECT 
            domain.DomainName as category,
            COUNT(*) as count
        FROM dbo.APP_IncidentCase c
        LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
        WHERE {date_filter}
        GROUP BY domain.DomainName
        ORDER BY count DESC;
        
        -- By severity
        SELECT 
            severity.SeverityName as severity,
            COUNT(*) as count
        FROM dbo.APP_IncidentCase c
        LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
        WHERE {date_filter}
        GROUP BY severity.SeverityName;
        
        -- Current month
        SELECT COUNT(*) as count
        FROM dbo.APP_IncidentCase c
        WHERE {date_filter}
        AND YEAR(c.FeedbackRecievedDate) = YEAR(GETDATE())
        AND MONTH(c.FeedbackRecievedDate) = MONTH(GETDATE());
        
        -- Previous month
        SELECT COUNT(*) as count
        FROM dbo.APP_IncidentCase c
        WHERE {date_filter}
        AND YEAR(c.FeedbackRecievedDate) = YEAR(DATEADD(MONTH, -1, GETDATE()))
        AND MONTH(c.FeedbackRecievedDate) = MONTH(DATEADD(MONTH, -1, GETDATE()));
        
        -- Never Events total
        SELECT COUNT(*) as total_never_events
        FROM dbo.APP_IncidentCase
        WHERE ClinicalRiskTypeID = 3;
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Execute multi-result query
        cursor.execute(query, params * 6)
        
        # Result 1: Totals and status
        totals_row = cursor.fetchone()
        
        # Result 2: By category
        cursor.nextset()
        by_category = {}
        for row in cursor.fetchall():
            if row[0]:
                by_category[row[0]] = row[1]
        
        # Result 3: By severity
        cursor.nextset()
        by_severity = {}
        for row in cursor.fetchall():
            if row[0]:
                by_severity[row[0]] = row[1]
        
        # Result 4: Current month
        cursor.nextset()
        current_month_count = cursor.fetchone()[0]
        
        # Result 5: Previous month
        cursor.nextset()
        previous_month_count = cursor.fetchone()[0]
        
        # Result 6: Total Never Events
        cursor.nextset()
        total_never_events = cursor.fetchone()[0]
        
        # Calculate overlaps
        red_flags_also_never_events = totals_row.never_event_overlap
        never_events_only = total_never_events - red_flags_also_never_events
        red_flags_only = totals_row.total_red_flags - red_flags_also_never_events
        
        # Map to UI-expected field names
        return {
            "total": totals_row.total_red_flags,
            "open": totals_row.open_count,
            "in_progress": totals_row.under_review_count,
            "resolved": totals_row.finished_count,
            "high_severity": by_severity.get('HIGH', 0),
            "medium_severity": by_severity.get('MEDIUM', 0),
            "low_severity": by_severity.get('LOW', 0),
            "never_events_count": red_flags_also_never_events,
            "average_resolution_days": 0,  # TODO: Calculate from actual resolution times
            "from_date": from_date or "all-time",
            "to_date": to_date or datetime.now().strftime("%Y-%m-%d")
        }
        
    finally:
        cursor.close()
        conn.close()


def get_red_flags_trends(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    granularity: str = "monthly",
    group_by: str = "none"
) -> Dict[str, Any]:
    """
    Fetch time-series trend data for Red Flags.
    
    Args:
        from_date: Start date for trend
        to_date: End date for trend
        granularity: 'monthly', 'quarterly', 'weekly'
        group_by: 'category', 'severity', 'department', 'none'
    
    Returns:
        Time-series data for trend chart visualization.
    """
    
    # Default date range: last 12 months
    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")
    if not from_date:
        from_date = (datetime.now() - relativedelta(months=12)).strftime("%Y-%m-%d")
    
    # Build date format based on granularity
    if granularity == "monthly":
        date_format = "FORMAT(c.FeedbackRecievedDate, 'MMM yyyy')"
        date_group = "YEAR(c.FeedbackRecievedDate), MONTH(c.FeedbackRecievedDate)"
    elif granularity == "quarterly":
        date_format = "CONCAT('Q', DATEPART(QUARTER, c.FeedbackRecievedDate), ' ', YEAR(c.FeedbackRecievedDate))"
        date_group = "YEAR(c.FeedbackRecievedDate), DATEPART(QUARTER, c.FeedbackRecievedDate)"
    else:  # weekly
        date_format = "FORMAT(c.FeedbackRecievedDate, 'yyyy-MM-dd')"
        date_group = "DATEPART(YEAR, c.FeedbackRecievedDate), DATEPART(WEEK, c.FeedbackRecievedDate)"
    
    # Base query without grouping
    if group_by == "none":
        query = f"""
            SELECT 
                {date_format} as period,
                COUNT(*) as count
            FROM dbo.APP_IncidentCase c
            WHERE c.ClinicalRiskTypeID = 2
            AND c.FeedbackRecievedDate >= ?
            AND c.FeedbackRecievedDate <= ?
            GROUP BY {date_group}
            ORDER BY {date_group}
        """
    elif group_by == "category":
        query = f"""
            SELECT 
                {date_format} as period,
                domain.DomainName as category,
                COUNT(*) as count
            FROM dbo.APP_IncidentCase c
            LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
            WHERE c.ClinicalRiskTypeID = 2
            AND c.FeedbackRecievedDate >= ?
            AND c.FeedbackRecievedDate <= ?
            GROUP BY {date_group}, domain.DomainName
            ORDER BY {date_group}, domain.DomainName
        """
    elif group_by == "severity":
        query = f"""
            SELECT 
                {date_format} as period,
                severity.SeverityName as severity,
                COUNT(*) as count
            FROM dbo.APP_IncidentCase c
            LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
            WHERE c.ClinicalRiskTypeID = 2
            AND c.FeedbackRecievedDate >= ?
            AND c.FeedbackRecievedDate <= ?
            GROUP BY {date_group}, severity.SeverityName
            ORDER BY {date_group}, severity.SeverityName
        """
    elif group_by == "department":
        query = f"""
            SELECT 
                {date_format} as period,
                org_unit.Name as department,
                COUNT(*) as count
            FROM dbo.APP_IncidentCase c
            LEFT JOIN AdminsrationUnit org_unit ON c.IssuingOrgUnitID = org_unit.UniqueID
            WHERE c.ClinicalRiskTypeID = 2
            AND c.FeedbackRecievedDate >= ?
            AND c.FeedbackRecievedDate <= ?
            GROUP BY {date_group}, org_unit.Name
            ORDER BY {date_group}, org_unit.Name
        """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(query, [from_date, to_date])
        rows = cursor.fetchall()
        
        trends = []
        
        if group_by == "none":
            # Get severity breakdown for each period
            severity_query = f"""
                SELECT 
                    {date_format} as period,
                    COUNT(*) as total,
                    SUM(CASE WHEN severity.SeverityName = 'HIGH' THEN 1 ELSE 0 END) as high,
                    SUM(CASE WHEN severity.SeverityName = 'MEDIUM' THEN 1 ELSE 0 END) as medium,
                    SUM(CASE WHEN severity.SeverityName = 'LOW' THEN 1 ELSE 0 END) as low,
                    MIN(c.FeedbackRecievedDate) as date
                FROM dbo.APP_IncidentCase c
                LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
                WHERE c.ClinicalRiskTypeID = 2
                AND c.FeedbackRecievedDate >= ?
                AND c.FeedbackRecievedDate <= ?
                GROUP BY {date_group}
                ORDER BY {date_group}
            """
            cursor.execute(severity_query, [from_date, to_date])
            rows = cursor.fetchall()
            
            for row in rows:
                date_val = row[5] if len(row) > 5 else None
                trends.append({
                    "period": row[0],
                    "date": date_val.strftime('%Y-%m-%d') if date_val else row[0],
                    "total": row[1],
                    "high": row[2],
                    "medium": row[3],
                    "low": row[4]
                })
        else:
            # Group by period and category/severity/department
            for row in rows:
                period = row[0]
                group_name = row[1] if row[1] else "Unknown"
                count = row[2]
                
                trends.append({
                    "period": period,
                    "date": period,  # Use period as date for grouped data
                    "group_label": group_name,
                    "count": count
                })
        
        return {
            "trends": trends
        }
        
    finally:
        cursor.close()
        conn.close()


def get_red_flag_details(red_flag_id: int) -> Dict[str, Any]:
    """
    Fetch comprehensive details for a single red flag.
    
    Includes red flag data, incident details, timeline, and related actions.
    """
    
    query = """
        SELECT 
            c.IncidentRequestCaseID as id,
            CONCAT('RF-', YEAR(c.FeedbackRecievedDate), '-', 
                   RIGHT('000' + CAST(c.IncidentRequestCaseID AS VARCHAR), 3)) as case_id,
            CASE 
                WHEN c.ClinicalRiskTypeID = 2 THEN risk_type.Name
                ELSE 'High Risk Incident'
            END as title,
            c.ComplaintText as description,
            severity.SeverityName as severity,
            status.Name as status,
            org_unit.Name as department,
            org_unit.UniqueID as department_id,
            domain.DomainName as category,
            domain.DomainID as category_id,
            domain.SubDomainName as subcategory,
            c.FeedbackRecievedDate as date,
            c.ReportingPerson as reported_by,
            c.CreatedByUser as assigned_to,
            CASE WHEN c.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END as is_never_event,
            c.ImmediateAction as root_cause,
            c.TakenAction as corrective_action,
            c.CreatedAt as created_at,
            c.UpdatedAt as updated_at
            
        FROM dbo.APP_IncidentCase c
        LEFT JOIN AdminsrationUnit org_unit ON c.IssuingOrgUnitID = org_unit.UniqueID
        LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
        LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
        LEFT JOIN APP_LOOKUP_CASE_STATUS status ON c.CaseStatusID = status.CaseStatusID
        LEFT JOIN APP_LOOKUP_CLINICAL_RISK_TYPE risk_type ON c.ClinicalRiskTypeID = risk_type.ClinicalRiskTypeID
        
        WHERE c.IncidentRequestCaseID = ?
        AND c.ClinicalRiskTypeID = 2
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(query, [red_flag_id])
        row = cursor.fetchone()
        
        if not row:
            return None
        
        columns = [column[0] for column in cursor.description]
        
        # Build red flag object
        red_flag = {}
        
        for idx, col_name in enumerate(columns):
            value = row[idx]
            if isinstance(value, (datetime, date)):
                value = value.strftime('%Y-%m-%d')
            elif col_name == 'is_never_event':
                value = bool(value)
            red_flag[col_name] = value if value is not None else ""
        
        # Build timeline (in production, query audit log)
        timeline = [
            {
                "date": red_flag.get("created_at", ""),
                "action": "Reported",
                "user": red_flag.get("reported_by", "System"),
                "notes": "Red flag reported"
            }
        ]
        if red_flag.get("status") == "FINISHED":
            timeline.append({
                "date": red_flag.get("updated_at", ""),
                "action": "Resolved",
                "user": red_flag.get("assigned_to", "System"),
                "notes": "Red flag resolved"
            })
        
        # Add mock fields expected by UI
        red_flag["timeline"] = timeline
        red_flag["attachments"] = []  # TODO: Query attachments table
        red_flag["related_cases"] = []  # TODO: Query related cases
        
        return red_flag
        
    finally:
        cursor.close()
        conn.close()


# ==================== ANALYTICS ENDPOINTS ====================

def get_red_flags_category_breakdown(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch category breakdown for red flags analytics cards.
    
    Args:
        from_date: Filter from date (optional)
        to_date: Filter to date (optional)
    
    Returns:
        Dictionary with category breakdown including counts and percentages
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Build date filter
        date_filter = "c.ClinicalRiskTypeID = 2"
        params = []
        
        if from_date:
            date_filter += " AND c.FeedbackRecievedDate >= ?"
            params.append(from_date)
        if to_date:
            date_filter += " AND c.FeedbackRecievedDate <= ?"
            params.append(to_date)
        
        # Get total count
        total_query = f"""
        SELECT COUNT(*) as total
        FROM dbo.APP_IncidentCase c
        WHERE {date_filter}
        """
        
        cursor.execute(total_query, params)
        total = cursor.fetchone()[0]
        
        # Get category breakdown
        query = f"""
        SELECT 
            domain.DomainName as category,
            domain.DomainID as category_id,
            COUNT(*) as count
        FROM dbo.APP_IncidentCase c
        LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
        WHERE {date_filter}
        GROUP BY domain.DomainName, domain.DomainID
        ORDER BY count DESC
        """
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        breakdown = []
        for row in rows:
            percentage = (row[2] / total * 100) if total > 0 else 0
            breakdown.append({
                "category": row[0] if row[0] else "Unknown",
                "category_id": row[1] if row[1] else 0,
                "count": row[2],
                "percentage": round(percentage, 1)
            })
        
        return {
            "breakdown": breakdown,
            "total": total
        }
        
    finally:
        cursor.close()
        conn.close()


def get_red_flags_department_breakdown(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Fetch department breakdown for red flags analytics cards.
    
    Args:
        from_date: Filter from date (optional)
        to_date: Filter to date (optional)
        limit: Max number of departments to return (default: 10)
    
    Returns:
        Dictionary with department breakdown including counts and percentages
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Build date filter
        date_filter = "c.ClinicalRiskTypeID = 2"
        params = []
        
        if from_date:
            date_filter += " AND c.FeedbackRecievedDate >= ?"
            params.append(from_date)
        if to_date:
            date_filter += " AND c.FeedbackRecievedDate <= ?"
            params.append(to_date)
        
        # Get total count
        total_query = f"""
        SELECT COUNT(*) as total
        FROM dbo.APP_IncidentCase c
        WHERE {date_filter}
        """
        
        cursor.execute(total_query, params)
        total = cursor.fetchone()[0]
        
        # Get department breakdown
        query = f"""
        SELECT TOP {limit}
            org_unit.Name as department,
            org_unit.UniqueID as department_id,
            COUNT(*) as count
        FROM dbo.APP_IncidentCase c
        LEFT JOIN AdminsrationUnit org_unit ON c.IssuingOrgUnitID = org_unit.UniqueID
        WHERE {date_filter}
        GROUP BY org_unit.Name, org_unit.UniqueID
        ORDER BY count DESC
        """
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        breakdown = []
        for row in rows:
            percentage = (row[2] / total * 100) if total > 0 else 0
            breakdown.append({
                "department": row[0] if row[0] else "Unknown",
                "department_id": row[1] if row[1] else 0,
                "count": row[2],
                "percentage": round(percentage, 1)
            })
        
        return {
            "breakdown": breakdown,
            "total": total
        }
        
    finally:
        cursor.close()
        conn.close()
