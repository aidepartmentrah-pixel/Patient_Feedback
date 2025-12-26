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
                   RIGHT('000' + CAST(c.IncidentRequestCaseID AS VARCHAR), 3)) as recordID,
            c.FeedbackRecievedDate as date,
            c.PatientName as patientName,
            c.PatientName as patientID,
            CASE 
                WHEN c.ClinicalRiskTypeID = 2 THEN risk_type.Name
                ELSE 'High Risk Incident'
            END as redFlagType,
            CASE 
                WHEN c.ClinicalRiskTypeID = 2 THEN risk_type.Name
                ELSE N'حادثة عالية الخطورة'
            END as redFlagTypeAr,
            domain.DomainName as redFlagCategory,
            status.Name as status,
            severity.SeverityName as severity,
            org_unit.Name as department,
            org_unit.Name as qism,
            c.IncidentRequestCaseID as incidentID,
            CASE WHEN c.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END as isNeverEvent
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
                else:
                    record[col_name] = value
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
        
        return {
            "total_red_flags": totals_row.total_red_flags,
            "unfinished_count": totals_row.unfinished_count,
            "finished_count": totals_row.finished_count,
            "by_status": {
                "OPEN": totals_row.open_count,
                "UNDER_REVIEW": totals_row.under_review_count,
                "FINISHED": totals_row.finished_count
            },
            "by_category": by_category,
            "by_severity": by_severity,
            "current_month": {
                "count": current_month_count,
                "month": datetime.now().strftime("%B %Y")
            },
            "previous_month": {
                "count": previous_month_count,
                "month": (datetime.now() - relativedelta(months=1)).strftime("%B %Y")
            },
            "never_event_overlap": {
                "total_never_events": total_never_events,
                "red_flags_also_never_events": red_flags_also_never_events,
                "never_events_only": never_events_only,
                "red_flags_only": red_flags_only
            },
            "period": {
                "from": from_date or "all-time",
                "to": to_date or datetime.now().strftime("%Y-%m-%d")
            }
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
        
        if group_by == "none":
            data = [{"period": row[0], "count": row[1]} for row in rows]
        else:
            # Group by period
            period_data = {}
            for row in rows:
                period = row[0]
                group_name = row[1]
                count = row[2]
                
                if period not in period_data:
                    period_data[period] = {"period": period, "total": 0, "breakdown": {}}
                
                period_data[period]["breakdown"][group_name or "Unknown"] = count
                period_data[period]["total"] += count
            
            data = list(period_data.values())
        
        return {
            "granularity": granularity,
            "group_by": group_by if group_by != "none" else None,
            "period": {
                "from": from_date,
                "to": to_date
            },
            "data": data
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
            -- Red flag basic info
            c.IncidentRequestCaseID as id,
            CONCAT('RF-', YEAR(c.FeedbackRecievedDate), '-', 
                   RIGHT('000' + CAST(c.IncidentRequestCaseID AS VARCHAR), 3)) as recordID,
            c.FeedbackRecievedDate as date,
            c.PatientName as patientName,
            c.PatientName as patientID,
            risk_type.Name as redFlagType,
            risk_type.Name as redFlagTypeAr,
            domain.DomainName as redFlagCategory,
            status.Name as status,
            severity.SeverityName as severity,
            org_unit.Name as department,
            org_unit.Name as qism,
            c.IncidentRequestCaseID as incidentID,
            CASE WHEN c.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END as isNeverEvent,
            
            -- Incident details
            c.ComplaintText,
            c.ImmediateAction,
            c.TakenAction,
            c.FeedbackRecievedDate as feedbackReceivedDate,
            harm.HarmLevel as harmLevel,
            stage.StageName as stage,
            c.CreatedAt as createdAt
            
        FROM dbo.APP_IncidentCase c
        LEFT JOIN AdminsrationUnit org_unit ON c.IssuingOrgUnitID = org_unit.UniqueID
        LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
        LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
        LEFT JOIN APP_LOOKUP_CASE_STATUS status ON c.CaseStatusID = status.CaseStatusID
        LEFT JOIN APP_LOOKUP_CLINICAL_RISK_TYPE risk_type ON c.ClinicalRiskTypeID = risk_type.ClinicalRiskTypeID
        LEFT JOIN APP_LOOKUP_HARM_LEVEL harm ON c.HarmLevelID = harm.HarmID
        LEFT JOIN APP_LOOKUP_CASE_STAGE stage ON c.StageID = stage.StageID
        
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
        incident_details = {}
        
        red_flag_fields = ["id", "recordID", "date", "patientName", "patientID", "redFlagType", 
                          "redFlagTypeAr", "redFlagCategory", "status", "severity", "department", 
                          "qism", "incidentID", "isNeverEvent"]
        
        for idx, col_name in enumerate(columns):
            value = row[idx]
            if isinstance(value, (datetime, date)):
                value = value.strftime('%Y-%m-%d')
            
            if col_name in red_flag_fields:
                red_flag[col_name] = value
            else:
                incident_details[col_name] = value
        
        # Mock timeline (in production, query audit log)
        timeline = [
            {
                "date": red_flag.get("date", ""),
                "event": "Red flag reported",
                "user": "System"
            }
        ]
        
        # Mock related actions (in production, query actions table)
        related_actions = []
        
        return {
            "red_flag": red_flag,
            "incident_details": incident_details,
            "timeline": timeline,
            "related_actions": related_actions
        }
        
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
        Dictionary with category breakdown including severity distribution
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Build date filter
        date_filter = f"ClinicalRiskTypeID = {REDFLAG}"
        period_display = "All time"
        
        if from_date:
            date_filter += f" AND FeedbackReceivedDate >= '{from_date}'"
            period_display = f"{from_date} to "
        if to_date:
            date_filter += f" AND FeedbackReceivedDate <= '{to_date}'"
            if from_date:
                period_display += to_date
            else:
                period_display = f"Until {to_date}"
        elif from_date:
            period_display += datetime.now().strftime('%Y-%m-%d')
        
        # Get total count
        total_query = f"""
        SELECT COUNT(*) as total
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS
        WHERE {date_filter}
        """
        
        cursor.execute(total_query)
        total = cursor.fetchone()[0]
        
        # Get category breakdown with severity
        query = f"""
        SELECT 
            ISNULL(d.NameEN, 'Unknown') as category_name,
            ISNULL(d.NameAR, 'غير محدد') as category_name_ar,
            COUNT(*) as count,
            SUM(CASE WHEN s.NameEN = 'CRITICAL' THEN 1 ELSE 0 END) as critical_count,
            SUM(CASE WHEN s.NameEN = 'HIGH' THEN 1 ELSE 0 END) as high_count
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS tbl
        LEFT JOIN IncidentManager.dbo.APP_LOOKUP_DOMAIN d ON tbl.Domain = d.ID
        LEFT JOIN IncidentManager.dbo.APP_LOOKUP_SEVERITY s ON tbl.SeverityLevel = s.ID
        WHERE {date_filter}
        GROUP BY d.NameEN, d.NameAR
        ORDER BY count DESC
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        categories = []
        for row in rows:
            percentage = (row.count / total * 100) if total > 0 else 0
            categories.append({
                "category_name": row.category_name,
                "category_name_ar": row.category_name_ar,
                "count": row.count,
                "percentage": round(percentage, 1),
                "severity_breakdown": {
                    "CRITICAL": row.critical_count,
                    "HIGH": row.high_count
                }
            })
        
        return {
            "total": total,
            "period": period_display,
            "categories": categories
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
        Dictionary with department breakdown including status distribution
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Build date filter
        date_filter = f"tbl.ClinicalRiskTypeID = {REDFLAG}"
        period_display = "All time"
        
        if from_date:
            date_filter += f" AND tbl.FeedbackReceivedDate >= '{from_date}'"
            period_display = f"{from_date} to "
        if to_date:
            date_filter += f" AND tbl.FeedbackReceivedDate <= '{to_date}'"
            if from_date:
                period_display += to_date
            else:
                period_display = f"Until {to_date}"
        elif from_date:
            period_display += datetime.now().strftime('%Y-%m-%d')
        
        # Get total count
        total_query = f"""
        SELECT COUNT(*) as total
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS tbl
        WHERE {date_filter}
        """
        
        cursor.execute(total_query)
        total = cursor.fetchone()[0]
        
        # Get department breakdown with status
        query = f"""
        SELECT TOP {limit}
            ISNULL(dept.NameAR, 'غير محدد') as department,
            ISNULL(dept.NameEN, 'Unknown') as department_en,
            COUNT(*) as count,
            SUM(CASE WHEN tbl.CaseStatusID = 1 THEN 1 ELSE 0 END) as open_count,
            SUM(CASE WHEN tbl.CaseStatusID = 2 THEN 1 ELSE 0 END) as under_review_count,
            SUM(CASE WHEN tbl.CaseStatusID = 3 THEN 1 ELSE 0 END) as finished_count
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS tbl
        LEFT JOIN IncidentManager.dbo.AdministrationUnit dept 
            ON tbl.IssuerDepartment = dept.AdministrationUnit_ID
        WHERE {date_filter}
        GROUP BY dept.NameAR, dept.NameEN
        ORDER BY count DESC
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        departments = []
        for row in rows:
            percentage = (row.count / total * 100) if total > 0 else 0
            departments.append({
                "department": row.department,
                "department_en": row.department_en,
                "count": row.count,
                "percentage": round(percentage, 1),
                "status_breakdown": {
                    "OPEN": row.open_count,
                    "UNDER_REVIEW": row.under_review_count,
                    "FINISHED": row.finished_count
                }
            })
        
        return {
            "total": total,
            "period": period_display,
            "departments": departments
        }
        
    finally:
        cursor.close()
        conn.close()
