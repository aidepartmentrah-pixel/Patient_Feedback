"""
Red Flags Service
Handles data retrieval and filtering for Red Flags (Critical Issues) page.
Red flags are high-risk incidents requiring immediate attention and governance follow-up.
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, date
import pyodbc
from dateutil.relativedelta import relativedelta

from core.database import get_connection


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
    sort_by: str = "date",
    sort_order: str = "desc",
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
    
    # Determine sort column and order
    sort_column_map = {
        "date": "c.FeedbackRecievedDate",
        "severity": "severity.SeverityName",
        "department": "org_unit.Name",
        "status": "status.Name",
        "created_at": "c.CreatedAt",
        "patient_name": "c.PatientName"
    }
    
    sort_column = sort_column_map.get(sort_by, "c.FeedbackRecievedDate")
    sort_direction = "DESC" if sort_order.upper() == "DESC" else "ASC"
    
    # Get paginated records
    list_query = f"""
        SELECT 
            c.IncidentRequestCaseID as id,
            CONCAT('RF-', YEAR(c.FeedbackRecievedDate), '-', 
                   RIGHT('000' + CAST(c.IncidentRequestCaseID AS VARCHAR), 3)) as record_id,
            CONCAT('RF-', YEAR(c.FeedbackRecievedDate), '-', 
                   RIGHT('000' + CAST(c.IncidentRequestCaseID AS VARCHAR), 3)) as case_id,
            
            -- Patient Information
            c.PatientName as patient_full_name,
            c.PatientName as patient_name,
            c.isINPatient as is_in_patient,
            
            -- Date Information
            c.FeedbackRecievedDate as feedback_received_date,
            c.CreatedAt as created_at,
            c.FeedbackRecievedDate as date,
            
            -- Department Information
            org_unit.Name as issuing_department,
            org_unit.Name as department,
            NULL as target_department,
            building.BuildingName as building,
            domain.DomainName as domain,
            
            -- Classification & Risk
            classification.Classification_AR as classification,
            domain.DomainName as category,
            subcategory.SubCategoryName as sub_category,
            risk_type.Name as clinical_risk_type,
            intent.NameEn as feedback_intent_type,
            
            -- Severity & Status
            UPPER(severity.SeverityName) as severity,
            harm.HarmLevel as harm_level,
            stage.StageName as stage,
            status.Name as case_status,
            status.Name as status,
            
            -- Never Event Flag
            CASE WHEN c.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END as is_never_event,
            
            -- Complaint Details
            c.ComplaintText as complaint_text,
            CASE 
                WHEN LEN(c.ComplaintText) > 100 THEN LEFT(c.ComplaintText, 100) + '...'
                ELSE c.ComplaintText
            END as complaint_summary,
            c.ImmediateAction as immediate_action,
            c.TakenAction as taken_action,
            
            -- Source & Assignment
            source.SourceName as source,
            NULL as created_by,
            c.CreatedByUserID as assigned_to,
            NULL as assigned_to_name
            
        FROM dbo.APP_IncidentCase c
        LEFT JOIN AdminsrationUnit org_unit ON c.IssuingOrgUnitID = org_unit.UniqueID
        LEFT JOIN APP_LOOKUP_BUILDING building ON c.BuildingID = building.BuildingID
        LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
        LEFT JOIN APP_LOOKUP_CLASSIFICATION classification ON c.ClassificationID = classification.ClassificationID
        LEFT JOIN APP_LOOKUP_CATEGORY category ON c.CategoryID = category.CategoryID
        LEFT JOIN APP_LOOKUP_SUBCATEGORY subcategory ON c.SubCategoryID = subcategory.SubCategoryID
        LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
        LEFT JOIN APP_LOOKUP_HARM_LEVEL harm ON c.HarmLevelID = harm.HarmID
        LEFT JOIN APP_LOOKUP_CASE_STAGE stage ON c.StageID = stage.StageID
        LEFT JOIN APP_LOOKUP_CASE_STATUS status ON c.CaseStatusID = status.CaseStatusID
        LEFT JOIN APP_LOOKUP_CLINICAL_RISK_TYPE risk_type ON c.ClinicalRiskTypeID = risk_type.ClinicalRiskTypeID
        LEFT JOIN APP_LOOKUP_FEEDBACK_INTENT_TYPE intent ON c.FeedbackIntentTypeID = intent.FeedbackIntentTypeID
        LEFT JOIN APP_LOOKUP_SOURCE source ON c.SourceID = source.SourceID
        {where_clause}
        ORDER BY {sort_column} {sort_direction}
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
            SUM(CASE WHEN status.Name IN ('OPEN', 'UNDER_REVIEW', 'IN_PROGRESS') THEN 1 ELSE 0 END) as unfinished_count,
            SUM(CASE WHEN status.Name IN ('FINISHED', 'RESOLVED', 'CLOSED') THEN 1 ELSE 0 END) as finished_count,
            SUM(CASE WHEN status.Name = 'OPEN' THEN 1 ELSE 0 END) as open_count,
            SUM(CASE WHEN status.Name IN ('UNDER_REVIEW', 'IN_PROGRESS') THEN 1 ELSE 0 END) as in_progress_count,
            SUM(CASE WHEN status.Name = 'RESOLVED' THEN 1 ELSE 0 END) as resolved_count,
            SUM(CASE WHEN status.Name IN ('CLOSED', 'FINISHED') THEN 1 ELSE 0 END) as closed_count,
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
            UPPER(severity.SeverityName) as severity,
            COUNT(*) as count
        FROM dbo.APP_IncidentCase c
        LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
        WHERE {date_filter}
        GROUP BY UPPER(severity.SeverityName);
        
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
        
        # Get current month name
        current_date = datetime.now()
        current_month_name = current_date.strftime("%B %Y")
        previous_date = current_date - relativedelta(months=1)
        previous_month_name = previous_date.strftime("%B %Y")
        
        # Map to UI-expected field names with nested structure
        return {
            "total_red_flags": totals_row.total_red_flags,
            "unfinished": totals_row.unfinished_count,
            "finished": totals_row.finished_count,
            
            "by_severity": {
                "CRITICAL": by_severity.get('CRITICAL', 0),
                "HIGH": by_severity.get('HIGH', 0),
                "MEDIUM": by_severity.get('MEDIUM', 0),
                "LOW": by_severity.get('LOW', 0)
            },
            
            "by_status": {
                "OPEN": totals_row.open_count,
                "IN_PROGRESS": totals_row.in_progress_count,
                "RESOLVED": totals_row.resolved_count,
                "CLOSED": totals_row.closed_count
            },
            
            "current_month": {
                "count": current_month_count,
                "month": current_month_name,
                "start_date": current_date.strftime("%Y-%m-01"),
                "end_date": (current_date.replace(day=1) + relativedelta(months=1) - relativedelta(days=1)).strftime("%Y-%m-%d")
            },
            
            "previous_month": {
                "count": previous_month_count,
                "month": previous_month_name
            },
            
            "never_event_overlap": {
                "total_never_events": total_never_events,
                "red_flags_also_never_events": red_flags_also_never_events,
                "never_events_only": never_events_only,
                "red_flags_only": red_flags_only
            },
            
            "average_resolution_days": 12.5,  # TODO: Calculate from actual resolution times
            
            "period": {
                "from_date": from_date or "2025-01-01",
                "to_date": to_date or datetime.now().strftime("%Y-%m-%d")
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
    
    # Build date selection and grouping based on granularity
    if granularity == "monthly":
        date_select = "YEAR(c.FeedbackRecievedDate) as year, MONTH(c.FeedbackRecievedDate) as month"
        date_group = "YEAR(c.FeedbackRecievedDate), MONTH(c.FeedbackRecievedDate)"
        date_order = "YEAR(c.FeedbackRecievedDate), MONTH(c.FeedbackRecievedDate)"
    elif granularity == "quarterly":
        date_select = "YEAR(c.FeedbackRecievedDate) as year, DATEPART(QUARTER, c.FeedbackRecievedDate) as quarter"
        date_group = "YEAR(c.FeedbackRecievedDate), DATEPART(QUARTER, c.FeedbackRecievedDate)"
        date_order = "YEAR(c.FeedbackRecievedDate), DATEPART(QUARTER, c.FeedbackRecievedDate)"
    else:  # weekly
        date_select = "YEAR(c.FeedbackRecievedDate) as year, DATEPART(WEEK, c.FeedbackRecievedDate) as week, MIN(c.FeedbackRecievedDate) as week_start"
        date_group = "YEAR(c.FeedbackRecievedDate), DATEPART(WEEK, c.FeedbackRecievedDate)"
        date_order = "YEAR(c.FeedbackRecievedDate), DATEPART(WEEK, c.FeedbackRecievedDate)"
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        trends = []
        
        if group_by == "none":
            # Get severity breakdown for each period
            query = f"""
                SELECT 
                    {date_select},
                    COUNT(*) as total,
                    SUM(CASE WHEN severity.SeverityName = 'HIGH' THEN 1 ELSE 0 END) as high,
                    SUM(CASE WHEN severity.SeverityName = 'MEDIUM' THEN 1 ELSE 0 END) as medium,
                    SUM(CASE WHEN severity.SeverityName = 'LOW' THEN 1 ELSE 0 END) as low
                FROM dbo.APP_IncidentCase c
                LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
                WHERE c.ClinicalRiskTypeID = 2
                AND c.FeedbackRecievedDate >= ?
                AND c.FeedbackRecievedDate <= ?
                GROUP BY {date_group}
                ORDER BY {date_order}
            """
            cursor.execute(query, [from_date, to_date])
            rows = cursor.fetchall()
            
            for row in rows:
                # Format period in Python based on granularity
                if granularity == "monthly":
                    period_date = datetime(row.year, row.month, 1)
                    period = period_date.strftime('%b %Y')  # e.g., "Feb 2026"
                    date_str = period_date.strftime('%Y-%m-%d')
                    total_idx, high_idx, med_idx, low_idx = 2, 3, 4, 5
                elif granularity == "quarterly":
                    period = f"Q{row.quarter} {row.year}"
                    period_date = datetime(row.year, (row.quarter - 1) * 3 + 1, 1)
                    date_str = period_date.strftime('%Y-%m-%d')
                    total_idx, high_idx, med_idx, low_idx = 2, 3, 4, 5
                else:  # weekly
                    period = row.week_start.strftime('%Y-%m-%d')
                    date_str = period
                    total_idx, high_idx, med_idx, low_idx = 3, 4, 5, 6
                
                trends.append({
                    "period": period,
                    "date": date_str,
                    "total": row[total_idx],
                    "high": row[high_idx],
                    "medium": row[med_idx],
                    "low": row[low_idx]
                })
        elif group_by == "category":
            query = f"""
                SELECT 
                    {date_select},
                    domain.DomainName as category,
                    COUNT(*) as count
                FROM dbo.APP_IncidentCase c
                LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
                WHERE c.ClinicalRiskTypeID = 2
                AND c.FeedbackRecievedDate >= ?
                AND c.FeedbackRecievedDate <= ?
                GROUP BY {date_group}, domain.DomainName
                ORDER BY {date_order}, domain.DomainName
            """
            cursor.execute(query, [from_date, to_date])
            rows = cursor.fetchall()
            
            for row in rows:
                # Format period in Python
                if granularity == "monthly":
                    period_date = datetime(row.year, row.month, 1)
                    period = period_date.strftime('%b %Y')
                    group_name = row[2] if row[2] else "Unknown"
                    count = row[3]
                elif granularity == "quarterly":
                    period = f"Q{row.quarter} {row.year}"
                    group_name = row[2] if row[2] else "Unknown"
                    count = row[3]
                else:  # weekly
                    period = row.week_start.strftime('%Y-%m-%d')
                    group_name = row[3] if row[3] else "Unknown"
                    count = row[4]
                
                trends.append({
                    "period": period,
                    "date": period,
                    "group_label": group_name,
                    "count": count
                })
        elif group_by == "severity":
            query = f"""
                SELECT 
                    {date_select},
                    severity.SeverityName as severity,
                    COUNT(*) as count
                FROM dbo.APP_IncidentCase c
                LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
                WHERE c.ClinicalRiskTypeID = 2
                AND c.FeedbackRecievedDate >= ?
                AND c.FeedbackRecievedDate <= ?
                GROUP BY {date_group}, severity.SeverityName
                ORDER BY {date_order}, severity.SeverityName
            """
            cursor.execute(query, [from_date, to_date])
            rows = cursor.fetchall()
            
            for row in rows:
                # Format period in Python
                if granularity == "monthly":
                    period_date = datetime(row.year, row.month, 1)
                    period = period_date.strftime('%b %Y')
                    group_name = row[2] if row[2] else "Unknown"
                    count = row[3]
                elif granularity == "quarterly":
                    period = f"Q{row.quarter} {row.year}"
                    group_name = row[2] if row[2] else "Unknown"
                    count = row[3]
                else:  # weekly
                    period = row.week_start.strftime('%Y-%m-%d')
                    group_name = row[3] if row[3] else "Unknown"
                    count = row[4]
                
                trends.append({
                    "period": period,
                    "date": period,
                    "group_label": group_name,
                    "count": count
                })
        elif group_by == "department":
            query = f"""
                SELECT 
                    {date_select},
                    org_unit.Name as department,
                    COUNT(*) as count
                FROM dbo.APP_IncidentCase c
                LEFT JOIN AdminsrationUnit org_unit ON c.IssuingOrgUnitID = org_unit.UniqueID
                WHERE c.ClinicalRiskTypeID = 2
                AND c.FeedbackRecievedDate >= ?
                AND c.FeedbackRecievedDate <= ?
                GROUP BY {date_group}, org_unit.Name
                ORDER BY {date_order}, org_unit.Name
            """
            cursor.execute(query, [from_date, to_date])
            rows = cursor.fetchall()
            
            for row in rows:
                # Format period in Python
                if granularity == "monthly":
                    period_date = datetime(row.year, row.month, 1)
                    period = period_date.strftime('%b %Y')
                    group_name = row[2] if row[2] else "Unknown"
                    count = row[3]
                elif granularity == "quarterly":
                    period = f"Q{row.quarter} {row.year}"
                    group_name = row[2] if row[2] else "Unknown"
                    count = row[3]
                else:  # weekly
                    period = row.week_start.strftime('%Y-%m-%d')
                    group_name = row[3] if row[3] else "Unknown"
                    count = row[4]
                
                trends.append({
                    "period": period,
                    "date": period,
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
            NULL as subcategory,
            c.FeedbackRecievedDate as date,
            NULL as reported_by,
            c.CreatedByUserID as assigned_to,
            CASE WHEN c.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END as is_never_event,
            c.ImmediateAction as root_cause,
            c.TakenAction as corrective_action,
            c.CreatedAt as created_at,
            NULL as updated_at
            
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
