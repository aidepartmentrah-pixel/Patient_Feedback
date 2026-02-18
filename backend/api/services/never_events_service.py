"""
Never Events Service
Handles data retrieval and filtering for Never Events page.
Never events are zero-tolerance incidents requiring immediate reporting and investigation.
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, date
import pyodbc
from dateutil.relativedelta import relativedelta

from core.database import get_connection


# Constants
NEVER_EVENT = 3  # ClinicalRiskTypeID for Never Events


# ==================== MAIN ENDPOINTS ====================

def get_never_events_list(
    search: Optional[str] = None,
    status: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    department: Optional[str] = None,
    category: Optional[str] = None,
    sort_by: str = "date",
    sort_order: str = "desc",
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Fetch list of never events with optional filtering and search.
    
    Never events are queried from APP_IncidentCase where ClinicalRiskTypeID = 3.
    
    Args:
        search: Search by case ID or patient name
        status: Filter by OPEN, UNDER_REVIEW, FINISHED, or "all"
        from_date: Filter from date (YYYY-MM-DD)
        to_date: Filter to date (YYYY-MM-DD)
        department: Filter by department
        category: Filter by never event category
        limit: Max results per page (default: 100)
        offset: Pagination offset (default: 0)
    
    Returns:
        Dictionary with never_events array, total count, limit, offset
    """
    
    # Build WHERE clause
    where_conditions = [f"c.ClinicalRiskTypeID = {NEVER_EVENT}"]
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
        "incident_date": "c.FeedbackRecievedDate",
        "severity": "severity.SeverityName",
        "department": "org_unit.Name",
        "status": "status.Name",
        "created_at": "c.CreatedAt",
        "patient_name": "c.PatientName",
        "category": "domain.DomainName"
    }
    
    sort_column = sort_column_map.get(sort_by, "c.FeedbackRecievedDate")
    sort_direction = "DESC" if sort_order.upper() == "DESC" else "ASC"
    
    # Get paginated records with full details
    list_query = f"""
        SELECT 
            c.IncidentRequestCaseID as id,
            CONCAT('NE-', YEAR(c.FeedbackRecievedDate), '-', 
                   RIGHT('000' + CAST(c.IncidentRequestCaseID AS VARCHAR), 3)) as record_id,
            CONCAT('NE-', YEAR(c.FeedbackRecievedDate), '-', 
                   RIGHT('000' + CAST(c.IncidentRequestCaseID AS VARCHAR), 3)) as case_id,
            CONCAT('NE-', YEAR(c.FeedbackRecievedDate), '-', 
                   RIGHT('000' + CAST(c.IncidentRequestCaseID AS VARCHAR), 3)) as never_event_id,
            
            -- Patient Information
            c.PatientName as patient_full_name,
            c.PatientName as patient_name,
            NULL as patient_id,
            c.isINPatient as is_in_patient,
            CASE WHEN c.isINPatient = 1 THEN 'Inpatient' ELSE 'Outpatient' END as patient_type,
            
            -- Date Information
            c.FeedbackRecievedDate as incident_date,
            c.FeedbackRecievedDate as reported_date,
            c.FeedbackRecievedDate as feedback_received_date,
            c.CreatedAt as created_at,
            c.FeedbackRecievedDate as date,
            
            -- Department Information
            org_unit.Name as department,
            org_unit.Name as issuing_department,
            NULL as target_department,
            NULL as section,
            NULL as sub_section,
            building.BuildingName as building,
            
            -- Never Event Classification
            risk_type.Name as never_event_type,
            NULL as never_event_type_ar,
            domain.DomainName as never_event_category,
            NULL as never_event_category_ar,
            subcategory.SubCategoryName as subcategory,
            
            -- Severity & Status
            UPPER(severity.SeverityName) as severity,
            harm.HarmLevel as harm_level,
            NULL as actual_harm,
            NULL as potential_harm,
            status.Name as status,
            status.Name as case_status,
            status.Name as investigation_status,
            stage.StageName as stage,
            
            -- Complaint Details
            c.ComplaintText as complaint_text,
            c.ComplaintText as incident_description,
            CASE 
                WHEN LEN(c.ComplaintText) > 100 THEN LEFT(c.ComplaintText, 100) + '...'
                ELSE c.ComplaintText
            END as complaint_summary,
            
            -- Immediate Response
            c.ImmediateAction as immediate_action,
            c.ImmediateAction as immediate_action_taken,
            c.TakenAction as taken_action,
            
            -- Investigation Details
            NULL as root_cause,
            NULL as root_cause_analysis,
            NULL as contributing_factors,
            NULL as corrective_actions,
            NULL as preventive_actions,
            
            -- Incident Tracking
            CONCAT('INC-', YEAR(c.FeedbackRecievedDate), '-', 
                   RIGHT('000' + CAST(c.IncidentRequestCaseID AS VARCHAR), 3)) as incident_number,
            CONCAT('INV-', YEAR(c.FeedbackRecievedDate), '-', 
                   RIGHT('000' + CAST(c.IncidentRequestCaseID AS VARCHAR), 3)) as investigation_number,
            0 as rca_completed,
            NULL as rca_completion_date,
            0 as reported_to_authorities,
            NULL as authority_report_date,
            
            -- Resolution
            NULL as resolution_date,
            NULL as closure_date,
            NULL as days_to_resolve,
            0 as resolved,
            
            -- People Involved
            NULL as reported_by,
            NULL as reported_by_id,
            NULL as assigned_to,
            c.CreatedByUserID as assigned_to_id,
            NULL as investigated_by,
            NULL as created_by,
            c.CreatedByUserID as created_by_id,
            
            -- Additional Context
            source.SourceName as source,
            intent.NameEn as feedback_intent_type,
            domain.DomainName as domain,
            classification.Classification_AR as classification,
            domain.DomainName as category,
            
            -- Risk Assessment
            NULL as likelihood,
            NULL as consequence,
            NULL as risk_score,
            NULL as residual_risk
            
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
        
        never_events = []
        for row in rows:
            record = {}
            for idx, col_name in enumerate(columns):
                value = row[idx]
                if isinstance(value, (datetime, date)):
                    record[col_name] = value.strftime('%Y-%m-%d')
                elif col_name in ['is_in_patient', 'rca_completed', 'reported_to_authorities', 'resolved']:
                    record[col_name] = bool(value) if value is not None else False
                else:
                    record[col_name] = value if value is not None else ""
            never_events.append(record)
        
        return {
            "never_events": never_events,
            "total": total,
            "limit": limit,
            "offset": offset,
            "goal": 0,
            "message": "Target: Zero Never Events"
        }
        
    finally:
        cursor.close()
        conn.close()


def get_never_events_statistics(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch summary statistics for Never Events KPI cards.
    
    Returns counts, breakdowns by status/category/severity.
    """
    
    # Build date filter
    date_conditions = [f"c.ClinicalRiskTypeID = {NEVER_EVENT}"]
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
            COUNT(*) as total_never_events,
            SUM(CASE WHEN status.Name IN ('OPEN', 'UNDER_REVIEW', 'IN_PROGRESS', 'UNDER_INVESTIGATION', 'RCA_IN_PROGRESS', 'PENDING_REVIEW') THEN 1 ELSE 0 END) as unfinished_count,
            SUM(CASE WHEN status.Name IN ('FINISHED', 'RESOLVED', 'CLOSED') THEN 1 ELSE 0 END) as finished_count,
            SUM(CASE WHEN status.Name = 'OPEN' THEN 1 ELSE 0 END) as open_count,
            SUM(CASE WHEN status.Name IN ('UNDER_REVIEW', 'UNDER_INVESTIGATION') THEN 1 ELSE 0 END) as under_investigation_count,
            SUM(CASE WHEN status.Name = 'RCA_IN_PROGRESS' THEN 1 ELSE 0 END) as rca_in_progress_count,
            SUM(CASE WHEN status.Name = 'PENDING_REVIEW' THEN 1 ELSE 0 END) as pending_review_count,
            SUM(CASE WHEN status.Name = 'RESOLVED' THEN 1 ELSE 0 END) as resolved_count,
            SUM(CASE WHEN status.Name = 'CLOSED' THEN 1 ELSE 0 END) as closed_count
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
        
        -- By harm level
        SELECT 
            harm.HarmLevel as harm_level,
            COUNT(*) as count
        FROM dbo.APP_IncidentCase c
        LEFT JOIN APP_LOOKUP_HARM_LEVEL harm ON c.HarmLevelID = harm.HarmID
        WHERE {date_filter}
        GROUP BY harm.HarmLevel;
        
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
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(query, params * 6)  # Params repeated for each sub-query
        
        # Total and status counts
        row = cursor.fetchone()
        total_never_events = row.total_never_events or 0
        unfinished_count = row.unfinished_count or 0
        finished_count = row.finished_count or 0
        open_count = row.open_count or 0
        under_investigation_count = row.under_investigation_count or 0
        rca_in_progress_count = row.rca_in_progress_count or 0
        pending_review_count = row.pending_review_count or 0
        resolved_count = row.resolved_count or 0
        closed_count = row.closed_count or 0
        cursor.nextset()
        
        # By category
        by_category = {}
        for row in cursor.fetchall():
            if row.category:
                by_category[row.category] = row.count
        cursor.nextset()
        
        # By severity
        by_severity = {}
        for row in cursor.fetchall():
            if row.severity:
                by_severity[row.severity] = row.count
        cursor.nextset()
        
        # By harm level
        by_harm_level = {}
        for row in cursor.fetchall():
            if row.harm_level:
                by_harm_level[row.harm_level] = row.count
        cursor.nextset()
        
        # Current month
        current_month = cursor.fetchone()
        current_month_count = current_month.count if current_month else 0
        cursor.nextset()
        
        # Previous month
        previous_month = cursor.fetchone()
        previous_month_count = previous_month.count if previous_month else 0
        
        # Calculate change percentage
        if previous_month_count > 0:
            change_percentage = ((current_month_count - previous_month_count) / previous_month_count) * 100
        else:
            change_percentage = 0
        
        # Get current month details
        current_date = datetime.now()
        current_month_name = current_date.strftime("%B %Y")
        previous_date = current_date - relativedelta(months=1)
        previous_month_name = previous_date.strftime("%B %Y")
        
        # Calculate category breakdown with percentages
        by_category_detailed = {}
        for category, count in by_category.items():
            percentage = (count / total_never_events * 100) if total_never_events > 0 else 0
            by_category_detailed[category] = {
                "count": count,
                "percentage": round(percentage, 1)
            }
        
        # Map to UI-expected nested structure
        return {
            "total_never_events": total_never_events,
            "goal": 0,
            "variance": total_never_events,
            "ytd_total": total_never_events,
            
            "unfinished_count": unfinished_count,
            "finished_count": finished_count,
            
            "by_status": {
                "OPEN": open_count,
                "UNDER_INVESTIGATION": under_investigation_count,
                "RCA_IN_PROGRESS": rca_in_progress_count,
                "PENDING_REVIEW": pending_review_count,
                "RESOLVED": resolved_count,
                "CLOSED": closed_count
            },
            
            "by_severity": {
                "CRITICAL": by_severity.get('CRITICAL', 0),
                "HIGH": by_severity.get('HIGH', 0),
                "MEDIUM": by_severity.get('MEDIUM', 0),
                "LOW": by_severity.get('LOW', 0)
            },
            
            "by_category": by_category_detailed,
            
            "by_harm_level": by_harm_level,
            
            "current_month": {
                "count": current_month_count,
                "month": current_month_name,
                "start_date": current_date.strftime("%Y-%m-01"),
                "end_date": (current_date.replace(day=1) + relativedelta(months=1) - relativedelta(days=1)).strftime("%Y-%m-%d"),
                "goal": 0,
                "status": "CRITICAL" if current_month_count > 0 else "GOOD"
            },
            
            "previous_month": {
                "count": previous_month_count,
                "month": previous_month_name,
                "comparison": f"{change_percentage:+.1f}%"
            },
            
            "rca_statistics": {
                "completed": 0,
                "in_progress": rca_in_progress_count,
                "pending": 0,
                "completion_rate": 0.0,
                "avg_days_to_complete": 12
            },
            
            "performance_indicators": {
                "time_to_investigation_avg_hours": 4,
                "time_to_resolution_avg_days": 21,
                "recurrence_rate": 0.0
            },
            
            "period": {
                "from_date": from_date or "2025-01-01",
                "to_date": to_date or datetime.now().strftime("%Y-%m-%d")
            }
        }
        
    finally:
        cursor.close()
        conn.close()


def get_never_events_trends(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    granularity: str = "monthly",
    group_by: str = "none"
) -> Dict[str, Any]:
    """
    Fetch time-series trend data for Never Events.
    
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
                WHERE c.ClinicalRiskTypeID = {NEVER_EVENT}
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
                WHERE c.ClinicalRiskTypeID = {NEVER_EVENT}
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
                WHERE c.ClinicalRiskTypeID = {NEVER_EVENT}
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
                WHERE c.ClinicalRiskTypeID = {NEVER_EVENT}
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
            "data": trends
        }
        
    finally:
        cursor.close()
        conn.close()


def get_never_event_details(never_event_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch comprehensive details for a specific never event.
    
    Args:
        never_event_id: Never event unique identifier (IncidentRequestCaseID)
    
    Returns:
        Dictionary with full never event details or None if not found
    """
    
    query = f"""
        SELECT 
            c.IncidentRequestCaseID as id,
            CONCAT('NE-', YEAR(c.FeedbackRecievedDate), '-', 
                   RIGHT('000' + CAST(c.IncidentRequestCaseID AS VARCHAR), 3)) as case_id,
            CASE 
                WHEN c.ClinicalRiskTypeID = 3 THEN risk_type.Name
                ELSE 'Never Event'
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
        AND c.ClinicalRiskTypeID = {NEVER_EVENT}
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(query, (never_event_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Build timeline
        timeline = [
            {
                "date": row[16].strftime('%Y-%m-%dT%H:%M:%S') if row[16] else "",
                "action": "Reported",
                "user": str(row[13]) if row[13] else "System",
                "notes": "Never event reported"
            }
        ]
        
        # Return flat structure matching UI expectations
        return {
            "id": row[0],
            "case_id": row[1],
            "title": row[2] or "",
            "description": row[3] or "",
            "severity": row[4] or "",
            "status": row[5] or "",
            "department": row[6] or "",
            "department_id": row[7] or 0,
            "category": row[8] or "",
            "category_id": row[9] or 0,
            "subcategory": row[10] or "",
            "date": row[11].strftime('%Y-%m-%d') if row[11] else "",
            "reported_by": row[12] or "",
            "assigned_to": str(row[13]) if row[13] else "",
            "root_cause_analysis": row[14] or "",
            "corrective_action_plan": row[15] or "",
            "timeline": timeline,
            "attachments": [],
            "created_at": row[16].strftime('%Y-%m-%d') if row[16] else "",
            "updated_at": row[17] if row[17] else ""
        }
        
    finally:
        cursor.close()
        conn.close()


# ==================== ANALYTICS ENDPOINTS ====================

def get_never_events_category_breakdown(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch category breakdown for never events analytics cards.
    
    Args:
        from_date: Filter from date (optional)
        to_date: Filter to date (optional)
    
    Returns:
        Dictionary with category breakdown including event types
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Build date filter
        date_filter = f"c.ClinicalRiskTypeID = {NEVER_EVENT}"
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
            ISNULL(domain.DomainName, 'Unknown') as category_name,
            COUNT(*) as count
        FROM dbo.APP_IncidentCase c
        LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
        WHERE {date_filter}
        GROUP BY domain.DomainName
        ORDER BY count DESC
        """
        
        cursor.execute(query, params)
        category_rows = cursor.fetchall()
        
        breakdown = []
        for cat_row in category_rows:
            percentage = (cat_row.count / total * 100) if total > 0 else 0
            
            breakdown.append({
                "category": cat_row.category_name,
                "category_id": 0,
                "count": cat_row.count,
                "percentage": round(percentage, 1)
            })
        
        return {
            "breakdown": breakdown,
            "total": total
        }
        
    finally:
        cursor.close()
        conn.close()


def get_never_events_timeline_comparison(
    period: str = "month"
) -> Dict[str, Any]:
    """
    Fetch timeline comparison for never events (current vs previous period).
    
    Args:
        period: Time period (month, quarter, or year)
    
    Returns:
        Dictionary with current vs previous period comparison
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        now = datetime.now()
        
        # Calculate date ranges based on period
        if period == "quarter":
            # Current quarter
            current_quarter = (now.month - 1) // 3 + 1
            current_start = datetime(now.year, (current_quarter - 1) * 3 + 1, 1)
            if current_quarter == 4:
                current_end = datetime(now.year, 12, 31)
            else:
                current_end = datetime(now.year, current_quarter * 3 + 1, 1) - relativedelta(days=1)
            
            # Previous quarter
            prev_start = current_start - relativedelta(months=3)
            prev_end = current_start - relativedelta(days=1)
            
            current_period = f"Q{current_quarter} {now.year}"
            prev_period = f"Q{((prev_start.month - 1) // 3 + 1)} {prev_start.year}"
            
        elif period == "year":
            # Current year
            current_start = datetime(now.year, 1, 1)
            current_end = datetime(now.year, 12, 31)
            
            # Previous year
            prev_start = datetime(now.year - 1, 1, 1)
            prev_end = datetime(now.year - 1, 12, 31)
            
            current_period = str(now.year)
            prev_period = str(now.year - 1)
            
        else:  # month (default)
            # Current month
            current_start = datetime(now.year, now.month, 1)
            if now.month == 12:
                current_end = datetime(now.year, 12, 31)
            else:
                current_end = datetime(now.year, now.month + 1, 1) - relativedelta(days=1)
            
            # Previous month
            prev_start = current_start - relativedelta(months=1)
            prev_end = current_start - relativedelta(days=1)
            
            current_period = current_start.strftime('%B %Y')
            prev_period = prev_start.strftime('%B %Y')
        
        # Get current period count
        current_query = f"""
        SELECT COUNT(*) as count
        FROM dbo.APP_IncidentCase
        WHERE ClinicalRiskTypeID = {NEVER_EVENT}
            AND FeedbackRecievedDate >= ?
            AND FeedbackRecievedDate <= ?
        """
        
        cursor.execute(current_query, [current_start.strftime('%Y-%m-%d'), current_end.strftime('%Y-%m-%d')])
        current_count = cursor.fetchone()[0]
        
        # Get previous period count
        prev_query = f"""
        SELECT COUNT(*) as count
        FROM dbo.APP_IncidentCase
        WHERE ClinicalRiskTypeID = {NEVER_EVENT}
            AND FeedbackRecievedDate >= ?
            AND FeedbackRecievedDate <= ?
        """
        
        cursor.execute(prev_query, [prev_start.strftime('%Y-%m-%d'), prev_end.strftime('%Y-%m-%d')])
        prev_count = cursor.fetchone()[0]
        
        # Calculate change
        percentage_change = ((current_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
        
        # Determine trend (for never events, fewer is better)
        if current_count < prev_count:
            trend = "improving"
        elif current_count > prev_count:
            trend = "worsening"
        else:
            trend = "stable"
        
        return {
            "current": {
                "period": current_period,
                "start_date": current_start.strftime('%Y-%m-%d'),
                "end_date": current_end.strftime('%Y-%m-%d'),
                "count": current_count
            },
            "previous": {
                "period": prev_period,
                "count": prev_count
            },
            "change": round(percentage_change, 1),
            "trend": trend
        }
        
    finally:
        cursor.close()
        conn.close()
