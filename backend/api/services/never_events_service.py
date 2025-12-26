"""
Never Events Service
Handles data retrieval and filtering for Never Events page.
Never events are zero-tolerance incidents requiring immediate reporting and investigation.
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, date
import pyodbc
from dateutil.relativedelta import relativedelta

from ..db_layer.database import get_connection


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
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Fetch list of never events with optional filtering and search.
    
    Args:
        search: Search by record ID, patient name, or event type
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
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Build WHERE clause dynamically
        where_conditions = [f"ClinicalRiskTypeID = {NEVER_EVENT}"]
        
        # Status filter
        if status and status.upper() != 'ALL':
            status_map = {
                'OPEN': 1,
                'UNDER_REVIEW': 2,
                'FINISHED': 3
            }
            status_id = status_map.get(status.upper())
            if status_id:
                where_conditions.append(f"tbl_main.CaseStatusID = {status_id}")
        
        # Date range filters
        if from_date:
            where_conditions.append(f"tbl_main.FeedbackReceivedDate >= '{from_date}'")
        if to_date:
            where_conditions.append(f"tbl_main.FeedbackReceivedDate <= '{to_date}'")
        
        # Department filter
        if department:
            where_conditions.append(f"dept.NameEN LIKE N'%{department}%'")
        
        # Category filter (Domain)
        if category:
            where_conditions.append(f"domain.NameEN LIKE N'%{category}%'")
        
        # Search filter (record ID, patient name, event type)
        if search:
            search_conditions = [
                f"tbl_main.RecordNo LIKE N'%{search}%'",
                f"tbl_main.PatientName LIKE N'%{search}%'",
                f"domain.NameAR LIKE N'%{search}%'",
                f"domain.NameEN LIKE N'%{search}%'"
            ]
            where_conditions.append(f"({' OR '.join(search_conditions)})")
        
        where_clause = " AND ".join(where_conditions)
        
        # Query for never events list
        query = f"""
        SELECT 
            tbl_main.ID as never_event_id,
            CONCAT('NE-', YEAR(tbl_main.FeedbackReceivedDate), '-', 
                   FORMAT(ROW_NUMBER() OVER (PARTITION BY YEAR(tbl_main.FeedbackReceivedDate) 
                                             ORDER BY tbl_main.FeedbackReceivedDate), '000')) as recordID,
            CONVERT(VARCHAR(10), tbl_main.FeedbackReceivedDate, 23) as date,
            tbl_main.PatientName as patient_name,
            tbl_main.PatientID as patient_id,
            domain.NameEN as never_event_type,
            domain.NameAR as never_event_type_ar,
            ISNULL(domain.NameEN, N'') as never_event_category,
            CASE 
                WHEN tbl_main.CaseStatusID = 1 THEN 'OPEN'
                WHEN tbl_main.CaseStatusID = 2 THEN 'UNDER_REVIEW'
                WHEN tbl_main.CaseStatusID = 3 THEN 'FINISHED'
                ELSE 'OPEN'
            END as status,
            ISNULL(severity.NameEN, 'HIGH') as severity,
            ISNULL(dept.NameEN, N'') as department,
            ISNULL(dept.NameAR, N'') as qism,
            CONCAT('INC-', YEAR(tbl_main.FeedbackReceivedDate), '-', 
                   FORMAT(tbl_main.ID, '0000')) as incident_id
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS tbl_main
        LEFT JOIN IncidentManager.dbo.AdministrationUnit dept 
            ON tbl_main.IssuerDepartment = dept.AdministrationUnit_ID
        LEFT JOIN IncidentManager.dbo.APP_LOOKUP_DOMAIN domain 
            ON tbl_main.Domain = domain.ID
        LEFT JOIN IncidentManager.dbo.APP_LOOKUP_SEVERITY severity 
            ON tbl_main.SeverityLevel = severity.ID
        WHERE {where_clause}
        ORDER BY tbl_main.FeedbackReceivedDate DESC
        OFFSET {offset} ROWS
        FETCH NEXT {limit} ROWS ONLY
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Get total count
        count_query = f"""
        SELECT COUNT(*) as total
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS tbl_main
        LEFT JOIN IncidentManager.dbo.AdministrationUnit dept 
            ON tbl_main.IssuerDepartment = dept.AdministrationUnit_ID
        LEFT JOIN IncidentManager.dbo.APP_LOOKUP_DOMAIN domain 
            ON tbl_main.Domain = domain.ID
        WHERE {where_clause}
        """
        
        cursor.execute(count_query)
        total = cursor.fetchone()[0]
        
        # Format results
        never_events = []
        for row in rows:
            never_events.append({
                "id": row.never_event_id,
                "recordID": row.recordID,
                "date": row.date,
                "patientName": row.patient_name or "",
                "patientID": row.patient_id or "",
                "neverEventType": row.never_event_type or "",
                "neverEventTypeAr": row.never_event_type_ar or "",
                "neverEventCategory": row.never_event_category or "",
                "status": row.status,
                "severity": row.severity,
                "department": row.department,
                "qism": row.qism,
                "incidentID": row.incident_id
            })
        
        return {
            "never_events": never_events,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    finally:
        cursor.close()
        conn.close()


def get_never_events_statistics(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch summary statistics for never events KPI cards.
    
    Args:
        from_date: Statistics from date (optional)
        to_date: Statistics to date (optional)
    
    Returns:
        Dictionary with aggregated statistics
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Build date filter
        date_filter = f"ClinicalRiskTypeID = {NEVER_EVENT}"
        if from_date:
            date_filter += f" AND FeedbackReceivedDate >= '{from_date}'"
        if to_date:
            date_filter += f" AND FeedbackReceivedDate <= '{to_date}'"
        
        # Multi-result query for all statistics
        query = f"""
        -- Total never events
        SELECT COUNT(*) as total_never_events
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS
        WHERE {date_filter};
        
        -- Unfinished count
        SELECT COUNT(*) as unfinished_count
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS
        WHERE {date_filter} AND CaseStatusID IN (1, 2);
        
        -- Finished count
        SELECT COUNT(*) as finished_count
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS
        WHERE {date_filter} AND CaseStatusID = 3;
        
        -- By status
        SELECT 
            CASE 
                WHEN CaseStatusID = 1 THEN 'OPEN'
                WHEN CaseStatusID = 2 THEN 'UNDER_REVIEW'
                WHEN CaseStatusID = 3 THEN 'FINISHED'
                ELSE 'OPEN'
            END as status,
            COUNT(*) as count
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS
        WHERE {date_filter}
        GROUP BY CaseStatusID;
        
        -- By category (Domain)
        SELECT 
            ISNULL(d.NameEN, 'Unknown') as category,
            COUNT(*) as count
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS tbl
        LEFT JOIN IncidentManager.dbo.APP_LOOKUP_DOMAIN d ON tbl.Domain = d.ID
        WHERE {date_filter}
        GROUP BY d.NameEN;
        
        -- By severity
        SELECT 
            ISNULL(s.NameEN, 'HIGH') as severity,
            COUNT(*) as count
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS tbl
        LEFT JOIN IncidentManager.dbo.APP_LOOKUP_SEVERITY s ON tbl.SeverityLevel = s.ID
        WHERE {date_filter}
        GROUP BY s.NameEN;
        
        -- Current month
        SELECT 
            COUNT(*) as count,
            FORMAT(GETDATE(), 'MMMM yyyy') as month
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS
        WHERE {date_filter}
            AND YEAR(FeedbackReceivedDate) = YEAR(GETDATE())
            AND MONTH(FeedbackReceivedDate) = MONTH(GETDATE());
        
        -- Previous month
        SELECT 
            COUNT(*) as count,
            FORMAT(DATEADD(MONTH, -1, GETDATE()), 'MMMM yyyy') as month
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS
        WHERE {date_filter}
            AND YEAR(FeedbackReceivedDate) = YEAR(DATEADD(MONTH, -1, GETDATE()))
            AND MONTH(FeedbackReceivedDate) = MONTH(DATEADD(MONTH, -1, GETDATE()));
        """
        
        cursor.execute(query)
        
        # Total never events
        total_never_events = cursor.fetchone()[0]
        cursor.nextset()
        
        # Unfinished count
        unfinished_count = cursor.fetchone()[0]
        cursor.nextset()
        
        # Finished count
        finished_count = cursor.fetchone()[0]
        cursor.nextset()
        
        # By status
        by_status = {"OPEN": 0, "UNDER_REVIEW": 0, "FINISHED": 0}
        for row in cursor.fetchall():
            by_status[row.status] = row.count
        cursor.nextset()
        
        # By category
        by_category = {}
        for row in cursor.fetchall():
            by_category[row.category] = row.count
        cursor.nextset()
        
        # By severity
        by_severity = {}
        for row in cursor.fetchall():
            by_severity[row.severity] = row.count
        cursor.nextset()
        
        # Current month
        current_month_row = cursor.fetchone()
        current_month = {
            "count": current_month_row.count,
            "month": current_month_row.month
        }
        cursor.nextset()
        
        # Previous month
        previous_month_row = cursor.fetchone()
        previous_month = {
            "count": previous_month_row.count,
            "month": previous_month_row.month
        }
        
        return {
            "total_never_events": total_never_events,
            "unfinished_count": unfinished_count,
            "finished_count": finished_count,
            "by_status": by_status,
            "by_category": by_category,
            "by_severity": by_severity,
            "current_month": current_month,
            "previous_month": previous_month,
            "period": {
                "from": from_date,
                "to": to_date
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
    Fetch time-series trend data for never events visualization.
    
    Args:
        from_date: Trend from date (default: last 12 months)
        to_date: Trend to date (default: today)
        granularity: monthly, quarterly, or weekly
        group_by: category, department, or none
    
    Returns:
        Dictionary with trend data array
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Default date range (last 12 months)
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        if not from_date:
            from_date = (datetime.now() - relativedelta(months=12)).strftime('%Y-%m-%d')
        
        # Date format based on granularity
        if granularity == "monthly":
            date_format = "FORMAT(FeedbackReceivedDate, 'MMMM yyyy')"
            date_format_ar = "FORMAT(FeedbackReceivedDate, 'MMMM yyyy', 'ar')"
        elif granularity == "quarterly":
            date_format = "CONCAT('Q', DATEPART(QUARTER, FeedbackReceivedDate), ' ', YEAR(FeedbackReceivedDate))"
            date_format_ar = date_format
        elif granularity == "weekly":
            date_format = "FORMAT(FeedbackReceivedDate, 'yyyy-MM-dd')"
            date_format_ar = date_format
        else:
            date_format = "FORMAT(FeedbackReceivedDate, 'MMMM yyyy')"
            date_format_ar = "FORMAT(FeedbackReceivedDate, 'MMMM yyyy', 'ar')"
        
        # Base query without grouping
        if group_by == "none":
            query = f"""
            SELECT 
                {date_format_ar} as period,
                COUNT(*) as count
            FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS
            WHERE ClinicalRiskTypeID = {NEVER_EVENT}
                AND FeedbackReceivedDate >= '{from_date}'
                AND FeedbackReceivedDate <= '{to_date}'
            GROUP BY {date_format}, {date_format_ar}
            ORDER BY MIN(FeedbackReceivedDate)
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            trends = []
            for row in rows:
                trends.append({
                    "period": row.period,
                    "count": row.count
                })
            
            return {
                "granularity": granularity,
                "period": {
                    "from": from_date,
                    "to": to_date
                },
                "data": trends
            }
        
        # Query with grouping
        elif group_by == "category":
            query = f"""
            SELECT 
                {date_format_ar} as period,
                ISNULL(d.NameEN, 'Unknown') as category,
                COUNT(*) as count
            FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS tbl
            LEFT JOIN IncidentManager.dbo.APP_LOOKUP_DOMAIN d ON tbl.Domain = d.ID
            WHERE tbl.ClinicalRiskTypeID = {NEVER_EVENT}
                AND tbl.FeedbackReceivedDate >= '{from_date}'
                AND tbl.FeedbackReceivedDate <= '{to_date}'
            GROUP BY {date_format}, {date_format_ar}, d.NameEN
            ORDER BY MIN(tbl.FeedbackReceivedDate), d.NameEN
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Group by period
            trends_dict = {}
            for row in rows:
                if row.period not in trends_dict:
                    trends_dict[row.period] = {"breakdown": {}, "total": 0}
                trends_dict[row.period]["breakdown"][row.category] = row.count
                trends_dict[row.period]["total"] += row.count
            
            trends = []
            for period, data in trends_dict.items():
                trends.append({
                    "period": period,
                    "total": data["total"],
                    "breakdown": data["breakdown"]
                })
            
            return {
                "granularity": granularity,
                "group_by": "category",
                "data": trends
            }
        
        elif group_by == "department":
            query = f"""
            SELECT 
                {date_format_ar} as period,
                ISNULL(dept.NameEN, 'Unknown') as department,
                COUNT(*) as count
            FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS tbl
            LEFT JOIN IncidentManager.dbo.AdministrationUnit dept 
                ON tbl.IssuerDepartment = dept.AdministrationUnit_ID
            WHERE tbl.ClinicalRiskTypeID = {NEVER_EVENT}
                AND tbl.FeedbackReceivedDate >= '{from_date}'
                AND tbl.FeedbackReceivedDate <= '{to_date}'
            GROUP BY {date_format}, {date_format_ar}, dept.NameEN
            ORDER BY MIN(tbl.FeedbackReceivedDate), dept.NameEN
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Group by period
            trends_dict = {}
            for row in rows:
                if row.period not in trends_dict:
                    trends_dict[row.period] = {"breakdown": {}, "total": 0}
                trends_dict[row.period]["breakdown"][row.department] = row.count
                trends_dict[row.period]["total"] += row.count
            
            trends = []
            for period, data in trends_dict.items():
                trends.append({
                    "period": period,
                    "total": data["total"],
                    "breakdown": data["breakdown"]
                })
            
            return {
                "granularity": granularity,
                "group_by": "department",
                "data": trends
            }
        
        else:
            # Invalid group_by, return ungrouped
            return get_never_events_trends(from_date, to_date, granularity, "none")
        
    finally:
        cursor.close()
        conn.close()


def get_never_event_details(never_event_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch comprehensive details for a specific never event.
    
    Args:
        never_event_id: Never event unique identifier
    
    Returns:
        Dictionary with full never event details or None if not found
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Query for never event details
        query = f"""
        SELECT 
            tbl_main.ID as never_event_id,
            CONCAT('NE-', YEAR(tbl_main.FeedbackReceivedDate), '-', 
                   FORMAT(ROW_NUMBER() OVER (PARTITION BY YEAR(tbl_main.FeedbackReceivedDate) 
                                             ORDER BY tbl_main.FeedbackReceivedDate), '000')) as recordID,
            CONVERT(VARCHAR(10), tbl_main.FeedbackReceivedDate, 23) as date,
            tbl_main.PatientName as patient_name,
            tbl_main.PatientID as patient_id,
            domain.NameEN as never_event_type,
            domain.NameAR as never_event_type_ar,
            ISNULL(domain.NameEN, N'') as never_event_category,
            CASE 
                WHEN tbl_main.CaseStatusID = 1 THEN 'OPEN'
                WHEN tbl_main.CaseStatusID = 2 THEN 'UNDER_REVIEW'
                WHEN tbl_main.CaseStatusID = 3 THEN 'FINISHED'
                ELSE 'OPEN'
            END as status,
            ISNULL(severity.NameEN, 'HIGH') as severity,
            ISNULL(dept.NameEN, N'') as department,
            ISNULL(dept.NameAR, N'') as qism,
            CONCAT('INC-', YEAR(tbl_main.FeedbackReceivedDate), '-', 
                   FORMAT(tbl_main.ID, '0000')) as incident_id,
            tbl_main.ComplaintContent as complaint_text,
            tbl_main.ImmediateAction as immediate_action,
            tbl_main.ActionsTaken as corrective_actions,
            tbl_main.RootCause as root_cause,
            tbl_main.ResponsiblePerson as responsible_person,
            target_dept.NameEN as target_department,
            CONVERT(VARCHAR(10), tbl_main.FeedbackReceivedDate, 23) as feedback_received_date,
            CONCAT(ISNULL(domain.NameEN, ''), ' > ', 
                   ISNULL(cat.NameEN, ''), ' > ',
                   ISNULL(subcat.NameEN, '')) as classification
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS tbl_main
        LEFT JOIN IncidentManager.dbo.AdministrationUnit dept 
            ON tbl_main.IssuerDepartment = dept.AdministrationUnit_ID
        LEFT JOIN IncidentManager.dbo.AdministrationUnit target_dept 
            ON tbl_main.TargetDepartment = target_dept.AdministrationUnit_ID
        LEFT JOIN IncidentManager.dbo.APP_LOOKUP_DOMAIN domain 
            ON tbl_main.Domain = domain.ID
        LEFT JOIN IncidentManager.dbo.APP_LOOKUP_CATEGORY cat 
            ON tbl_main.Category = cat.ID
        LEFT JOIN IncidentManager.dbo.APP_LOOKUP_SUBCATEGORY subcat 
            ON tbl_main.SubCategory = subcat.ID
        LEFT JOIN IncidentManager.dbo.APP_LOOKUP_SEVERITY severity 
            ON tbl_main.SeverityLevel = severity.ID
        WHERE tbl_main.ID = ? AND tbl_main.ClinicalRiskTypeID = {NEVER_EVENT}
        """
        
        cursor.execute(query, (never_event_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Format never event basic info
        never_event = {
            "id": row.never_event_id,
            "recordID": row.recordID,
            "date": row.date,
            "patientName": row.patient_name or "",
            "patientID": row.patient_id or "",
            "neverEventType": row.never_event_type or "",
            "neverEventTypeAr": row.never_event_type_ar or "",
            "neverEventCategory": row.never_event_category or "",
            "status": row.status,
            "severity": row.severity,
            "department": row.department,
            "qism": row.qism,
            "incidentID": row.incident_id
        }
        
        # Format incident details
        incident_details = {
            "incidentID": row.incident_id,
            "complaintText": row.complaint_text or "",
            "immediateAction": row.immediate_action or "",
            "corrective_actions": row.corrective_actions or "",
            "rootCause": row.root_cause or "",
            "responsiblePerson": row.responsible_person or "",
            "targetDepartment": row.target_department or "",
            "feedbackReceivedDate": row.feedback_received_date,
            "classification": row.classification
        }
        
        # Mock timeline (in production, this would come from audit log)
        timeline = [
            {
                "date": f"{row.date}T08:30:00Z",
                "event": "تم الإبلاغ عن الحدث",
                "user": "نظام الإبلاغ"
            },
            {
                "date": f"{row.date}T09:00:00Z",
                "event": "بدء التحقيق",
                "user": "فريق الجودة"
            }
        ]
        
        # Mock related actions (in production, this would come from actions table)
        related_actions = [
            {
                "action_id": "ACT-2024-001",
                "description": "مراجعة البروتوكولات",
                "status": "in_progress",
                "due_date": datetime.now().strftime('%Y-%m-%d')
            }
        ]
        
        return {
            "never_event": never_event,
            "incident_details": incident_details,
            "timeline": timeline,
            "related_actions": related_actions
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
        date_filter = f"tbl.ClinicalRiskTypeID = {NEVER_EVENT}"
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
        
        # Get category breakdown with types
        query = f"""
        SELECT 
            ISNULL(d.NameEN, 'Unknown') as category_name,
            ISNULL(d.NameAR, 'غير محدد') as category_name_ar,
            COUNT(*) as count
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS tbl
        LEFT JOIN IncidentManager.dbo.APP_LOOKUP_DOMAIN d ON tbl.Domain = d.ID
        WHERE {date_filter}
        GROUP BY d.NameEN, d.NameAR
        ORDER BY count DESC
        """
        
        cursor.execute(query)
        category_rows = cursor.fetchall()
        
        categories = []
        for cat_row in category_rows:
            percentage = (cat_row.count / total * 100) if total > 0 else 0
            
            # Get types for this category
            types_query = f"""
            SELECT 
                ISNULL(d.NameEN, 'Unknown') as type,
                ISNULL(d.NameAR, 'غير محدد') as type_ar,
                COUNT(*) as count
            FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS tbl
            LEFT JOIN IncidentManager.dbo.APP_LOOKUP_DOMAIN d ON tbl.Domain = d.ID
            WHERE {date_filter}
                AND ISNULL(d.NameEN, 'Unknown') = ?
            GROUP BY d.NameEN, d.NameAR
            ORDER BY count DESC
            """
            
            cursor.execute(types_query, (cat_row.category_name,))
            type_rows = cursor.fetchall()
            
            types = []
            for type_row in type_rows:
                types.append({
                    "type": type_row.type,
                    "type_ar": type_row.type_ar,
                    "count": type_row.count
                })
            
            categories.append({
                "category_name": cat_row.category_name,
                "category_name_ar": cat_row.category_name_ar,
                "count": cat_row.count,
                "percentage": round(percentage, 1),
                "types": types
            })
        
        return {
            "total": total,
            "goal": 0,
            "period": period_display,
            "categories": categories
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
            current_period_ar = f"الربع {current_quarter} {now.year}"
            prev_period_ar = f"الربع {((prev_start.month - 1) // 3 + 1)} {prev_start.year}"
            
        elif period == "year":
            # Current year
            current_start = datetime(now.year, 1, 1)
            current_end = datetime(now.year, 12, 31)
            
            # Previous year
            prev_start = datetime(now.year - 1, 1, 1)
            prev_end = datetime(now.year - 1, 12, 31)
            
            current_period = str(now.year)
            prev_period = str(now.year - 1)
            current_period_ar = str(now.year)
            prev_period_ar = str(now.year - 1)
            
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
            current_period_ar = current_start.strftime('%B %Y')  # Would need Arabic month names
            prev_period_ar = prev_start.strftime('%B %Y')
        
        # Get current period count
        current_query = f"""
        SELECT COUNT(*) as count
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS
        WHERE ClinicalRiskTypeID = {NEVER_EVENT}
            AND FeedbackReceivedDate >= '{current_start.strftime('%Y-%m-%d')}'
            AND FeedbackReceivedDate <= '{current_end.strftime('%Y-%m-%d')}'
        """
        
        cursor.execute(current_query)
        current_count = cursor.fetchone()[0]
        
        # Get previous period count
        prev_query = f"""
        SELECT COUNT(*) as count
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS
        WHERE ClinicalRiskTypeID = {NEVER_EVENT}
            AND FeedbackReceivedDate >= '{prev_start.strftime('%Y-%m-%d')}'
            AND FeedbackReceivedDate <= '{prev_end.strftime('%Y-%m-%d')}'
        """
        
        cursor.execute(prev_query)
        prev_count = cursor.fetchone()[0]
        
        # Calculate change
        absolute_change = current_count - prev_count
        percentage_change = ((current_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
        
        # Determine trend (for never events, fewer is better)
        if current_count < prev_count:
            trend = "improving"
        elif current_count > prev_count:
            trend = "worsening"
        else:
            trend = "stable"
        
        # Get year-to-date stats
        ytd_start = datetime(now.year, 1, 1)
        ytd_query = f"""
        SELECT COUNT(*) as count
        FROM IncidentManager.dbo.MAIN_COMPLAINT_ADVERSE_EVENTS_FORMS
        WHERE ClinicalRiskTypeID = {NEVER_EVENT}
            AND FeedbackReceivedDate >= '{ytd_start.strftime('%Y-%m-%d')}'
            AND FeedbackReceivedDate <= '{now.strftime('%Y-%m-%d')}'
        """
        
        cursor.execute(ytd_query)
        ytd_count = cursor.fetchone()[0]
        
        # Calculate average per month YTD
        months_elapsed = now.month
        avg_per_month = ytd_count / months_elapsed if months_elapsed > 0 else 0
        
        return {
            "goal": 0,
            "current": {
                "period": current_period,
                "period_ar": current_period_ar,
                "count": current_count,
                "start_date": current_start.strftime('%Y-%m-%d'),
                "end_date": current_end.strftime('%Y-%m-%d')
            },
            "previous": {
                "period": prev_period,
                "period_ar": prev_period_ar,
                "count": prev_count
            },
            "change": {
                "absolute": absolute_change,
                "percentage": round(percentage_change, 1),
                "trend": trend
            },
            "year_to_date": {
                "count": ytd_count,
                "average_per_month": round(avg_per_month, 1)
            }
        }
        
    finally:
        cursor.close()
        conn.close()
