"""
Worker Reporting Database Layer
Handles read-only SQL queries for worker profile and performance metrics.

This module provides aggregation functions to gather worker (employee) statistics
from incidents, action items, and explanation workflows.

Note:
    - All queries are SELECT only (read-only)
    - Uses standard get_connection() pattern
    - Returns dicts or scalars, never Pydantic models
    - No business logic or scoring algorithms here
"""

from datetime import date, datetime
from typing import Dict, Any, Optional, List
from core.database import get_connection
from core.table_config import HR_EMPLOYEES_TABLE


# =============================================
# WORKER IDENTITY
# =============================================

def get_worker_identity(employee_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve worker identity information from HR employee view.
    
    Queries the APP_VIEWTABLE_HR_EMPLOYEES table to get basic employee
    information including name, job title, and organizational assignments.
    
    Args:
        employee_id: Unique employee identifier from HR system
    
    Returns:
        Dictionary with employee identity fields:
        - employee_id: int
        - full_name: str
        - job_title: str | None
        - department_id: int | None
        - section_id: int | None
        - administration_id: int | None
        - is_active: bool | None
        
        Returns None if employee not found.
    
    Example:
        >>> worker = get_worker_identity(12345)
        >>> print(worker['full_name'])
        'Ahmed Mohammed Al-Shahrani'
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT
                EmployeeID,
                FullName,
                JobTitle,
                DepartmentID,
                SectionID,
                AdministrationID,
                IsActive
            FROM {HR_EMPLOYEES_TABLE}
            WHERE EmployeeID = ?
        """, (employee_id,))
        
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return {
            "employee_id": row.EmployeeID,
            "full_name": row.FullName,
            "job_title": row.JobTitle,
            "department_id": row.DepartmentID,
            "section_id": row.SectionID,
            "administration_id": row.AdministrationID,
            "is_active": bool(row.IsActive) if row.IsActive is not None else None
        }
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# =============================================
# INCIDENT METRICS (via APP_IncidentCaseEmployee junction table)
# =============================================

def count_worker_incidents(
    employee_id: int,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None
) -> int:
    """
    Count incidents linked to a specific worker via APP_IncidentCaseEmployee.
    
    Uses the many-to-many junction table to find all incidents where
    this employee is linked (regardless of IsPrimary flag).
    
    Args:
        employee_id: Employee identifier to count incidents for
        date_from: Start date for incident filtering (optional)
        date_to: End date for incident filtering (optional)
    
    Returns:
        Integer count of incidents linked to this worker.
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        date_conditions = []
        params = [employee_id]
        
        if date_from:
            date_conditions.append("ic.FeedbackRecievedDate >= ?")
            params.append(date_from)
        
        if date_to:
            date_conditions.append("ic.FeedbackRecievedDate <= ?")
            params.append(date_to)
        
        date_filter = ""
        if date_conditions:
            date_filter = "AND " + " AND ".join(date_conditions)
        
        query = f"""
            SELECT COUNT(DISTINCT ic.IncidentRequestCaseID)
            FROM dbo.APP_IncidentCase ic
            INNER JOIN dbo.APP_IncidentCaseEmployee ice 
                ON ic.IncidentRequestCaseID = ice.IncidentRequestCaseID
            WHERE ice.EmployeeID = ?
            {date_filter}
        """
        
        cursor.execute(query, params)
        result = cursor.fetchone()
        return result[0] if result else 0
        
    except Exception as e:
        print(f"Error counting worker incidents: {str(e)}")
        return 0
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def count_worker_incidents_by_severity(
    employee_id: int,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None
) -> Dict[str, int]:
    """
    Count worker's incidents broken down by severity level.
    
    Uses APP_IncidentCaseEmployee join + APP_LOOKUP_SEVERITY.
    SeverityID 3 = High (user confirmed).
    
    Returns:
        Dict with keys: high, medium, low
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        date_conditions = []
        params = [employee_id]
        
        if date_from:
            date_conditions.append("ic.FeedbackRecievedDate >= ?")
            params.append(date_from)
        if date_to:
            date_conditions.append("ic.FeedbackRecievedDate <= ?")
            params.append(date_to)
        
        date_filter = ""
        if date_conditions:
            date_filter = "AND " + " AND ".join(date_conditions)
        
        query = f"""
            SELECT
                COUNT(CASE WHEN s.SeverityName = 'High' THEN 1 END) as high,
                COUNT(CASE WHEN s.SeverityName = 'Medium' THEN 1 END) as medium,
                COUNT(CASE WHEN s.SeverityName = 'Low' THEN 1 END) as low
            FROM dbo.APP_IncidentCase ic
            INNER JOIN dbo.APP_IncidentCaseEmployee ice 
                ON ic.IncidentRequestCaseID = ice.IncidentRequestCaseID
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY s 
                ON ic.SeverityID = s.SeverityID
            WHERE ice.EmployeeID = ?
            {date_filter}
        """
        
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if not row:
            return {"high": 0, "medium": 0, "low": 0}
        
        return {
            "high": row.high or 0,
            "medium": row.medium or 0,
            "low": row.low or 0
        }
        
    except Exception as e:
        print(f"Error counting worker severity: {str(e)}")
        return {"high": 0, "medium": 0, "low": 0}
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def count_worker_incidents_by_intent(
    employee_id: int,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None
) -> Dict[str, int]:
    """
    Count worker's incidents by feedback intent type (Good vs Bad classification).
    
    Classification from APP_LOOKUP_FEEDBACK_INTENT_TYPE:
      - FeedbackIntentTypeID 2 (NOTICE / تنويه) = Good
      - FeedbackIntentTypeID 3 (CRITIQUE_SUGGESTION / نقد/اقتراح) = Bad
      - FeedbackIntentTypeID 1 (IMPROVEMENT_OPPORTUNITY) = Neutral
      - FeedbackIntentTypeID 4 (OTHER) = Neutral
    
    Returns:
        Dict: {good: int, bad: int, neutral: int}
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        date_conditions = []
        params = [employee_id]
        
        if date_from:
            date_conditions.append("ic.FeedbackRecievedDate >= ?")
            params.append(date_from)
        if date_to:
            date_conditions.append("ic.FeedbackRecievedDate <= ?")
            params.append(date_to)
        
        date_filter = ""
        if date_conditions:
            date_filter = "AND " + " AND ".join(date_conditions)
        
        query = f"""
            SELECT
                COUNT(CASE WHEN ic.FeedbackIntentTypeID = 2 THEN 1 END) as good,
                COUNT(CASE WHEN ic.FeedbackIntentTypeID = 3 THEN 1 END) as bad,
                COUNT(CASE WHEN ic.FeedbackIntentTypeID IN (1, 4) THEN 1 END) as neutral
            FROM dbo.APP_IncidentCase ic
            INNER JOIN dbo.APP_IncidentCaseEmployee ice 
                ON ic.IncidentRequestCaseID = ice.IncidentRequestCaseID
            WHERE ice.EmployeeID = ?
            {date_filter}
        """
        
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if not row:
            return {"good": 0, "bad": 0, "neutral": 0}
        
        return {
            "good": row.good or 0,
            "bad": row.bad or 0,
            "neutral": row.neutral or 0
        }
        
    except Exception as e:
        print(f"Error counting worker intent: {str(e)}")
        return {"good": 0, "bad": 0, "neutral": 0}
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_worker_incidents_detail(
    employee_id: int,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    limit: int = 500
) -> List[Dict[str, Any]]:
    """
    Get detailed list of all incidents linked to a worker.
    
    Returns full incident details including severity, category, status,
    feedback intent type (good/bad), and case dates.
    
    Args:
        employee_id: Employee identifier
        date_from: Optional start date filter
        date_to: Optional end date filter
        limit: Max results (default 500)
    
    Returns:
        List of incident detail dicts
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        date_conditions = []
        params = [employee_id]
        
        if date_from:
            date_conditions.append("ic.FeedbackRecievedDate >= ?")
            params.append(date_from)
        if date_to:
            date_conditions.append("ic.FeedbackRecievedDate <= ?")
            params.append(date_to)
        
        date_filter = ""
        if date_conditions:
            date_filter = "AND " + " AND ".join(date_conditions)
        
        query = f"""
            SELECT TOP {limit}
                ic.IncidentRequestCaseID as id,
                ic.FeedbackRecievedDate as date,
                ic.PatientName as patient_name,
                ic.ComplaintText as complaint_text,
                cat.CategoryName as category,
                s.SeverityName as severity,
                cs.Name as status,
                fit.NameAr as intent_type_ar,
                fit.NameEn as intent_type_en,
                ic.FeedbackIntentTypeID as intent_type_id,
                CASE 
                    WHEN ic.FeedbackIntentTypeID = 2 THEN 'good'
                    WHEN ic.FeedbackIntentTypeID = 3 THEN 'bad'
                    ELSE 'neutral'
                END as classification,
                ice.IsPrimary as is_primary
            FROM dbo.APP_IncidentCase ic
            INNER JOIN dbo.APP_IncidentCaseEmployee ice 
                ON ic.IncidentRequestCaseID = ice.IncidentRequestCaseID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat 
                ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY s 
                ON ic.SeverityID = s.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs 
                ON ic.CaseStatusID = cs.CaseStatusID
            LEFT JOIN dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE fit 
                ON ic.FeedbackIntentTypeID = fit.FeedbackIntentTypeID
            WHERE ice.EmployeeID = ?
            {date_filter}
            ORDER BY ic.FeedbackRecievedDate DESC
        """
        
        cursor.execute(query, params)
        columns = [col[0] for col in cursor.description]
        incidents = []
        
        for row in cursor.fetchall():
            incident = dict(zip(columns, row))
            # Format date
            if incident.get('date'):
                incident['date'] = incident['date'].strftime('%Y-%m-%d')
            incidents.append(incident)
        
        return incidents
        
    except Exception as e:
        print(f"Error getting worker incidents detail: {str(e)}")
        return []
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# =============================================
# ACTION ITEM METRICS
# =============================================

def count_worker_action_items(
    employee_id: int,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None
) -> Dict[str, int]:
    """
    Count action items for incidents linked to this worker.
    
    Joins through APP_IncidentCaseEmployee -> APP_IncidentCase -> APP_ActionItem
    to find action items on incidents where this employee is linked.
    
    Args:
        employee_id: Employee identifier
        date_from: Start date for filtering (optional)
        date_to: End date for filtering (optional)
    
    Returns:
        Dict: {total, completed, overdue}
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        date_conditions = []
        params = [employee_id]
        
        if date_from:
            date_conditions.append("ai.CreatedAt >= ?")
            params.append(datetime.combine(date_from, datetime.min.time()))
        
        if date_to:
            date_conditions.append("ai.CreatedAt <= ?")
            params.append(datetime.combine(date_to, datetime.max.time()))
        
        date_filter = ""
        if date_conditions:
            date_filter = "AND " + " AND ".join(date_conditions)
        
        query = f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN ai.IsDone = 1 THEN 1 ELSE 0 END) as completed,
                SUM(CASE 
                    WHEN ai.IsDone = 0 
                    AND ai.DueDate IS NOT NULL 
                    AND ai.DueDate < CAST(GETDATE() AS DATE) 
                    THEN 1 
                    ELSE 0 
                END) as overdue
            FROM dbo.APP_ActionItem ai
            INNER JOIN dbo.APP_IncidentCaseEmployee ice 
                ON ai.IncidentRequestCaseID = ice.IncidentRequestCaseID
            WHERE ice.EmployeeID = ?
            {date_filter}
        """
        
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if not row:
            return {"total": 0, "completed": 0, "overdue": 0}
        
        return {
            "total": row.total or 0,
            "completed": row.completed or 0,
            "overdue": row.overdue or 0
        }
        
    except Exception as e:
        print(f"Error counting worker action items: {str(e)}")
        return {"total": 0, "completed": 0, "overdue": 0}
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# =============================================
# EXPLANATION STATUS METRICS
# =============================================

def count_worker_explanation_status(
    employee_id: int,
    status_id: Optional[int] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None
) -> Dict[int, int]:
    """
    Count explanation statuses for incidents created/handled by a worker.
    
    Counts incident cases by their ExplanationStatusID, optionally filtered
    by specific status and date range. This helps track how many explanations
    were accepted, rejected, or are pending for incidents associated with
    this worker.
    
    Explanation Status IDs:
        - 1: Waiting (pending explanation)
        - 2: Submitted (explanation provided, under review)
        - 3: Accepted (explanation approved) ✅
        - 4: Rejected (explanation declined, needs revision) ⚠️
    
    Discovery Note:
        Uses CreatedByUserID to link workers to incidents. Future enhancement
        may include tracking which employee submitted explanations separately.
    
    Args:
        employee_id: Employee identifier to count explanations for
        status_id: Optional filter for specific ExplanationStatusID
        date_from: Start date for incident filtering (optional)
        date_to: End date for incident filtering (optional)
    
    Returns:
        Dictionary mapping status_id -> count
        If status_id specified, returns single key-value pair.
        If status_id is None, returns all statuses with counts.
    
    Example:
        >>> # Get all explanation statuses
        >>> statuses = count_worker_explanation_status(12345)
        >>> print(f"Accepted: {statuses.get(3, 0)}, Rejected: {statuses.get(4, 0)}")
        
        >>> # Get only rejected explanations
        >>> rejected = count_worker_explanation_status(12345, status_id=4)
        >>> print(f"Rejected: {rejected.get(4, 0)}")
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Build filter conditions - use APP_IncidentCaseEmployee junction
        conditions = ["ice.EmployeeID = ?"]
        params = [employee_id]
        
        if status_id is not None:
            conditions.append("ic.ExplanationStatusID = ?")
            params.append(status_id)
        
        if date_from:
            conditions.append("ic.FeedbackRecievedDate >= ?")
            params.append(date_from)
        
        if date_to:
            conditions.append("ic.FeedbackRecievedDate <= ?")
            params.append(date_to)
        
        where_clause = " AND ".join(conditions)
        
        # Group by ExplanationStatusID to get counts per status
        query = f"""
            SELECT
                ic.ExplanationStatusID,
                COUNT(*) as count
            FROM dbo.APP_IncidentCase ic
            INNER JOIN dbo.APP_IncidentCaseEmployee ice
                ON ic.IncidentRequestCaseID = ice.IncidentRequestCaseID
            WHERE {where_clause}
            GROUP BY ic.ExplanationStatusID
        """
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Build status_id -> count mapping
        result = {}
        for row in rows:
            result[row.ExplanationStatusID] = row.count
        
        return result
        
    except Exception as e:
        print(f"Error counting worker explanation status: {str(e)}")
        return {}
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# =============================================
# WORKER METRICS (for full-history)
# =============================================

def get_worker_metrics(
    employee_id: int,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None
) -> Dict[str, Any]:
    """
    Get aggregated metrics for a worker including severity and category breakdowns.
    Uses APP_IncidentCaseEmployee join to find related incidents.
    
    Args:
        employee_id: Employee ID
        date_from: Optional start date filter
        date_to: Optional end date filter
    
    Returns:
        Dict with total_incidents, last_incident_date, severity_breakdown, category_breakdown
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        conditions = ["ice.EmployeeID = ?"]
        params = [employee_id]
        
        if date_from:
            conditions.append("ic.FeedbackRecievedDate >= ?")
            params.append(date_from)
        
        if date_to:
            conditions.append("ic.FeedbackRecievedDate <= ?")
            params.append(date_to)
        
        where_clause = " AND ".join(conditions)
        
        # Get total incidents and last incident date
        cursor.execute(f"""
            SELECT 
                COUNT(DISTINCT ic.IncidentRequestCaseID) as total_incidents,
                MAX(CONVERT(VARCHAR(10), ic.FeedbackRecievedDate, 23)) as last_incident_date
            FROM dbo.APP_IncidentCase ic
            INNER JOIN dbo.APP_IncidentCaseEmployee ice 
                ON ic.IncidentRequestCaseID = ice.IncidentRequestCaseID
            WHERE {where_clause}
        """, params)
        
        summary_row = cursor.fetchone()
        total_incidents = summary_row[0] if summary_row else 0
        last_incident_date = summary_row[1] if summary_row else None
        
        # Get severity breakdown
        cursor.execute(f"""
            SELECT 
                COALESCE(sev.SeverityName, 'Unknown') as severity,
                COUNT(DISTINCT ic.IncidentRequestCaseID) as count
            FROM dbo.APP_IncidentCase ic
            INNER JOIN dbo.APP_IncidentCaseEmployee ice 
                ON ic.IncidentRequestCaseID = ice.IncidentRequestCaseID
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev ON ic.SeverityID = sev.SeverityID
            WHERE {where_clause}
            GROUP BY sev.SeverityName
        """, params)
        
        severity_breakdown = {}
        for row in cursor.fetchall():
            severity_breakdown[row[0]] = row[1]
        
        # Get category breakdown
        cursor.execute(f"""
            SELECT 
                COALESCE(cat.CategoryName, 'Unknown') as category,
                COUNT(DISTINCT ic.IncidentRequestCaseID) as count
            FROM dbo.APP_IncidentCase ic
            INNER JOIN dbo.APP_IncidentCaseEmployee ice 
                ON ic.IncidentRequestCaseID = ice.IncidentRequestCaseID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat ON ic.CategoryID = cat.CategoryID
            WHERE {where_clause}
            GROUP BY cat.CategoryName
        """, params)
        
        category_breakdown = {}
        for row in cursor.fetchall():
            category_breakdown[row[0]] = row[1]
        
        return {
            "total_incidents": total_incidents,
            "last_incident_date": last_incident_date,
            "severity_breakdown": severity_breakdown,
            "category_breakdown": category_breakdown
        }
    
    except Exception as e:
        print(f"Error getting worker metrics: {str(e)}")
        return {
            "total_incidents": 0,
            "last_incident_date": None,
            "severity_breakdown": {},
            "category_breakdown": {}
        }
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# =============================================
# WORKER INCIDENTS FOR EXPORT
# =============================================

def get_worker_incidents_for_export(
    employee_id: int,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    include_profile: bool = True
) -> Dict[str, Any]:
    """
    Get worker data formatted for export (CSV/JSON/Word).
    
    Args:
        employee_id: Employee ID
        date_from: Optional start date filter
        date_to: Optional end date filter
        include_profile: Include worker profile in export
    
    Returns:
        Dict with worker profile and incidents list
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        export_data = {
            "export_date": datetime.now().isoformat(),
            "format": "json",
            "worker": None,
            "incidents": []
        }
        
        # Get worker profile if requested
        if include_profile:
            identity = get_worker_identity(employee_id)
            if identity:
                export_data["worker"] = {
                    "employee_id": identity.get('employee_id'),
                    "full_name": identity.get('full_name'),
                    "job_title": identity.get('job_title'),
                    "department_id": identity.get('department_id'),
                    "section_id": identity.get('section_id'),
                    "is_active": identity.get('is_active'),
                    "total_incidents": 0  # Will be computed below
                }
        
        # Build WHERE clause
        conditions = ["ice.EmployeeID = ?"]
        params = [employee_id]
        
        if date_from:
            conditions.append("ic.FeedbackRecievedDate >= ?")
            params.append(date_from)
        
        if date_to:
            conditions.append("ic.FeedbackRecievedDate <= ?")
            params.append(date_to)
        
        where_clause = " AND ".join(conditions)
        
        # Get incidents
        query = f"""
            SELECT
                ic.IncidentRequestCaseID as RecordID,
                CONVERT(VARCHAR(10), COALESCE(ic.FeedbackRecievedDate, ic.CreatedAt), 23) as Date,
                ic.PatientName as PatientName,
                COALESCE(cat.CategoryName, 'Unknown') as Category,
                COALESCE(sev.SeverityName, 'Unknown') as Severity,
                COALESCE(cs.Name, 'Open') as Status,
                CASE WHEN crt.Code IN ('RED_FLAG', 'NEVER_EVENT') THEN 'Yes' ELSE 'No' END as RedFlag,
                CASE 
                    WHEN ic.FeedbackIntentTypeID = 2 THEN 'Good'
                    WHEN ic.FeedbackIntentTypeID = 3 THEN 'Bad'
                    ELSE 'Neutral'
                END as Classification,
                ic.ComplaintText as Description
            FROM dbo.APP_IncidentCase ic
            INNER JOIN dbo.APP_IncidentCaseEmployee ice 
                ON ic.IncidentRequestCaseID = ice.IncidentRequestCaseID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs ON ic.CaseStatusID = cs.CaseStatusID
            LEFT JOIN dbo.APP_LOOKUP_CLINICAL_RISK_TYPE crt ON ic.ClinicalRiskTypeID = crt.ClinicalRiskTypeID
            WHERE {where_clause}
            ORDER BY ic.FeedbackRecievedDate DESC
        """
        
        cursor.execute(query, params)
        columns = [col[0] for col in cursor.description]
        export_data["incidents"] = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Update total count
        if export_data["worker"]:
            export_data["worker"]["total_incidents"] = len(export_data["incidents"])
        
        return export_data
    
    except Exception as e:
        print(f"Error getting worker incidents for export: {str(e)}")
        return {
            "export_date": datetime.now().isoformat(),
            "format": "json",
            "worker": None,
            "incidents": []
        }
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
