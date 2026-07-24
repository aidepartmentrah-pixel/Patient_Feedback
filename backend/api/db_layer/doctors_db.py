"""
Database layer for Doctor endpoints.
Handles all SQL queries for doctor profiles, statistics, and incident tracking.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from core.database import get_connection
from core.table_config import DOCTORS_TABLE


# =============================================
# DOCTOR SEARCH / LIST
# =============================================

def search_doctors(
    query: Optional[str] = None,
    department: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Search for doctors from BOTH hospital and reserve tables.
    
    Merges results from:
    - APP_LOOKUP_DOCTOR (hospital system - read-only)
    - APP_RESERVE_DOCTOR (reserve table - user-created)
    
    Args:
        query: Search term (name)
        department: Filter by department (not used - tables don't have this field)
        status: Filter by status (active, inactive)
        limit: Max results
    
    Returns:
        List of doctor records with 'source' field indicating origin
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    where_parts_hospital = []
    where_parts_reserve = []
    params = []
    
    # Free-text search on name (both tables have DoctorName)
    if query:
        where_parts_hospital.append("d.DoctorName LIKE ?")
        where_parts_reserve.append("r.DoctorName LIKE ?")
        search_param = f"%{query}%"
        params.extend([search_param, search_param])
    
    # Status filter (both tables have IsActive as BIT)
    if status:
        status_value = 1 if status.lower() == 'active' else 0
        where_parts_hospital.append("d.IsActive = ?")
        where_parts_reserve.append("r.IsActive = ?")
        params.extend([status_value, status_value])
    
    where_clause_hospital = " WHERE " + " AND ".join(where_parts_hospital) if where_parts_hospital else ""
    where_clause_reserve = " WHERE " + " AND ".join(where_parts_reserve) if where_parts_reserve else ""
    
    # UNION query: merge hospital and reserve tables
    query_sql = f"""
        SELECT TOP {limit} *
        FROM (
            -- Hospital doctors
            SELECT
                d.DoctorID as id,
                d.DoctorName as name_en,
                d.DoctorName as name_ar,
                d.Specialty as specialty,
                CASE WHEN d.IsActive = 1 THEN 'active' ELSE 'inactive' END as status,
                'hospital' as source,
                d.SourceSystem as source_system,
                d.LastSyncedAt as last_synced_at
            FROM dbo.APP_LOOKUP_DOCTOR d
            {where_clause_hospital}
            
            UNION ALL
            
            -- Reserve doctors
            SELECT
                r.DoctorID as id,
                r.DoctorName as name_en,
                r.DoctorName as name_ar,
                r.Specialty as specialty,
                CASE WHEN r.IsActive = 1 THEN 'active' ELSE 'inactive' END as status,
                'reserve' as source,
                r.SourceSystem as source_system,
                r.LastSyncedAt as last_synced_at
            FROM dbo.APP_RESERVE_DOCTOR r
            {where_clause_reserve}
        ) AS combined
        ORDER BY name_en ASC
    """
    
    try:
        cursor.execute(query_sql, params)
        columns = [col[0] for col in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            doctor = dict(zip(columns, row))
            # Format datetime if present
            if doctor.get('last_synced_at'):
                doctor['last_synced_at'] = doctor['last_synced_at'].strftime('%Y-%m-%d %H:%M:%S')
            results.append(doctor)
        
        return results
    finally:
        cursor.close()
        conn.close()


# =============================================
# GET RESERVE DOCTORS ONLY
# =============================================

def get_reserve_doctors(limit: int = 100, include_inactive: bool = False) -> List[Dict[str, Any]]:
    """
    Get all doctors from the reserve table only.
    
    Returns only user-created doctors (not from hospital system).
    By default, only returns active doctors.
    
    Args:
        limit: Max results
        include_inactive: If False (default), only returns active doctors
    
    Returns:
        List of reserve doctor records
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    where_clause = "" if include_inactive else "WHERE r.IsActive = 1"
    
    query_sql = f"""
        SELECT TOP {limit}
            r.DoctorID as id,
            r.DoctorName as name_en,
            r.DoctorName as name_ar,
            r.Specialty as specialty,
            CASE WHEN r.IsActive = 1 THEN 'active' ELSE 'inactive' END as status,
            'reserve' as source,
            r.SourceSystem as source_system,
            r.LastSyncedAt as last_synced_at
        FROM dbo.APP_RESERVE_DOCTOR r
        {where_clause}
        ORDER BY r.LastSyncedAt DESC, r.DoctorName ASC
    """
    
    try:
        cursor.execute(query_sql)
        columns = [col[0] for col in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            doctor = dict(zip(columns, row))
            # Format datetime if present
            if doctor.get('last_synced_at'):
                doctor['last_synced_at'] = doctor['last_synced_at'].strftime('%Y-%m-%d %H:%M:%S')
            results.append(doctor)
        
        return results
    finally:
        cursor.close()
        conn.close()


# =============================================
# DOCTOR PROFILE
# =============================================

def get_doctor_profile(doctor_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch detailed profile information for a doctor from BOTH sources.
    
    Checks both hospital and reserve tables. Reserve table is checked first
    to give priority to user-created doctors if IDs conflict (unlikely).
    
    Args:
        doctor_id: Doctor ID
    
    Returns:
        Doctor profile dict or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Try reserve table first (user-created doctors)
    query_reserve = """
        SELECT
            r.DoctorID as id,
            r.DoctorName as name_en,
            r.DoctorName as name_ar,
            r.Specialty as specialty,
            CASE WHEN r.IsActive = 1 THEN 'active' ELSE 'inactive' END as status,
            'reserve' as source,
            r.SourceSystem as source_system,
            r.LastSyncedAt as last_synced_at
        FROM dbo.APP_RESERVE_DOCTOR r
        WHERE r.DoctorID = ?
    """
    
    try:
        cursor.execute(query_reserve, (doctor_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [col[0] for col in cursor.description]
            profile = dict(zip(columns, row))
            
            # Format datetime
            if profile.get('last_synced_at'):
                profile['last_synced_at'] = profile['last_synced_at'].strftime('%Y-%m-%d %H:%M:%S')
            
            return profile
        
        # Not found in reserve, try the hospital lookup table (name from
        # db_settings views.doctors -- APP_LOOKUP_DOCTOR, not the obsolete
        # VW_Doctors view which is intentionally absent on fresh installs).
        # Same column shape as APP_RESERVE_DOCTOR (see search_doctors above).
        query_hospital = f"""
            SELECT
                d.DoctorID as id,
                d.DoctorName as name_en,
                d.DoctorName as name_ar,
                d.Specialty as specialty,
                CASE WHEN d.IsActive = 1 THEN 'active' ELSE 'inactive' END as status,
                'hospital' as source,
                d.SourceSystem as source_system,
                d.LastSyncedAt as last_synced_at
            FROM dbo.{DOCTORS_TABLE} d
            WHERE d.DoctorID = ?
        """
        
        cursor.execute(query_hospital, (doctor_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        columns = [col[0] for col in cursor.description]
        profile = dict(zip(columns, row))
        
        # Format datetime
        if profile.get('last_synced_at'):
            profile['last_synced_at'] = profile['last_synced_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        return profile
        
    finally:
        cursor.close()
        conn.close()


# =============================================
# DOCTOR STATISTICS
# =============================================

def get_doctor_statistics(
    doctor_id: int,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get aggregated incident statistics for a doctor.
    
    Args:
        doctor_id: Doctor ID
        from_date: Start date (YYYY-MM-DD) or None for default
        to_date: End date (YYYY-MM-DD) or None for default
    
    Returns:
        Dict with total, high, medium, low, red_flags counts
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Parse dates or use defaults
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    if not from_date:
        from_date_obj = datetime.now() - timedelta(days=180)  # 6 months
        from_date = from_date_obj.strftime('%Y-%m-%d')
    
    query = """
        SELECT
            COUNT(DISTINCT ic.IncidentRequestCaseID) as total,
            COUNT(CASE WHEN s.SeverityName = 'High' THEN 1 END) as high,
            COUNT(CASE WHEN s.SeverityName = 'Medium' THEN 1 END) as medium,
            COUNT(CASE WHEN s.SeverityName = 'Low' THEN 1 END) as low,
            COUNT(CASE WHEN crt.Code IN ('RED_FLAG', 'NEVER_EVENT') THEN 1 END) as red_flags,
            COUNT(CASE WHEN ic.FeedbackIntentTypeID = 2 THEN 1 END) as good_feedback,
            COUNT(CASE WHEN ic.FeedbackIntentTypeID = 3 THEN 1 END) as bad_feedback,
            COUNT(CASE WHEN ic.FeedbackIntentTypeID IN (1, 4) OR ic.FeedbackIntentTypeID IS NULL THEN 1 END) as neutral_feedback
        FROM dbo.APP_IncidentCase ic
        INNER JOIN dbo.APP_IncidentCaseDoctor icd ON ic.IncidentRequestCaseID = icd.IncidentRequestCaseID
        LEFT JOIN dbo.APP_LOOKUP_SEVERITY s ON ic.SeverityID = s.SeverityID
        LEFT JOIN dbo.APP_LOOKUP_CLINICAL_RISK_TYPE crt ON ic.ClinicalRiskTypeID = crt.ClinicalRiskTypeID
        WHERE icd.DoctorID = ?
        AND ic.FeedbackRecievedDate >= ?
        AND ic.FeedbackRecievedDate <= ?
        AND ic.RecordTypeID = 1
    """
    
    try:
        cursor.execute(query, (doctor_id, from_date, to_date))
        row = cursor.fetchone()
        
        if not row:
            return {
                'total': 0,
                'high': 0,
                'medium': 0,
                'low': 0,
                'red_flags': 0,
                'good_feedback': 0,
                'bad_feedback': 0,
                'neutral_feedback': 0
            }
        
        return {
            'total': row.total or 0,
            'high': row.high or 0,
            'medium': row.medium or 0,
            'low': row.low or 0,
            'red_flags': row.red_flags or 0,
            'good_feedback': row.good_feedback or 0,
            'bad_feedback': row.bad_feedback or 0,
            'neutral_feedback': row.neutral_feedback or 0
        }
    finally:
        cursor.close()
        conn.close()


# =============================================
# DOCTOR CATEGORY BREAKDOWN
# =============================================

def get_doctor_category_breakdown(
    doctor_id: int,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get category breakdown for doctor's incidents.
    
    Args:
        doctor_id: Doctor ID
        from_date: Start date or None for default
        to_date: End date or None for default
    
    Returns:
        List of {name, count} dicts
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Parse dates or use defaults
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    if not from_date:
        from_date_obj = datetime.now() - timedelta(days=180)
        from_date = from_date_obj.strftime('%Y-%m-%d')
    
    query = """
        SELECT
            cat.CategoryName as name,
            COUNT(DISTINCT ic.IncidentRequestCaseID) as count
        FROM dbo.APP_IncidentCase ic
        INNER JOIN dbo.APP_IncidentCaseDoctor icd ON ic.IncidentRequestCaseID = icd.IncidentRequestCaseID
        LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat ON ic.CategoryID = cat.CategoryID
        WHERE icd.DoctorID = ?
        AND ic.FeedbackRecievedDate >= ?
        AND ic.FeedbackRecievedDate <= ?
        AND ic.RecordTypeID = 1
        GROUP BY cat.CategoryID, cat.CategoryName
        ORDER BY count DESC
    """
    
    try:
        cursor.execute(query, (doctor_id, from_date, to_date))
        columns = [col[0] for col in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return results
    finally:
        cursor.close()
        conn.close()


# =============================================
# DOCTOR MONTHLY TREND
# =============================================

def get_doctor_monthly_trend(
    doctor_id: int,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get monthly incident trend for doctor.
    
    Args:
        doctor_id: Doctor ID
        from_date: Start date or None for default
        to_date: End date or None for default
    
    Returns:
        List of {month, count} dicts, including zero-count months
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Parse dates or use defaults
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    if not from_date:
        from_date_obj = datetime.now() - timedelta(days=180)
        from_date = from_date_obj.strftime('%Y-%m-%d')
    
    query = """
        SELECT
            YEAR(ic.FeedbackRecievedDate) as year,
            MONTH(ic.FeedbackRecievedDate) as month,
            COUNT(DISTINCT ic.IncidentRequestCaseID) as count
        FROM dbo.APP_IncidentCase ic
        INNER JOIN dbo.APP_IncidentCaseDoctor icd ON ic.IncidentRequestCaseID = icd.IncidentRequestCaseID
        WHERE icd.DoctorID = ?
        AND ic.FeedbackRecievedDate >= ?
        AND ic.FeedbackRecievedDate <= ?
        GROUP BY YEAR(ic.FeedbackRecievedDate), MONTH(ic.FeedbackRecievedDate)
        ORDER BY year, month
    """
    
    try:
        cursor.execute(query, (doctor_id, from_date, to_date))
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            year = row.year
            month = row.month
            count = row.count or 0
            
            # Format month label
            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_label = month_names[month]
            
            # Include year if spanning multiple years
            from_year = int(from_date[:4])
            to_year = int(to_date[:4])
            if from_year != to_year:
                month_label = f"{month_label} {year}"
            
            results.append({
                'month': month_label,
                'count': count
            })
        
        return results
    finally:
        cursor.close()
        conn.close()


# =============================================
# DOCTOR INCIDENTS
# =============================================

def get_doctor_incidents(
    doctor_id: int,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    red_flags_only: bool = False,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Get paginated incidents linked to a doctor.
    
    Args:
        doctor_id: Doctor ID
        from_date: Start date or None for default
        to_date: End date or None for default
        severity: Filter by severity (HIGH, MEDIUM, LOW)
        status: Filter by status (OPEN, UNDER_REVIEW, CLOSED)
        red_flags_only: Only return red flag incidents
        limit: Page size
        offset: Pagination offset
    
    Returns:
        Dict with incidents array and pagination info
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Parse dates or use defaults
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    if not from_date:
        from_date_obj = datetime.now() - timedelta(days=180)
        from_date = from_date_obj.strftime('%Y-%m-%d')
    
    where_parts = [
        "icd.DoctorID = ?",
        "ic.FeedbackRecievedDate >= ?",
        "ic.FeedbackRecievedDate <= ?"
    ]
    params = [doctor_id, from_date, to_date]
    
    if severity:
        where_parts.append("s.SeverityName = ?")
        params.append(severity)
    
    if status:
        where_parts.append("cs.Name = ?")
        params.append(status)
    
    if red_flags_only:
        where_parts.append("crt.Code IN ('RED_FLAG', 'NEVER_EVENT')")
    
    where_clause = " AND ".join(where_parts)
    
    # Get total count
    count_query = f"""
        SELECT COUNT(DISTINCT ic.IncidentRequestCaseID) as total
        FROM dbo.APP_IncidentCase ic
        INNER JOIN dbo.APP_IncidentCaseDoctor icd ON ic.IncidentRequestCaseID = icd.IncidentRequestCaseID
        LEFT JOIN dbo.APP_LOOKUP_SEVERITY s ON ic.SeverityID = s.SeverityID
        LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs ON ic.CaseStatusID = cs.CaseStatusID
        LEFT JOIN dbo.APP_LOOKUP_CLINICAL_RISK_TYPE crt ON ic.ClinicalRiskTypeID = crt.ClinicalRiskTypeID
        WHERE {where_clause}
    """
    
    cursor.execute(count_query, params)
    total = cursor.fetchone().total or 0
    
    # Get paginated results
    incidents_query = f"""
        SELECT
            ic.IncidentRequestCaseID as id,
            ic.FeedbackRecievedDate as date,
            CAST(ic.IncidentRequestCaseID AS VARCHAR) as incident_id,
            ic.PatientName as patient_id,
            ic.PatientName as patient_name,
            cat.CategoryName as category,
            cat.CategoryName as category_ar,
            s.SeverityName as severity,
            cs.Name as status,
            CASE WHEN crt.Code IN ('RED_FLAG', 'NEVER_EVENT') THEN 1 ELSE 0 END as is_red_flag,
            ic.FeedbackIntentTypeID as intent_type_id,
            fit.NameAr as intent_type_ar,
            fit.NameEn as intent_type_en,
            CASE
                WHEN ic.FeedbackIntentTypeID = 2 THEN 'good'
                WHEN ic.FeedbackIntentTypeID = 3 THEN 'bad'
                ELSE 'neutral'
            END as classification
        FROM dbo.APP_IncidentCase ic
        INNER JOIN dbo.APP_IncidentCaseDoctor icd ON ic.IncidentRequestCaseID = icd.IncidentRequestCaseID
        LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat ON ic.CategoryID = cat.CategoryID
        LEFT JOIN dbo.APP_LOOKUP_SEVERITY s ON ic.SeverityID = s.SeverityID
        LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs ON ic.CaseStatusID = cs.CaseStatusID
        LEFT JOIN dbo.APP_LOOKUP_CLINICAL_RISK_TYPE crt ON ic.ClinicalRiskTypeID = crt.ClinicalRiskTypeID
        LEFT JOIN dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE fit ON ic.FeedbackIntentTypeID = fit.FeedbackIntentTypeID
        WHERE {where_clause}
        ORDER BY ic.FeedbackRecievedDate DESC
        OFFSET {offset} ROWS
        FETCH NEXT {limit} ROWS ONLY
    """
    
    try:
        cursor.execute(incidents_query, params)
        columns = [col[0] for col in cursor.description]
        incidents = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Format dates
        for incident in incidents:
            if incident.get('date'):
                incident['date'] = incident['date'].strftime('%Y-%m-%d')
        
        return {
            'incidents': incidents,
            'total': total,
            'limit': limit,
            'offset': offset
        }
    finally:
        cursor.close()
        conn.close()


# =============================================
# DOCTOR METRICS (for full-history)
# =============================================

def get_doctor_metrics(
    doctor_id: int,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get aggregated metrics for a doctor including severity and category breakdowns.
    Uses APP_IncidentCaseDoctor join to find related incidents.
    
    Args:
        doctor_id: Doctor ID
        from_date: Optional start date filter (YYYY-MM-DD)
        to_date: Optional end date filter (YYYY-MM-DD)
    
    Returns:
        Dict with total_incidents, last_incident_date, severity_breakdown, category_breakdown
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        conditions = ["icd.DoctorID = ?"]
        params = [doctor_id]
        
        if from_date:
            conditions.append("CONVERT(DATE, ic.FeedbackRecievedDate) >= ?")
            params.append(from_date)
        
        if to_date:
            conditions.append("CONVERT(DATE, ic.FeedbackRecievedDate) <= ?")
            params.append(to_date)
        
        where_clause = " AND ".join(conditions)
        
        # Get total incidents and last incident date
        cursor.execute(f"""
            SELECT 
                COUNT(DISTINCT ic.IncidentRequestCaseID) as total_incidents,
                MAX(CONVERT(VARCHAR(10), ic.FeedbackRecievedDate, 23)) as last_incident_date
            FROM dbo.APP_IncidentCase ic
            INNER JOIN dbo.APP_IncidentCaseDoctor icd ON ic.IncidentRequestCaseID = icd.IncidentRequestCaseID
            WHERE {where_clause}
        """, params)
        
        summary_row = cursor.fetchone()
        total_incidents = summary_row[0] if summary_row else 0
        last_incident_date = summary_row[1] if summary_row else None
        
        # Get severity breakdown (complaints only — severity is NULL for notices)
        cursor.execute(f"""
            SELECT
                COALESCE(sev.SeverityName, 'Unknown') as severity,
                COUNT(DISTINCT ic.IncidentRequestCaseID) as count
            FROM dbo.APP_IncidentCase ic
            INNER JOIN dbo.APP_IncidentCaseDoctor icd ON ic.IncidentRequestCaseID = icd.IncidentRequestCaseID
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev ON ic.SeverityID = sev.SeverityID
            WHERE {where_clause} AND ic.RecordTypeID = 1
            GROUP BY sev.SeverityName
        """, params)

        severity_breakdown = {}
        for row in cursor.fetchall():
            severity_breakdown[row[0]] = row[1]

        # Get category breakdown (complaints only — category is NULL for notices)
        cursor.execute(f"""
            SELECT
                COALESCE(cat.CategoryName, 'Unknown') as category,
                COUNT(DISTINCT ic.IncidentRequestCaseID) as count
            FROM dbo.APP_IncidentCase ic
            INNER JOIN dbo.APP_IncidentCaseDoctor icd ON ic.IncidentRequestCaseID = icd.IncidentRequestCaseID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat ON ic.CategoryID = cat.CategoryID
            WHERE {where_clause} AND ic.RecordTypeID = 1
            GROUP BY cat.CategoryName
        """, params)

        category_breakdown = {}
        for row in cursor.fetchall():
            category_breakdown[row[0]] = row[1]

        # Get notice count (additive — callers that don't read it are unaffected)
        cursor.execute(f"""
            SELECT COUNT(DISTINCT ic.IncidentRequestCaseID)
            FROM dbo.APP_IncidentCase ic
            INNER JOIN dbo.APP_IncidentCaseDoctor icd ON ic.IncidentRequestCaseID = icd.IncidentRequestCaseID
            WHERE {where_clause} AND ic.RecordTypeID = 2
        """, params)
        notice_row = cursor.fetchone()
        total_notices = notice_row[0] if notice_row else 0

        return {
            "total_incidents": total_incidents,
            "total_notices": total_notices,
            "last_incident_date": last_incident_date,
            "severity_breakdown": severity_breakdown,
            "category_breakdown": category_breakdown
        }
    
    finally:
        cursor.close()
        conn.close()


# =============================================
# DOCTOR INCIDENTS FOR EXPORT
# =============================================

def get_doctor_incidents_for_export(
    doctor_id: int,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    include_profile: bool = True
) -> Dict[str, Any]:
    """
    Get doctor data formatted for export (CSV/JSON/Word).
    
    Args:
        doctor_id: Doctor ID
        from_date: Optional start date filter (YYYY-MM-DD)
        to_date: Optional end date filter (YYYY-MM-DD)
        include_profile: Include doctor profile in export
    
    Returns:
        Dict with doctor profile and incidents list
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        export_data = {
            "export_date": datetime.now().isoformat(),
            "format": "json",
            "doctor": None,
            "incidents": []
        }
        
        # Get doctor profile if requested
        if include_profile:
            profile = get_doctor_profile(doctor_id)
            if profile:
                export_data["doctor"] = {
                    "doctor_id": profile.get('id'),
                    "full_name": profile.get('name_en'),
                    "specialty": profile.get('specialty'),
                    "status": profile.get('status'),
                    "source": profile.get('source'),
                    "total_incidents": 0  # Will be computed below
                }
        
        # Build WHERE clause
        conditions = ["icd.DoctorID = ?"]
        params = [doctor_id]
        
        if from_date:
            conditions.append("CONVERT(DATE, ic.FeedbackRecievedDate) >= ?")
            params.append(from_date)
        
        if to_date:
            conditions.append("CONVERT(DATE, ic.FeedbackRecievedDate) <= ?")
            params.append(to_date)
        
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
                ic.ComplaintText as Description
            FROM dbo.APP_IncidentCase ic
            INNER JOIN dbo.APP_IncidentCaseDoctor icd ON ic.IncidentRequestCaseID = icd.IncidentRequestCaseID
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
        if export_data["doctor"]:
            export_data["doctor"]["total_incidents"] = len(export_data["incidents"])
        
        return export_data
    
    finally:
        cursor.close()
        conn.close()


# =============================================
# CREATE DOCTOR (RESERVE TABLE)
# =============================================

def create_doctor(
    doctor_name: str,
    specialty: Optional[str] = None,
    is_active: bool = True,
    source_system: str = 'MANUAL'
) -> Dict[str, Any]:
    """
    Create a new doctor in the reserve table (APP_RESERVE_DOCTOR).
    
    This function ONLY writes to the reserve table. It does NOT touch
    the hospital's view (APP_LOOKUP_DOCTOR).
    
    Reserve table structure is IDENTICAL to hospital table:
    - DoctorID (auto-increment primary key)
    - DoctorName (required)
    - Specialty (optional)
    - IsActive (bit: 1=active, 0=inactive)
    - SourceSystem (defaults to 'MANUAL')
    - LastSyncedAt (auto-set to current datetime)
    
    Args:
        doctor_name: Doctor's full name (required)
        specialty: Medical specialty (optional)
        is_active: Active status (default: True)
        source_system: Source identifier (default: 'MANUAL')
    
    Returns:
        Dict with created doctor data including generated DoctorID
        
    Raises:
        ValueError: If doctor_name already exists
        Exception: For other database errors
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # ============================================
        # STEP 1: Check for duplicate doctor name
        # ============================================
        # Check if doctor with same name already exists in reserve table
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM dbo.APP_RESERVE_DOCTOR 
            WHERE DoctorName = ?
        """, (doctor_name,))
        
        reserve_count = cursor.fetchone()[0]
        
        if reserve_count > 0:
            raise ValueError(
                f"Doctor name '{doctor_name}' already exists in reserve table. "
                "Doctor profile already created."
            )
        
        # ============================================
        # STEP 2: Insert into reserve table
        # ============================================
        
        insert_query = """
            INSERT INTO dbo.APP_RESERVE_DOCTOR (
                DoctorName,
                Specialty,
                IsActive,
                SourceSystem,
                LastSyncedAt
            )
            OUTPUT INSERTED.DoctorID,
                   INSERTED.DoctorName,
                   INSERTED.Specialty,
                   INSERTED.IsActive,
                   INSERTED.SourceSystem,
                   INSERTED.LastSyncedAt
            VALUES (?, ?, ?, ?, GETDATE())
        """
        
        cursor.execute(insert_query, (
            doctor_name,
            specialty,
            1 if is_active else 0,
            source_system
        ))
        
        # Fetch the inserted record
        row = cursor.fetchone()
        conn.commit()
        
        # ============================================
        # STEP 3: Format and return result
        # ============================================
        
        result = {
            'id': row.DoctorID,
            'name_en': row.DoctorName,
            'name_ar': row.DoctorName,
            'specialty': row.Specialty if row.Specialty else '',
            'status': 'active' if row.IsActive else 'inactive',
            'source': 'reserve',
            'source_system': row.SourceSystem if row.SourceSystem else '',
            'last_synced_at': row.LastSyncedAt.strftime('%Y-%m-%d %H:%M:%S') if row.LastSyncedAt else ''
        }
        
        return result
        
    except ValueError:
        # Re-raise validation errors
        raise
    except Exception as e:
        conn.rollback()
        raise Exception(f"Failed to create doctor: {str(e)}")
    finally:
        cursor.close()
        conn.close()


# =============================================
# DELETE DOCTOR (RESERVE TABLE)
# =============================================

def count_incidents_by_doctor(doctor_id: int) -> int:
    """
    Count the number of incidents associated with a doctor.
    
    Checks the APP_IncidentCaseDoctor table for linked incidents.
    
    Args:
        doctor_id: The doctor's ID to check
        
    Returns:
        Count of incidents linked to this doctor
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT COUNT(*) 
            FROM dbo.APP_IncidentCaseDoctor 
            WHERE DoctorID = ?
        """, (doctor_id,))
        
        count = cursor.fetchone()[0]
        return count
        
    finally:
        cursor.close()
        conn.close()


def get_reserve_doctor_by_id(doctor_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a reserve doctor by ID.
    
    Args:
        doctor_id: The doctor's ID
        
    Returns:
        Dict with doctor data or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                DoctorID,
                DoctorName,
                Specialty,
                IsActive,
                SourceSystem,
                LastSyncedAt
            FROM dbo.APP_RESERVE_DOCTOR
            WHERE DoctorID = ?
        """, (doctor_id,))
        
        row = cursor.fetchone()
        
        if row:
            return {
                'id': row.DoctorID,
                'name_en': row.DoctorName,
                'name_ar': row.DoctorName,
                'specialty': row.Specialty if row.Specialty else '',
                'is_active': bool(row.IsActive),
                'source_system': row.SourceSystem if row.SourceSystem else '',
                'last_synced_at': row.LastSyncedAt.strftime('%Y-%m-%d %H:%M:%S') if row.LastSyncedAt else ''
            }
        return None
        
    finally:
        cursor.close()
        conn.close()


def deactivate_reserve_doctor(doctor_id: int) -> bool:
    """
    Soft-delete a reserve doctor by setting IsActive = 0.
    
    Args:
        doctor_id: The doctor's ID to deactivate
        
    Returns:
        True if successful, False if doctor not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            UPDATE dbo.APP_RESERVE_DOCTOR
            SET IsActive = 0, LastSyncedAt = GETDATE()
            WHERE DoctorID = ?
        """, (doctor_id,))
        
        rows_affected = cursor.rowcount
        conn.commit()
        
        return rows_affected > 0
        
    except Exception as e:
        conn.rollback()
        raise Exception(f"Failed to deactivate doctor: {str(e)}")
    finally:
        cursor.close()
        conn.close()


def hard_delete_reserve_doctor(doctor_id: int) -> bool:
    """
    Permanently delete a reserve doctor from the table.
    
    Only call this if the doctor has no associated incidents.
    
    Args:
        doctor_id: The doctor's ID to delete
        
    Returns:
        True if successful, False if doctor not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            DELETE FROM dbo.APP_RESERVE_DOCTOR
            WHERE DoctorID = ?
        """, (doctor_id,))
        
        rows_affected = cursor.rowcount
        conn.commit()
        
        return rows_affected > 0
        
    except Exception as e:
        conn.rollback()
        raise Exception(f"Failed to delete doctor: {str(e)}")
    finally:
        cursor.close()
        conn.close()
