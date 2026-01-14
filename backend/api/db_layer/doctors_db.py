"""
Database layer for Doctor endpoints.
Handles all SQL queries for doctor profiles, statistics, and incident tracking.
"""

import pyodbc
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


def get_connection():
    """Get database connection."""
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn


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
    Search for doctors by name, employee ID, or department.
    
    Args:
        query: Search term (name or employee ID)
        department: Filter by department
        status: Filter by status (active, inactive, suspended)
        limit: Max results
    
    Returns:
        List of doctor records
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    where_parts = []
    params = []
    
    # Free-text search on name and employee ID
    if query:
        search_condition = """(
            d.DoctorName LIKE ? 
            OR d.EmployeeID LIKE ?
        )"""
        where_parts.append(search_condition)
        search_param = f"%{query}%"
        params.extend([search_param, search_param])
    
    # Department filter (if applicable)
    if department:
        where_parts.append("d.Department LIKE ?")
        params.append(f"%{department}%")
    
    # Status filter
    if status:
        status_map = {
            'active': 1,
            'inactive': 0,
            'suspended': 2
        }
        status_value = status_map.get(status.lower())
        if status_value is not None:
            where_parts.append("d.IsActive = ?")
            params.append(status_value)
    
    where_clause = " WHERE " + " AND ".join(where_parts) if where_parts else ""
    
    query_sql = f"""
        SELECT TOP {limit}
            d.DoctorID as id,
            d.EmployeeID as employee_id,
            d.DoctorName as name_en,
            d.DoctorName as name_ar,
            d.Department as department,
            d.Specialty as specialty,
            d.HireDate as hire_date,
            CASE WHEN d.IsActive = 1 THEN 'active'
                 WHEN d.IsActive = 2 THEN 'suspended'
                 ELSE 'inactive'
            END as status
        FROM dbo.APP_LOOKUP_DOCTOR d
        {where_clause}
        ORDER BY d.DoctorName ASC
    """
    
    try:
        cursor.execute(query_sql, params)
        columns = [col[0] for col in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return results
    finally:
        cursor.close()
        conn.close()


# =============================================
# DOCTOR PROFILE
# =============================================

def get_doctor_profile(doctor_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch detailed profile information for a doctor.
    
    Args:
        doctor_id: Doctor ID
    
    Returns:
        Doctor profile dict or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT
            d.DoctorID as id,
            d.EmployeeID as employee_id,
            d.DoctorName as name_en,
            d.DoctorName as name_ar,
            d.Department as department,
            d.Specialty as specialty,
            d.HireDate as hire_date,
            d.Email as email,
            d.Phone as phone,
            d.LicenseNumber as license_number,
            CASE WHEN d.IsActive = 1 THEN 'active'
                 WHEN d.IsActive = 2 THEN 'suspended'
                 ELSE 'inactive'
            END as status
        FROM dbo.APP_LOOKUP_DOCTOR d
        WHERE d.DoctorID = ?
    """
    
    try:
        cursor.execute(query, (doctor_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        columns = [col[0] for col in cursor.description]
        profile = dict(zip(columns, row))
        
        # Calculate years of service
        if profile.get('hire_date'):
            hire_date = profile['hire_date']
            today = datetime.now()
            years_of_service = (today - hire_date).days // 365
            profile['years_of_service'] = years_of_service
        
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
            COUNT(CASE WHEN crt.Code IN ('RED_FLAG', 'NEVER_EVENT') THEN 1 END) as red_flags
        FROM dbo.APP_IncidentCase ic
        INNER JOIN dbo.APP_IncidentCaseDoctor icd ON ic.IncidentRequestCaseID = icd.IncidentRequestCaseID
        LEFT JOIN dbo.APP_LOOKUP_SEVERITY s ON ic.SeverityID = s.SeverityID
        LEFT JOIN dbo.APP_LOOKUP_CLINICAL_RISK_TYPE crt ON ic.ClinicalRiskTypeID = crt.ClinicalRiskTypeID
        WHERE icd.DoctorID = ?
        AND ic.FeedbackRecievedDate >= ?
        AND ic.FeedbackRecievedDate <= ?
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
                'red_flags': 0
            }
        
        return {
            'total': row.total or 0,
            'high': row.high or 0,
            'medium': row.medium or 0,
            'low': row.low or 0,
            'red_flags': row.red_flags or 0
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
            cat.CategoryName as category,
            cat.CategoryName as category_ar,
            s.SeverityName as severity,
            cs.Name as status,
            CASE WHEN crt.Code IN ('RED_FLAG', 'NEVER_EVENT') THEN 1 ELSE 0 END as is_red_flag
        FROM dbo.APP_IncidentCase ic
        INNER JOIN dbo.APP_IncidentCaseDoctor icd ON ic.IncidentRequestCaseID = icd.IncidentRequestCaseID
        LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat ON ic.CategoryID = cat.CategoryID
        LEFT JOIN dbo.APP_LOOKUP_SEVERITY s ON ic.SeverityID = s.SeverityID
        LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs ON ic.CaseStatusID = cs.CaseStatusID
        LEFT JOIN dbo.APP_LOOKUP_CLINICAL_RISK_TYPE crt ON ic.ClinicalRiskTypeID = crt.ClinicalRiskTypeID
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
