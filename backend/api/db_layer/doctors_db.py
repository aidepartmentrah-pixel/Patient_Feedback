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

def get_reserve_doctors(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get all doctors from the reserve table only.
    
    Returns only user-created doctors (not from hospital system).
    
    Args:
        limit: Max results
    
    Returns:
        List of reserve doctor records
    """
    conn = get_connection()
    cursor = conn.cursor()
    
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
        
        # Not found in reserve, try hospital table
        query_hospital = """
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
