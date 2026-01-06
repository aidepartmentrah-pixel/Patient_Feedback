"""
Patients Database Layer
Queries patient information and related incidents from SQL Server.
"""

import pyodbc
from datetime import datetime
from typing import List, Dict, Any, Optional


def get_connection():
    """Get SQL Server connection."""
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn


# ==================== SEARCH PATIENTS ====================

def search_patients(
    query: Optional[str] = None,
    mrn: Optional[str] = None,
    phone: Optional[str] = None,
    date_of_birth: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Search for patients by name, MRN, phone, or date of birth.
    
    Args:
        query: Partial match on patient name
        mrn: Exact match on Medical Record Number
        phone: Partial match on phone number
        date_of_birth: Exact match on date of birth (YYYY-MM-DD)
        limit: Max results to return
    
    Returns:
        List of patient search results with lightweight fields
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        conditions = []
        params = []
        
        # Build dynamic WHERE clause
        if query:
            conditions.append("(PatientName LIKE ? OR PatientNameEnglish LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        
        if mrn:
            conditions.append("MRN = ?")
            params.append(mrn)
        
        if phone:
            conditions.append("Phone LIKE ?")
            params.append(f"%{phone}%")
        
        if date_of_birth:
            conditions.append("CONVERT(DATE, DateOfBirth) = ?")
            params.append(date_of_birth)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query_str = f"""
            SELECT TOP {limit}
                PatientID,
                MRN,
                PatientName,
                PatientNameEnglish,
                CONVERT(VARCHAR(10), DateOfBirth, 23) as DateOfBirth,
                DATEDIFF(YEAR, DateOfBirth, GETDATE()) as Age,
                Gender,
                Phone
            FROM dbo.APP_Patient
            WHERE {where_clause}
            ORDER BY PatientName ASC
        """
        
        cursor.execute(query_str, params)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        
        return [dict(zip(columns, row)) for row in rows]
    
    finally:
        conn.close()


# ==================== GET PATIENT PROFILE ====================

def get_patient_profile(patient_id: int) -> Optional[Dict[str, Any]]:
    """
    Get complete patient profile information.
    
    Args:
        patient_id: Patient unique identifier
    
    Returns:
        Patient profile dict or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT
                PatientID,
                MRN,
                PatientName,
                PatientNameEnglish,
                CONVERT(VARCHAR(10), DateOfBirth, 23) as DateOfBirth,
                DATEDIFF(YEAR, DateOfBirth, GETDATE()) as Age,
                Gender,
                Nationality,
                Phone,
                Email,
                Address,
                EmergencyContact,
                EmergencyPhone,
                CONVERT(VARCHAR(19), RegistrationDate, 121) as RegistrationDate
            FROM dbo.APP_Patient
            WHERE PatientID = ?
        """, patient_id)
        
        row = cursor.fetchone()
        if not row:
            return None
        
        columns = [col[0] for col in cursor.description]
        profile = dict(zip(columns, row))
        
        # Get total incidents count and last visit date
        cursor.execute("""
            SELECT 
                COUNT(*) as TotalIncidents,
                MAX(CONVERT(VARCHAR(10), FeedbackRecievedDate, 23)) as LastVisitDate
            FROM dbo.APP_IncidentCase
            WHERE PatientName = ? OR PatientID = ?
        """, profile['PatientName'], patient_id)
        
        incident_row = cursor.fetchone()
        if incident_row:
            profile['TotalIncidents'] = incident_row[0] or 0
            profile['LastVisitDate'] = incident_row[1]
        else:
            profile['TotalIncidents'] = 0
            profile['LastVisitDate'] = None
        
        return profile
    
    finally:
        conn.close()


# ==================== GET PATIENT INCIDENTS ====================

def get_patient_incidents(
    patient_id: int,
    patient_name: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    department: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Get all feedback/incident records for a patient.
    
    Args:
        patient_id: Patient unique identifier
        patient_name: Patient name (alternative search method)
        from_date: Filter from date (YYYY-MM-DD)
        to_date: Filter to date (YYYY-MM-DD)
        department: Filter by department name
        severity: Filter by severity (High, Medium, Low)
        status: Filter by status
        limit: Max results per page
        offset: Pagination offset
    
    Returns:
        Dict with incidents list and pagination info
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        conditions = ["(PatientName = ? OR PatientID = ?)"]
        params = [patient_name or "", patient_id]
        
        if from_date:
            conditions.append("CONVERT(DATE, FeedbackRecievedDate) >= ?")
            params.append(from_date)
        
        if to_date:
            conditions.append("CONVERT(DATE, FeedbackRecievedDate) <= ?")
            params.append(to_date)
        
        if department:
            conditions.append("OrgUnitName LIKE ?")
            params.append(f"%{department}%")
        
        if severity:
            conditions.append("SeverityName = ?")
            params.append(severity)
        
        if status:
            conditions.append("CaseStatusName = ?")
            params.append(status)
        
        where_clause = " AND ".join(conditions)
        
        # Get total count
        count_query = f"""
            SELECT COUNT(*)
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.APP_OrgUnit ou ON ic.IssuingOrgUnitID = ou.OrgUnitID
            LEFT JOIN dbo.APP_Severity sev ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_CaseStatus cs ON ic.CaseStatusID = cs.CaseStatusID
            WHERE {where_clause}
        """
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]
        
        # Get incidents with pagination
        incidents_query = f"""
            SELECT
                ic.IncidentRequestCaseID as IncidentID,
                ic.IncidentRequestCaseID as RecordID,
                CONVERT(VARCHAR(10), ic.CreatedAt, 23) as Date,
                CONVERT(VARCHAR(10), ic.FeedbackRecievedDate, 23) as FeedbackReceivedDate,
                COALESCE(ou.OrgUnitNameEN, 'Unknown') as Department,
                COALESCE(ou.OrgUnitName, 'غير محدد') as DepartmentAr,
                COALESCE(cat.CategoryNameEN, 'Unknown') as Category,
                COALESCE(cat.CategoryName, 'غير محدد') as CategoryAr,
                COALESCE(sev.SeverityName, 'Unknown') as Severity,
                ic.PatientName as DoctorName,
                COALESCE(cs.CaseStatusName, 'Open') as Status,
                LEFT(ic.ComplaintText, 200) as Description,
                CASE WHEN ic.ClinicalRiskTypeID = 2 THEN 1 ELSE 0 END as IsRedFlag,
                CASE WHEN ic.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END as IsNeverEvent
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.APP_OrgUnit ou ON ic.IssuingOrgUnitID = ou.OrgUnitID
            LEFT JOIN dbo.APP_Category cat ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_Severity sev ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_CaseStatus cs ON ic.CaseStatusID = cs.CaseStatusID
            WHERE {where_clause}
            ORDER BY ic.FeedbackRecievedDate DESC
            OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY
        """
        cursor.execute(incidents_query, params)
        columns = [col[0] for col in cursor.description]
        incidents = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Get patient name for response
        patient_profile = get_patient_profile(patient_id)
        patient_name_result = patient_profile['PatientName'] if patient_profile else "Unknown"
        
        return {
            "patient_id": patient_id,
            "patient_name": patient_name_result,
            "incidents": incidents,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    
    finally:
        conn.close()


# ==================== GET INCIDENT DETAILS ====================

def get_incident_details(patient_id: int, incident_id: int) -> Optional[Dict[str, Any]]:
    """
    Get full details for a specific incident.
    
    Args:
        patient_id: Patient unique identifier (for validation)
        incident_id: Incident unique identifier
    
    Returns:
        Full incident details dict or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT
                ic.IncidentRequestCaseID as IncidentID,
                ic.IncidentRequestCaseID as RecordID,
                CONVERT(VARCHAR(10), ic.CreatedAt, 23) as Date,
                CONVERT(VARCHAR(10), ic.FeedbackRecievedDate, 23) as FeedbackReceivedDate,
                ic.PatientID,
                ic.PatientName,
                COALESCE(ou.OrgUnitNameEN, 'Unknown') as Department,
                COALESCE(ou_target.OrgUnitNameEN, 'Unknown') as TargetDepartment,
                COALESCE(cat.CategoryNameEN, 'Unknown') as Category,
                COALESCE(cat.CategoryName, 'غير محدد') as CategoryAr,
                CONCAT(
                    COALESCE(dom.DomainNameEN, ''),
                    ' > ',
                    COALESCE(cat.CategoryNameEN, ''),
                    ' > ',
                    COALESCE(subcat.SubCategoryNameEN, '')
                ) as Classification,
                COALESCE(sev.SeverityName, 'Unknown') as Severity,
                COALESCE(hl.HarmLevelName, 'Unknown') as HarmLevel,
                COALESCE(st.StageName, 'Unknown') as Stage,
                ic.PatientName as DoctorName,
                COALESCE(cs.CaseStatusName, 'Open') as Status,
                ic.ComplaintText,
                ic.ImmediateAction,
                ic.TakenAction,
                CASE WHEN ic.ClinicalRiskTypeID = 2 THEN 1 ELSE 0 END as IsRedFlag,
                CASE WHEN ic.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END as IsNeverEvent,
                CONVERT(VARCHAR(19), ic.CreatedAt, 121) as CreatedAt,
                CONVERT(VARCHAR(19), ic.UpdatedAt, 121) as LastUpdatedAt
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.APP_OrgUnit ou ON ic.IssuingOrgUnitID = ou.OrgUnitID
            LEFT JOIN dbo.APP_OrgUnit ou_target ON ic.TargetOrgUnitID = ou_target.OrgUnitID
            LEFT JOIN dbo.APP_Category cat ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_Domain dom ON ic.DomainID = dom.DomainID
            LEFT JOIN dbo.APP_SubCategory subcat ON ic.SubCategoryID = subcat.SubCategoryID
            LEFT JOIN dbo.APP_Severity sev ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_HarmLevel hl ON ic.HarmLevelID = hl.HarmLevelID
            LEFT JOIN dbo.APP_Stage st ON ic.StageID = st.StageID
            LEFT JOIN dbo.APP_CaseStatus cs ON ic.CaseStatusID = cs.CaseStatusID
            WHERE ic.IncidentRequestCaseID = ? AND (ic.PatientID = ? OR ic.PatientName IN (
                SELECT PatientName FROM dbo.APP_Patient WHERE PatientID = ?
            ))
        """, incident_id, patient_id, patient_id)
        
        row = cursor.fetchone()
        if not row:
            return None
        
        columns = [col[0] for col in cursor.description]
        return dict(zip(columns, row))
    
    finally:
        conn.close()


# ==================== EXPORT PATIENT HISTORY ====================

def get_patient_incidents_for_export(
    patient_id: int,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    include_profile: bool = True
) -> Dict[str, Any]:
    """
    Get patient data for export in CSV or JSON format.
    
    Args:
        patient_id: Patient unique identifier
        from_date: Filter from date (YYYY-MM-DD)
        to_date: Filter to date (YYYY-MM-DD)
        include_profile: Include patient profile in export
    
    Returns:
        Dict with patient profile and incidents
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        export_data = {
            "export_date": datetime.now().isoformat(),
            "format": "json",
            "patient": None,
            "incidents": []
        }
        
        # Get patient profile if requested
        if include_profile:
            profile = get_patient_profile(patient_id)
            if profile:
                export_data["patient"] = {
                    "patient_id": profile['PatientID'],
                    "mrn": profile['MRN'],
                    "full_name": profile['PatientName'],
                    "total_incidents": profile['TotalIncidents']
                }
        
        # Get incidents for export
        conditions = ["(ic.PatientID = ? OR ic.PatientName IN (SELECT PatientName FROM dbo.APP_Patient WHERE PatientID = ?))"]
        params = [patient_id, patient_id]
        
        if from_date:
            conditions.append("CONVERT(DATE, ic.FeedbackRecievedDate) >= ?")
            params.append(from_date)
        
        if to_date:
            conditions.append("CONVERT(DATE, ic.FeedbackRecievedDate) <= ?")
            params.append(to_date)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT
                ic.IncidentRequestCaseID as RecordID,
                CONVERT(VARCHAR(10), ic.CreatedAt, 23) as Date,
                COALESCE(ou.OrgUnitNameEN, 'Unknown') as Department,
                COALESCE(cat.CategoryNameEN, 'Unknown') as Category,
                COALESCE(sev.SeverityName, 'Unknown') as Severity,
                ic.PatientName as DoctorName,
                COALESCE(cs.CaseStatusName, 'Open') as Status,
                ic.ComplaintText,
                ic.ImmediateAction,
                ic.TakenAction,
                CONCAT(
                    COALESCE(dom.DomainNameEN, ''),
                    ' > ',
                    COALESCE(cat.CategoryNameEN, ''),
                    ' > ',
                    COALESCE(subcat.SubCategoryNameEN, '')
                ) as Classification,
                COALESCE(hl.HarmLevelName, 'Unknown') as HarmLevel,
                COALESCE(st.StageName, 'Unknown') as Stage,
                COALESCE(ou_target.OrgUnitNameEN, 'Unknown') as TargetDepartment
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.APP_OrgUnit ou ON ic.IssuingOrgUnitID = ou.OrgUnitID
            LEFT JOIN dbo.APP_OrgUnit ou_target ON ic.TargetOrgUnitID = ou_target.OrgUnitID
            LEFT JOIN dbo.APP_Category cat ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_Domain dom ON ic.DomainID = dom.DomainID
            LEFT JOIN dbo.APP_SubCategory subcat ON ic.SubCategoryID = subcat.SubCategoryID
            LEFT JOIN dbo.APP_Severity sev ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_HarmLevel hl ON ic.HarmLevelID = hl.HarmLevelID
            LEFT JOIN dbo.APP_Stage st ON ic.StageID = st.StageID
            LEFT JOIN dbo.APP_CaseStatus cs ON ic.CaseStatusID = cs.CaseStatusID
            WHERE {where_clause}
            ORDER BY ic.FeedbackRecievedDate DESC
        """
        
        cursor.execute(query, params)
        columns = [col[0] for col in cursor.description]
        export_data["incidents"] = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return export_data
    
    finally:
        conn.close()
