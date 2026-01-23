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


# ==================== CREATE PATIENT (RESERVE TABLE) ====================

def create_patient(
    first_name: str,
    middle_name: Optional[str] = None,
    last_name: Optional[str] = None,
    mother_name: Optional[str] = None,
    phone_number: Optional[str] = None,
    birth_date: Optional[str] = None,
    sex: Optional[str] = None,
    document_number: Optional[str] = None,
    medical_file_number: Optional[str] = None,
    spouse: Optional[str] = None,
    address_line1: Optional[str] = None,
    address_line2: Optional[str] = None,
    phone_number2: Optional[str] = None,
    created_by_user_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a new patient in the APP_RESERVE_PATIENT table.
    
    This function writes ONLY to the reserve table (user-created patients).
    Hospital patients come from APP_VIEWTABLE_PATIENT_ADMISSION (read-only).
    
    Args:
        first_name: Patient's first name (REQUIRED)
        middle_name: Patient's middle name (optional)
        last_name: Patient's last name (optional)
        mother_name: Patient's mother name (optional)
        phone_number: Primary phone number (optional)
        birth_date: Birth date as string YYYY-MM-DD (optional)
        sex: Gender M/F (optional)
        document_number: National ID or document number (optional)
        medical_file_number: Medical record number (optional)
        spouse: Spouse name (optional)
        address_line1: Primary address (optional)
        address_line2: Secondary address (optional)
        phone_number2: Secondary phone number (optional)
        created_by_user_id: User ID who created this patient (optional)
    
    Returns:
        Dict with created patient data including PatientAdmissionID
    
    Raises:
        ValueError: If validation fails or duplicate detected
        Exception: If database operation fails
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Build full name from components
        name_parts = [
            first_name or '',
            middle_name or '',
            last_name or ''
        ]
        full_name = ' '.join(part for part in name_parts if part).strip()
        
        if not full_name:
            raise ValueError("At least FirstName must be provided to build FullName")
        
        # Check for duplicate in RESERVE table only
        # We check by FullName as the primary duplicate detection
        cursor.execute("""
            SELECT PatientAdmissionID, FullName 
            FROM APP_RESERVE_PATIENT 
            WHERE FullName = ?
        """, (full_name,))
        
        existing = cursor.fetchone()
        if existing:
            raise ValueError(
                f"Patient with name '{full_name}' already exists in reserve table "
                f"(ID: {existing[0]}). Cannot create duplicate."
            )
        
        # Additional check by DocumentNumber if provided
        if document_number:
            cursor.execute("""
                SELECT PatientAdmissionID, FullName, DocumentNumber
                FROM APP_RESERVE_PATIENT 
                WHERE DocumentNumber = ?
            """, (document_number,))
            
            existing_doc = cursor.fetchone()
            if existing_doc:
                raise ValueError(
                    f"Patient with DocumentNumber '{document_number}' already exists "
                    f"(ID: {existing_doc[0]}, Name: {existing_doc[1]}). Cannot create duplicate."
                )
        
        # Additional check by MedicalFileNumber if provided
        if medical_file_number:
            cursor.execute("""
                SELECT PatientAdmissionID, FullName, MedicalFileNumber
                FROM APP_RESERVE_PATIENT 
                WHERE MedicalFileNumber = ?
            """, (medical_file_number,))
            
            existing_mrn = cursor.fetchone()
            if existing_mrn:
                raise ValueError(
                    f"Patient with MedicalFileNumber '{medical_file_number}' already exists "
                    f"(ID: {existing_mrn[0]}, Name: {existing_mrn[1]}). Cannot create duplicate."
                )
        
        # Insert into APP_RESERVE_PATIENT
        cursor.execute("""
            INSERT INTO APP_RESERVE_PATIENT (
                FirstName, 
                MiddleName, 
                LastName, 
                MotherName, 
                FullName,
                PhoneNumber1, 
                PhoneNumber2,
                BirthDate, 
                SEX, 
                DocumentNumber, 
                MedicalFileNumber,
                Spouse,
                AddressLine1,
                AddressLine2,
                SystemTime
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
        """, (
            first_name,
            middle_name,
            last_name,
            mother_name,
            full_name,
            phone_number,
            phone_number2,
            birth_date,
            sex,
            document_number,
            medical_file_number,
            spouse,
            address_line1,
            address_line2
        ))
        
        # Get the new PatientAdmissionID
        cursor.execute("SELECT @@IDENTITY")
        new_id = int(cursor.fetchone()[0])
        
        # Commit transaction
        conn.commit()
        
        # Return created patient data
        return {
            "PatientAdmissionID": new_id,
            "FullName": full_name,
            "FirstName": first_name,
            "MiddleName": middle_name,
            "LastName": last_name,
            "MotherName": mother_name,
            "PhoneNumber1": phone_number,
            "PhoneNumber2": phone_number2,
            "BirthDate": birth_date,
            "SEX": sex,
            "DocumentNumber": document_number,
            "MedicalFileNumber": medical_file_number,
            "Spouse": spouse,
            "AddressLine1": address_line1,
            "AddressLine2": address_line2,
            "Source": "reserve",
            "CreatedAt": datetime.now().isoformat()
        }
        
    except ValueError as ve:
        # Re-raise validation errors
        if conn:
            conn.rollback()
        raise ve
    except Exception as e:
        # Rollback on any error
        if conn:
            conn.rollback()
        raise Exception(f"Failed to create patient in database: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ==================== GET ALL RESERVE PATIENTS ====================

def get_all_reserve_patients(
    limit: int = 100,
    offset: int = 0,
    order_by: str = 'SystemTime'
) -> Dict[str, Any]:
    """
    Get all patients from the reserve table (APP_RESERVE_PATIENT).
    
    This function retrieves ONLY user-created patients from the reserve table,
    not hospital patients.
    
    Args:
        limit: Maximum number of records to return (default: 100)
        offset: Number of records to skip for pagination (default: 0)
        order_by: Field to order by - 'SystemTime' (newest first) or 'FullName' (alphabetical)
    
    Returns:
        Dict with:
            - patients: List of patient records
            - total: Total count of reserve patients
            - limit: Applied limit
            - offset: Applied offset
    
    Raises:
        Exception: If database operation fails
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Determine order clause
        if order_by == 'FullName':
            order_clause = "ORDER BY FullName ASC"
        else:
            order_clause = "ORDER BY SystemTime DESC"  # Default: newest first
        
        # Get total count first
        cursor.execute("SELECT COUNT(*) FROM APP_RESERVE_PATIENT")
        total_count = cursor.fetchone()[0]
        
        # Get paginated results
        query = f"""
            SELECT 
                PatientAdmissionID,
                FullName,
                FirstName,
                MiddleName,
                LastName,
                MotherName,
                PhoneNumber1,
                PhoneNumber2,
                CONVERT(VARCHAR(10), BirthDate, 23) as BirthDate,
                SEX,
                DocumentNumber,
                MedicalFileNumber,
                Spouse,
                AddressLine1,
                AddressLine2,
                CONVERT(VARCHAR(19), SystemTime, 121) as CreatedAt
            FROM APP_RESERVE_PATIENT
            {order_clause}
            OFFSET ? ROWS
            FETCH NEXT ? ROWS ONLY
        """
        
        cursor.execute(query, (offset, limit))
        
        patients = []
        for row in cursor.fetchall():
            patients.append({
                "patient_admission_id": row.PatientAdmissionID,
                "full_name": row.FullName,
                "first_name": row.FirstName,
                "middle_name": row.MiddleName,
                "last_name": row.LastName,
                "mother_name": row.MotherName,
                "phone_number": row.PhoneNumber1,
                "phone_number2": row.PhoneNumber2,
                "birth_date": row.BirthDate,
                "sex": row.SEX,
                "document_number": row.DocumentNumber,
                "medical_file_number": row.MedicalFileNumber,
                "spouse": row.Spouse,
                "address_line1": row.AddressLine1,
                "address_line2": row.AddressLine2,
                "created_at": row.CreatedAt,
                "source": "reserve"
            })
        
        return {
            "patients": patients,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "count": len(patients)
        }
        
    except Exception as e:
        raise Exception(f"Failed to retrieve reserve patients: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


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
    Uses UNION to search both hospital (APP_VIEWTABLE_PATIENT_ADMISSION) 
    and reserve (APP_RESERVE_PATIENT) tables.
    
    Args:
        query: Partial match on patient name
        mrn: Match on MedicalFileNumber
        phone: Partial match on phone number
        date_of_birth: Exact match on date of birth (YYYY-MM-DD)
        limit: Max results to return
    
    Returns:
        List of patient search results with lightweight fields
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        conditions_hospital = []
        conditions_reserve = []
        params_hospital = []
        params_reserve = []
        
        # Build dynamic WHERE clauses for both sources
        if query:
            conditions_hospital.append("(FullName LIKE ? OR FirstName LIKE ? OR LastName LIKE ?)")
            params_hospital.extend([f"%{query}%", f"%{query}%", f"%{query}%"])
            conditions_reserve.append("(FullName LIKE ? OR FirstName LIKE ? OR LastName LIKE ?)")
            params_reserve.extend([f"%{query}%", f"%{query}%", f"%{query}%"])
        
        if mrn:
            conditions_hospital.append("MedicalFileNumber LIKE ?")
            params_hospital.append(f"%{mrn}%")
            conditions_reserve.append("MedicalFileNumber LIKE ?")
            params_reserve.append(f"%{mrn}%")
        
        if phone:
            conditions_hospital.append("PhoneNumber1 LIKE ?")
            params_hospital.append(f"%{phone}%")
            conditions_reserve.append("PhoneNumber1 LIKE ?")
            params_reserve.append(f"%{phone}%")
        
        if date_of_birth:
            conditions_hospital.append("CONVERT(DATE, BirthDate) = ?")
            params_hospital.append(date_of_birth)
            conditions_reserve.append("CONVERT(DATE, BirthDate) = ?")
            params_reserve.append(date_of_birth)
        
        where_hospital = " AND ".join(conditions_hospital) if conditions_hospital else "1=1"
        where_reserve = " AND ".join(conditions_reserve) if conditions_reserve else "1=1"
        
        # UNION query combining both sources
        query_str = f"""
            SELECT TOP {limit} * FROM (
                -- Hospital patients
                SELECT
                    PatientAdmissionID as patient_id,
                    MedicalFileNumber as mrn,
                    FullName as patient_name,
                    FirstName as first_name,
                    LastName as last_name,
                    CONVERT(VARCHAR(10), BirthDate, 23) as date_of_birth,
                    DATEDIFF(YEAR, BirthDate, GETDATE()) as age,
                    SEX as gender,
                    PhoneNumber1 as phone,
                    'hospital' as source
                FROM dbo.APP_VIEWTABLE_PATIENT_ADMISSION
                WHERE {where_hospital}
                
                UNION ALL
                
                -- Reserve patients
                SELECT
                    PatientAdmissionID as patient_id,
                    MedicalFileNumber as mrn,
                    FullName as patient_name,
                    FirstName as first_name,
                    LastName as last_name,
                    CONVERT(VARCHAR(10), BirthDate, 23) as date_of_birth,
                    DATEDIFF(YEAR, BirthDate, GETDATE()) as age,
                    SEX as gender,
                    PhoneNumber1 as phone,
                    'reserve' as source
                FROM dbo.APP_RESERVE_PATIENT
                WHERE {where_reserve}
            ) AS CombinedPatients
            ORDER BY patient_name ASC
        """
        
        # Combine parameters for both queries
        all_params = params_hospital + params_reserve
        
        cursor.execute(query_str, all_params)
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
