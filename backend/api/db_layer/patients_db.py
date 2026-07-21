"""
Patients Database Layer
Queries patient information and related incidents from SQL Server.

SESSION C1 NOTE: this module is now RESERVE-ONLY (APP_RESERVE_PATIENT).
External (hospital-system) patient reads go through
core/hospital_directory_client.py + api/services/patient_directory_service.py
instead of the old dbo.VW_PatientAdmission view — this file no longer
imports PATIENT_ADMISSION_TABLE at all. Incident matching for a patient
(get_incident_stats_for_name) is keyed purely on the PatientName text
snapshot on APP_IncidentCase, which was already true regardless of whether
the patient came from the hospital view or the reserve table (see
Investigation 1 §5 — APP_IncidentCase has no patient FK, only free text).
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from core.database import get_connection


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
    order_by: str = 'SystemTime',
    include_inactive: bool = False
) -> Dict[str, Any]:
    """
    Get all patients from the reserve table (APP_RESERVE_PATIENT).
    
    This function retrieves ONLY user-created patients from the reserve table,
    not hospital patients. By default, excludes soft-deleted (IsActive=0) patients.
    
    Args:
        limit: Maximum number of records to return (default: 100)
        offset: Number of records to skip for pagination (default: 0)
        order_by: Field to order by - 'SystemTime' (newest first) or 'FullName' (alphabetical)
        include_inactive: If True, include soft-deleted patients (default: False)
    
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
        
        # Build WHERE clause for IsActive filter
        where_clause = "" if include_inactive else "WHERE IsActive = 1"
        
        # Get total count first
        count_query = f"SELECT COUNT(*) FROM APP_RESERVE_PATIENT {where_clause}"
        cursor.execute(count_query)
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
                CONVERT(VARCHAR(19), SystemTime, 121) as CreatedAt,
                IsActive
            FROM APP_RESERVE_PATIENT
            {where_clause}
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
                "is_active": bool(row.IsActive),
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
    Search RESERVE patients (APP_RESERVE_PATIENT) only, by name, MRN, phone,
    or date of birth.

    External (hospital-system) patients are no longer read from SQL Server —
    see api/services/patient_directory_service.py, which calls this function
    for the reserve half of a merged search and hospital_directory_client
    for the external half.

    Args:
        query: Partial match on patient name
        mrn: Match on MedicalFileNumber
        phone: Partial match on phone number
        date_of_birth: Exact match on date of birth (YYYY-MM-DD)
        limit: Max results to return

    Returns:
        List of patient search results with lightweight fields, source='reserve'
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        conditions = []
        params = []

        if query:
            conditions.append("(FullName LIKE ? OR FirstName LIKE ? OR LastName LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%", f"%{query}%"])

        if mrn:
            conditions.append("MedicalFileNumber LIKE ?")
            params.append(f"%{mrn}%")

        if phone:
            conditions.append("PhoneNumber1 LIKE ?")
            params.append(f"%{phone}%")

        if date_of_birth:
            conditions.append("CONVERT(DATE, BirthDate) = ?")
            params.append(date_of_birth)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query_str = f"""
            SELECT TOP (?)
                PatientAdmissionID as patient_id,
                MedicalFileNumber as mrn,
                FullName as full_name,
                FirstName as first_name,
                LastName as last_name,
                CONVERT(VARCHAR(10), BirthDate, 23) as date_of_birth,
                DATEDIFF(YEAR, BirthDate, GETDATE()) as age,
                CASE
                    WHEN SEX = 'M' THEN 'Male'
                    WHEN SEX = 'F' THEN 'Female'
                    ELSE SEX
                END as gender,
                PhoneNumber1 as phone,
                'reserve' as source
            FROM dbo.APP_RESERVE_PATIENT
            WHERE {where_clause}
            ORDER BY full_name ASC
        """

        cursor.execute(query_str, [limit] + params)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    finally:
        conn.close()


# ==================== GET PATIENT PROFILE ====================

def get_reserve_patient_profile(patient_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a RESERVE patient's profile fields (demographics only — no incident
    stats; see get_incident_stats_for_name for that, shared with external
    patients since both are matched against APP_IncidentCase.PatientName the
    same way).

    Args:
        patient_id: The patient's PatientAdmissionID (reserve table)

    Returns:
        Patient profile dict (source='reserve') or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT TOP 1
                PatientAdmissionID as PatientID,
                MedicalFileNumber as MRN,
                FullName as PatientName,
                FirstName as PatientNameEnglish,
                CONVERT(VARCHAR(10), BirthDate, 23) as DateOfBirth,
                DATEDIFF(YEAR, BirthDate, GETDATE()) as Age,
                CASE
                    WHEN SEX = 'M' THEN 'Male'
                    WHEN SEX = 'F' THEN 'Female'
                    WHEN SEX = N'ذكر' THEN 'Male'
                    WHEN SEX = N'أنثى' THEN 'Female'
                    ELSE SEX
                END as Gender,
                '' as Nationality,
                PhoneNumber1 as Phone,
                '' as Email,
                '' as Address,
                '' as EmergencyContact,
                '' as EmergencyPhone,
                CONVERT(VARCHAR(19), SystemTime, 121) as RegistrationDate
            FROM dbo.APP_RESERVE_PATIENT
            WHERE PatientAdmissionID = ?
        """, patient_id)

        row = cursor.fetchone()
        if not row:
            return None

        columns = [col[0] for col in cursor.description]
        profile = dict(zip(columns, row))
        profile['source'] = 'reserve'
        return profile

    finally:
        conn.close()


def get_incident_stats_for_name(patient_name: str) -> Dict[str, Any]:
    """
    Count incidents and find the last feedback date for a patient, matched
    purely by the PatientName text snapshot on APP_IncidentCase — the only
    linkage that has ever existed for incidents (no FK, see Investigation 1
    §5). Used for both reserve and external patient profiles alike, since
    neither is ever joined to APP_IncidentCase by ID.

    Returns:
        {"TotalIncidents": int, "LastVisitDate": str|None}
    """
    if not patient_name:
        return {"TotalIncidents": 0, "LastVisitDate": None}

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT
                COUNT(*) as TotalIncidents,
                MAX(CONVERT(VARCHAR(10), FeedbackRecievedDate, 23)) as LastVisitDate
            FROM dbo.APP_IncidentCase ic
            WHERE ic.PatientName = ?
        """, patient_name)

        row = cursor.fetchone()
        if row:
            return {"TotalIncidents": row[0] or 0, "LastVisitDate": row[1]}
        return {"TotalIncidents": 0, "LastVisitDate": None}

    finally:
        conn.close()


# ==================== GET PATIENT METRICS WITH AGGREGATION ====================

def get_patient_metrics(
    patient_id: int,
    patient_name: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get aggregated metrics for a patient including severity and category breakdowns.
    
    Args:
        patient_id: Patient unique identifier
        patient_name: Patient name for filtering
        from_date: Optional start date filter
        to_date: Optional end date filter
    
    Returns:
        Dict with total incidents and breakdowns by severity and category
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Matched purely on the PatientName text snapshot (see
        # get_incident_stats_for_name docstring) — patient_id is no longer
        # used to re-derive a name here; the caller (patients_service)
        # already resolved patient_name via the profile lookup.
        if patient_name:
            conditions = ["ic.PatientName = ?"]
            params = [patient_name]
        else:
            conditions = ["1=0"]  # No patient name — nothing to match
            params = []

        if from_date:
            conditions.append("CONVERT(DATE, ic.FeedbackRecievedDate) >= ?")
            params.append(from_date)
        
        if to_date:
            conditions.append("CONVERT(DATE, ic.FeedbackRecievedDate) <= ?")
            params.append(to_date)
        
        where_clause = " AND ".join(conditions)
        
        # Get total incidents and last visit date
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total_incidents,
                MAX(CONVERT(VARCHAR(10), ic.FeedbackRecievedDate, 23)) as last_visit_date
            FROM dbo.APP_IncidentCase ic
            WHERE {where_clause}
        """, params)
        
        summary_row = cursor.fetchone()
        total_incidents = summary_row[0] if summary_row else 0
        last_visit_date = summary_row[1] if summary_row else None
        
        # Get severity breakdown
        cursor.execute(f"""
            SELECT 
                COALESCE(sev.SeverityName, 'Unknown') as severity,
                COUNT(*) as count
            FROM dbo.APP_IncidentCase ic
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
                COUNT(*) as count
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat ON ic.CategoryID = cat.CategoryID
            WHERE {where_clause}
            GROUP BY cat.CategoryName
        """, params)
        
        category_breakdown = {}
        for row in cursor.fetchall():
            category_breakdown[row[0]] = row[1]
        
        return {
            "total_incidents": total_incidents,
            "last_visit_date": last_visit_date,
            "severity_breakdown": severity_breakdown,
            "category_breakdown": category_breakdown
        }
    
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
        conditions = ["ic.PatientName = ?"]
        params = [patient_name or ""]
        
        if from_date:
            conditions.append("CONVERT(DATE, FeedbackRecievedDate) >= ?")
            params.append(from_date)
        
        if to_date:
            conditions.append("CONVERT(DATE, FeedbackRecievedDate) <= ?")
            params.append(to_date)
        
        if department:
            conditions.append("ou.Name LIKE ?")
            params.append(f"%{department}%")
        
        if severity:
            conditions.append("SeverityName = ?")
            params.append(severity)
        
        if status:
            conditions.append("cs.Name = ?")
            params.append(status)
        
        where_clause = " AND ".join(conditions)
        
        # Get total count
        count_query = f"""
            SELECT COUNT(*)
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.AdminsrationUnit ou WITH (NOLOCK) ON ic.IssuingOrgUnitID = ou.UniqueID
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs ON ic.CaseStatusID = cs.CaseStatusID
            WHERE {where_clause}
        """
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]
        
        # Get incidents with pagination
        incidents_query = f"""
            SELECT
                ic.IncidentRequestCaseID as incident_id,
                ic.IncidentRequestCaseID as record_id,
                CONVERT(VARCHAR(10), ic.CreatedAt, 23) as date,
                CONVERT(VARCHAR(10), ic.FeedbackRecievedDate, 23) as feedback_received_date,
                COALESCE(ou.Name, 'Unknown') as department,
                COALESCE(ou.Name, 'غير محدد') as department_ar,
                COALESCE(cat.CategoryName, 'Unknown') as category,
                COALESCE(cat.CategoryName, 'غير محدد') as category_ar,
                COALESCE(sev.SeverityName, 'Unknown') as severity,
                COALESCE(icd.DoctorName, 'غير محدد') as doctor_name,
                COALESCE(cs.Name, 'Open') as status,
                LEFT(ic.ComplaintText, 200) as description,
                CASE WHEN ic.ClinicalRiskTypeID = 2 THEN CAST(1 AS BIT) ELSE CAST(0 AS BIT) END as is_red_flag,
                CASE WHEN ic.ClinicalRiskTypeID = 3 THEN CAST(1 AS BIT) ELSE CAST(0 AS BIT) END as is_never_event
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.AdminsrationUnit ou WITH (NOLOCK) ON ic.IssuingOrgUnitID = ou.UniqueID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs ON ic.CaseStatusID = cs.CaseStatusID
            LEFT JOIN dbo.APP_IncidentCaseDoctor icd ON ic.IncidentRequestCaseID = icd.IncidentRequestCaseID AND icd.IsPrimary = 1
            WHERE {where_clause}
            ORDER BY ic.FeedbackRecievedDate DESC
            OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY
        """
        cursor.execute(incidents_query, params)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        
        # Convert boolean integers to proper booleans
        incidents = []
        for row in rows:
            incident_dict = dict(zip(columns, row))
            # Convert bit fields to boolean
            if 'is_red_flag' in incident_dict:
                incident_dict['is_red_flag'] = bool(incident_dict['is_red_flag'])
            if 'is_never_event' in incident_dict:
                incident_dict['is_never_event'] = bool(incident_dict['is_never_event'])
            incidents.append(incident_dict)
        
        return {
            "patient_id": patient_id,
            "full_name": patient_name or "Unknown",
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
                ic.PatientName,
                COALESCE(ou.Name, 'Unknown') as Department,
                COALESCE(ou_target.Name, 'Unknown') as TargetDepartment,
                COALESCE(cat.CategoryName, 'Unknown') as Category,
                COALESCE(cat.CategoryName, 'غير محدد') as CategoryAr,
                CONCAT(
                    COALESCE(dom.DomainName, ''),
                    ' > ',
                    COALESCE(cat.CategoryName, ''),
                    ' > ',
                    COALESCE(subcat.SubCategoryName, '')
                ) as Classification,
                COALESCE(sev.SeverityName, 'Unknown') as Severity,
                COALESCE(hl.HarmLevel, 'Unknown') as HarmLevel,
                COALESCE(st.StageName, 'Unknown') as Stage,
                COALESCE(icd.DoctorName, 'غير محدد') as DoctorName,
                COALESCE(cs.Name, 'Open') as Status,
                ic.ComplaintText,
                ic.ImmediateAction,
                ic.TakenAction,
                CASE WHEN ic.ClinicalRiskTypeID = 2 THEN CAST(1 AS BIT) ELSE CAST(0 AS BIT) END as IsRedFlag,
                CASE WHEN ic.ClinicalRiskTypeID = 3 THEN CAST(1 AS BIT) ELSE CAST(0 AS BIT) END as IsNeverEvent,
                CONVERT(VARCHAR(19), ic.CreatedAt, 121) as CreatedAt
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.AdminsrationUnit ou WITH (NOLOCK) ON ic.IssuingOrgUnitID = ou.UniqueID
            LEFT JOIN dbo.AdminsrationUnit ou_target WITH (NOLOCK) ON ic.TargetOrgUnitID = ou_target.UniqueID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_LOOKUP_DOMAIN dom ON ic.DomainID = dom.DomainID
            LEFT JOIN dbo.APP_LOOKUP_SUBCATEGORY subcat ON ic.SubCategoryID = subcat.SubCategoryID
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_HARM_LEVEL hl ON ic.HarmLevelID = hl.HarmID
            LEFT JOIN dbo.APP_LOOKUP_CASE_STAGE st ON ic.StageID = st.StageID
            LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs ON ic.CaseStatusID = cs.CaseStatusID
            LEFT JOIN dbo.APP_IncidentCaseDoctor icd ON ic.IncidentRequestCaseID = icd.IncidentRequestCaseID AND icd.IsPrimary = 1
            WHERE ic.IncidentRequestCaseID = ?
        """, incident_id)
        
        row = cursor.fetchone()
        if not row:
            return None
        
        columns = [col[0] for col in cursor.description]
        incident_dict = dict(zip(columns, row))
        
        # Convert bit fields to boolean
        if 'IsRedFlag' in incident_dict:
            incident_dict['IsRedFlag'] = bool(incident_dict['IsRedFlag'])
        if 'IsNeverEvent' in incident_dict:
            incident_dict['IsNeverEvent'] = bool(incident_dict['IsNeverEvent'])
        
        return incident_dict
    
    finally:
        conn.close()


# ==================== EXPORT PATIENT HISTORY ====================

def get_patient_incidents_for_export(
    patient_id: Any,
    profile: Optional[Dict[str, Any]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    include_profile: bool = True
) -> Dict[str, Any]:
    """
    Get patient data for export in CSV or JSON format.

    Args:
        patient_id: Patient identifier (reserve int or external composite id)
        profile: The patient's already-resolved profile dict (from
            api.services.patient_directory_service.resolve_patient_profile),
            or None if the patient could not be resolved. Resolving here
            used to mean a fallback UNION query against the hospital view;
            since the caller (patients_service) already resolves the
            profile through the reserve/external adapter before calling
            this, that fallback is no longer needed.
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

        patient_name = profile.get('PatientName') if profile else None

        if include_profile and profile:
            export_data["patient"] = {
                "patient_id": profile.get('PatientID', patient_id),
                "mrn": profile.get('MRN'),
                "full_name": patient_name,
                "total_incidents": profile.get('TotalIncidents', 0)
            }

        # No resolved name — nothing in APP_IncidentCase can match (see
        # get_incident_stats_for_name docstring for why name is the only key).
        if not patient_name:
            return export_data

        conditions = ["ic.PatientName = ?"]
        params = [patient_name]

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
                CONVERT(VARCHAR(10), COALESCE(ic.CreatedAt, ic.FeedbackRecievedDate), 23) as Date,
                COALESCE(org.Name, 'غير محدد') as Department,
                COALESCE(cat.CategoryName, 'غير محدد') as Category,
                COALESCE(sev.SeverityName, 'غير محدد') as Severity,
                COALESCE(cs.Name, 'غير محدد') as Status,
                ic.ComplaintText
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.AdminsrationUnit org ON ic.IssuingOrgUnitID = org.UniqueID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs ON ic.CaseStatusID = cs.CaseStatusID
            WHERE {where_clause}
            ORDER BY COALESCE(ic.FeedbackRecievedDate, ic.CreatedAt) DESC
        """
        
        cursor.execute(query, params)
        columns = [col[0] for col in cursor.description]
        export_data["incidents"] = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return export_data
    
    finally:
        conn.close()


# ==================== SOFT DELETE / DEACTIVATE PATIENT ====================

def get_reserve_patient_by_id(patient_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a specific patient from the reserve table by ID.
    
    Args:
        patient_id: The patient's PatientAdmissionID
        
    Returns:
        Dict with patient data or None if not found
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
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
                CONVERT(VARCHAR(19), SystemTime, 121) as CreatedAt,
                IsActive
            FROM APP_RESERVE_PATIENT
            WHERE PatientAdmissionID = ?
        """, (patient_id,))
        
        row = cursor.fetchone()
        if row:
            return {
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
                "is_active": bool(row.IsActive),
                "source": "reserve"
            }
        return None
        
    except Exception as e:
        raise Exception(f"Failed to get reserve patient: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def count_incidents_by_patient(patient_id: int) -> int:
    """
    Count the number of incidents associated with a patient.
    
    NOTE: APP_IncidentCase does not have a direct PatientID/PatientAdmissionID 
    foreign key - patients are stored by name text only. Reserve patients 
    (APP_RESERVE_PATIENT) are user-created records that are not directly 
    linked to incidents by ID.
    
    Therefore, we return 0 to allow soft-delete for any reserve patient.
    Hard-delete should be used with caution as it cannot check incident linkage.
    
    Args:
        patient_id: The patient's PatientAdmissionID
        
    Returns:
        Always returns 0 since reserve patients are not linked by ID
    """
    # APP_IncidentCase uses PatientName (text) not PatientAdmissionID (FK)
    # Reserve patients are not directly linked to incidents by ID
    # Return 0 to allow soft-delete (deactivation) of reserve patients
    return 0


def deactivate_reserve_patient(patient_id: int) -> bool:
    """
    Soft-delete a reserve patient by setting IsActive = 0.
    
    Args:
        patient_id: The patient's PatientAdmissionID to deactivate
        
    Returns:
        True if successful, False if patient not found
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE dbo.APP_RESERVE_PATIENT
            SET IsActive = 0
            WHERE PatientAdmissionID = ?
        """, (patient_id,))
        
        rows_affected = cursor.rowcount
        conn.commit()
        
        return rows_affected > 0
        
    except Exception as e:
        conn.rollback()
        raise Exception(f"Failed to deactivate patient: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def hard_delete_reserve_patient(patient_id: int) -> bool:
    """
    Permanently delete a reserve patient from the table.
    
    Only call this if the patient has no associated incidents.
    
    Args:
        patient_id: The patient's PatientAdmissionID to delete
        
    Returns:
        True if successful, False if patient not found
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM dbo.APP_RESERVE_PATIENT
            WHERE PatientAdmissionID = ?
        """, (patient_id,))
        
        rows_affected = cursor.rowcount
        conn.commit()
        
        return rows_affected > 0
        
    except Exception as e:
        conn.rollback()
        raise Exception(f"Failed to delete patient: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def reactivate_reserve_patient(patient_id: int) -> bool:
    """
    Re-activate a soft-deleted patient by setting IsActive = 1.
    
    Args:
        patient_id: The patient's PatientAdmissionID to reactivate
        
    Returns:
        True if successful, False if patient not found
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE dbo.APP_RESERVE_PATIENT
            SET IsActive = 1
            WHERE PatientAdmissionID = ?
        """, (patient_id,))
        
        rows_affected = cursor.rowcount
        conn.commit()
        
        return rows_affected > 0
        
    except Exception as e:
        conn.rollback()
        raise Exception(f"Failed to reactivate patient: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

