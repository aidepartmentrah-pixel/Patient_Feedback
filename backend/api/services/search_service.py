"""
Search Service
Provides search functionality for patients, doctors, and employees.
Used by the insert page to search and select entities from the database.
"""

from typing import List, Dict, Any
from core.database import get_connection


def search_patients(search_text: str, limit: int = 20) -> Dict[str, Any]:
    """
    Search for patients by name or document number.
    
    DUAL-SOURCE PATTERN: Merges results from both hospital table 
    (APP_VIEWTABLE_PATIENT_ADMISSION) and reserve table (APP_RESERVE_PATIENT).
    
    Args:
        search_text: Text to search for in patient names or document numbers
        limit: Maximum number of results to return (default: 20)
    
    Returns:
        Dictionary containing list of matching patients with their details
        Each patient includes a 'source' field ('hospital' or 'reserve')
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Search in FullName, FirstName, LastName, and DocumentNumber
        search_pattern = f"%{search_text}%"
        
        # UNION query: Merge hospital + reserve patients
        cursor.execute("""
            SELECT TOP (?) * FROM (
                -- Hospital patients
                SELECT 
                    PatientAdmissionID,
                    FullName,
                    FirstName,
                    LastName,
                    DocumentNumber,
                    PhoneNumber1,
                    BirthDate,
                    SEX,
                    MedicalFileNumber,
                    AdmissionDate,
                    'hospital' as Source
                FROM APP_VIEWTABLE_PATIENT_ADMISSION
                WHERE 
                    FullName LIKE ? 
                    OR FirstName LIKE ? 
                    OR LastName LIKE ? 
                    OR DocumentNumber LIKE ?
                    OR MedicalFileNumber LIKE ?
                
                UNION ALL
                
                -- Reserve patients
                SELECT 
                    PatientAdmissionID,
                    FullName,
                    FirstName,
                    LastName,
                    DocumentNumber,
                    PhoneNumber1,
                    BirthDate,
                    SEX,
                    MedicalFileNumber,
                    SystemTime as AdmissionDate,
                    'reserve' as Source
                FROM APP_RESERVE_PATIENT
                WHERE 
                    FullName LIKE ? 
                    OR FirstName LIKE ? 
                    OR LastName LIKE ? 
                    OR DocumentNumber LIKE ?
                    OR MedicalFileNumber LIKE ?
            ) AS CombinedPatients
            ORDER BY AdmissionDate DESC
        """, (limit, 
              search_pattern, search_pattern, search_pattern, search_pattern, search_pattern,
              search_pattern, search_pattern, search_pattern, search_pattern, search_pattern))
        
        patients = []
        for row in cursor.fetchall():
            patients.append({
                "patient_admission_id": row.PatientAdmissionID,
                "full_name": row.FullName,
                "first_name": row.FirstName,
                "last_name": row.LastName,
                "document_number": row.DocumentNumber,
                "phone_number": row.PhoneNumber1,
                "birth_date": row.BirthDate.isoformat() if row.BirthDate else None,
                "sex": row.SEX,
                "medical_file_number": row.MedicalFileNumber,
                "admission_date": row.AdmissionDate.isoformat() if row.AdmissionDate else None,
                "source": row.Source  # NEW: Indicates if from hospital or reserve
            })
        
        return {
            "success": True,
            "patients": patients,
            "count": len(patients)
        }
        
    except Exception as e:
        return {
            "success": False,
            "patients": [],
            "count": 0,
            "error": f"Failed to search patients: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def search_doctors(search_text: str, limit: int = 20) -> Dict[str, Any]:
    """
    Search for doctors by name.
    
    DUAL-SOURCE PATTERN: Merges results from both hospital view
    (APP_VIEWTABLE_VW_DOCTORS) and reserve table (APP_RESERVE_DOCTOR).
    
    Args:
        search_text: Text to search for in doctor names
        limit: Maximum number of results to return (default: 20)
    
    Returns:
        Dictionary containing list of matching doctors with speciality
        Each doctor includes a 'source' field ('hospital' or 'reserve')
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        search_pattern = f"%{search_text}%"
        
        # UNION query: Merge hospital + reserve doctors
        cursor.execute("""
            SELECT TOP (?) * FROM (
                -- Hospital doctors
                SELECT 
                    DoctorID,
                    Name,
                    SpecialityID,
                    SpecialityName,
                    IsActive,
                    IsAdmitted,
                    IsClinic,
                    'hospital' as Source
                FROM APP_VIEWTABLE_VW_DOCTORS
                WHERE 
                    Name LIKE ?
                    AND IsActive = 1
                
                UNION ALL
                
                -- Reserve doctors
                SELECT 
                    DoctorID,
                    DoctorName as Name,
                    NULL as SpecialityID,
                    Specialty as SpecialityName,
                    IsActive,
                    0 as IsAdmitted,
                    0 as IsClinic,
                    'reserve' as Source
                FROM APP_RESERVE_DOCTOR
                WHERE 
                    DoctorName LIKE ?
                    AND IsActive = 1
            ) AS CombinedDoctors
            ORDER BY Name
        """, (limit, search_pattern, search_pattern))
        
        doctors = []
        for row in cursor.fetchall():
            doctors.append({
                "doctor_id": row.DoctorID,
                "name": row.Name,
                "speciality_id": row.SpecialityID,
                "speciality_name": row.SpecialityName,
                "is_active": bool(row.IsActive),
                "is_admitted": bool(row.IsAdmitted) if row.IsAdmitted else False,
                "is_clinic": bool(row.IsClinic) if row.IsClinic else False,
                "source": row.Source  # NEW: Indicates if from hospital or reserve
            })
        
        return {
            "success": True,
            "doctors": doctors,
            "count": len(doctors)
        }
        
    except Exception as e:
        return {
            "success": False,
            "doctors": [],
            "count": 0,
            "error": f"Failed to search doctors: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def search_employees(search_text: str, limit: int = 20) -> Dict[str, Any]:
    """
    Search for employees by name.
    Returns employees with their job title (speciality) information.
    
    Args:
        search_text: Text to search for in employee names
        limit: Maximum number of results to return (default: 20)
    
    Returns:
        Dictionary containing list of matching employees with job title
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        search_pattern = f"%{search_text}%"
        
        cursor.execute("""
            SELECT TOP (?)
                EmployeeID,
                FullName,
                JobTitle,
                JobID,
                DepartmentID,
                SectionID,
                AdministrationID,
                IsManager,
                IsActive
            FROM APP_VIEWTABLE_HR_EMPLOYEES
            WHERE 
                FullName LIKE ?
                AND IsActive = 1
            ORDER BY FullName
        """, (limit, search_pattern))
        
        employees = []
        for row in cursor.fetchall():
            employees.append({
                "employee_id": row.EmployeeID,
                "full_name": row.FullName,
                "job_title": row.JobTitle,  # This is the "speciality" for employees
                "job_id": row.JobID,
                "department_id": row.DepartmentID,
                "section_id": row.SectionID,
                "administration_id": row.AdministrationID,
                "is_manager": bool(row.IsManager) if row.IsManager else False,
                "is_active": bool(row.IsActive)
            })
        
        return {
            "success": True,
            "employees": employees,
            "count": len(employees)
        }
        
    except Exception as e:
        return {
            "success": False,
            "employees": [],
            "count": 0,
            "error": f"Failed to search employees: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_patient_by_id(patient_admission_id: int) -> Dict[str, Any]:
    """
    Get a specific patient by PatientAdmissionID.
    Used to verify patient selection.
    
    DUAL-SOURCE PATTERN: Checks reserve table first, then hospital table.
    This prioritizes user-created patients.
    
    Args:
        patient_admission_id: The patient admission ID
    
    Returns:
        Dictionary containing patient details or error
        Includes 'source' field ('hospital' or 'reserve')
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check reserve table first
        cursor.execute("""
            SELECT 
                PatientAdmissionID,
                FullName,
                FirstName,
                LastName,
                DocumentNumber,
                PhoneNumber1,
                BirthDate,
                SEX,
                MedicalFileNumber,
                SystemTime as AdmissionDate,
                'reserve' as Source
            FROM APP_RESERVE_PATIENT
            WHERE PatientAdmissionID = ?
        """, (patient_admission_id,))
        
        row = cursor.fetchone()
        source = 'reserve'
        
        # If not found in reserve, check hospital table
        if not row:
            cursor.execute("""
                SELECT 
                    PatientAdmissionID,
                    FullName,
                    FirstName,
                    LastName,
                    DocumentNumber,
                    PhoneNumber1,
                    BirthDate,
                    SEX,
                    MedicalFileNumber,
                    AdmissionDate,
                    'hospital' as Source
                FROM APP_VIEWTABLE_PATIENT_ADMISSION
                WHERE PatientAdmissionID = ?
            """, (patient_admission_id,))
            
            row = cursor.fetchone()
            source = 'hospital'
        
        if row:
            return {
                "success": True,
                "patient": {
                    "patient_admission_id": row.PatientAdmissionID,
                    "full_name": row.FullName,
                    "first_name": row.FirstName,
                    "last_name": row.LastName,
                    "document_number": row.DocumentNumber,
                    "phone_number": row.PhoneNumber1,
                    "birth_date": row.BirthDate.isoformat() if row.BirthDate else None,
                    "sex": row.SEX,
                    "medical_file_number": row.MedicalFileNumber,
                    "admission_date": row.AdmissionDate.isoformat() if row.AdmissionDate else None,
                    "source": row.Source  # NEW: Indicates if from hospital or reserve
                }
            }
        else:
            return {
                "success": False,
                "patient": None,
                "error": "Patient not found"
            }
        
    except Exception as e:
        return {
            "success": False,
            "patient": None,
            "error": f"Failed to get patient: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_doctor_by_id(doctor_id: int) -> Dict[str, Any]:
    """
    Get a specific doctor by DoctorID.
    Used to verify doctor selection.
    
    DUAL-SOURCE PATTERN: Checks reserve table first, then hospital view.
    This prioritizes user-created doctors.
    
    Args:
        doctor_id: The doctor ID
    
    Returns:
        Dictionary containing doctor details or error
        Includes 'source' field ('hospital' or 'reserve')
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check reserve table first
        cursor.execute("""
            SELECT 
                DoctorID,
                DoctorName as Name,
                NULL as SpecialityID,
                Specialty as SpecialityName,
                IsActive,
                0 as IsAdmitted,
                0 as IsClinic,
                'reserve' as Source
            FROM APP_RESERVE_DOCTOR
            WHERE DoctorID = ?
        """, (doctor_id,))
        
        row = cursor.fetchone()
        source = 'reserve'
        
        # If not found in reserve, check hospital view
        if not row:
            cursor.execute("""
                SELECT 
                    DoctorID,
                    Name,
                    SpecialityID,
                    SpecialityName,
                    IsActive,
                    IsAdmitted,
                    IsClinic,
                    'hospital' as Source
                FROM APP_VIEWTABLE_VW_DOCTORS
                WHERE DoctorID = ?
            """, (doctor_id,))
            
            row = cursor.fetchone()
            source = 'hospital'
        
        if row:
            return {
                "success": True,
                "doctor": {
                    "doctor_id": row.DoctorID,
                    "name": row.Name,
                    "speciality_id": row.SpecialityID,
                    "speciality_name": row.SpecialityName,
                    "is_active": bool(row.IsActive),
                    "is_admitted": bool(row.IsAdmitted) if row.IsAdmitted else False,
                    "is_clinic": bool(row.IsClinic) if row.IsClinic else False,
                    "source": row.Source  # NEW: Indicates if from hospital or reserve
                }
            }
        else:
            return {
                "success": False,
                "doctor": None,
                "error": "Doctor not found"
            }
        
    except Exception as e:
        return {
            "success": False,
            "doctor": None,
            "error": f"Failed to get doctor: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_employee_by_id(employee_id: int) -> Dict[str, Any]:
    """
    Get a specific employee by EmployeeID.
    Used to verify employee selection.
    
    Args:
        employee_id: The employee ID
    
    Returns:
        Dictionary containing employee details or error
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                EmployeeID,
                FullName,
                JobTitle,
                JobID,
                DepartmentID,
                SectionID,
                AdministrationID,
                IsManager,
                IsActive
            FROM APP_VIEWTABLE_HR_EMPLOYEES
            WHERE EmployeeID = ?
        """, (employee_id,))
        
        row = cursor.fetchone()
        
        if row:
            return {
                "success": True,
                "employee": {
                    "employee_id": row.EmployeeID,
                    "full_name": row.FullName,
                    "job_title": row.JobTitle,
                    "job_id": row.JobID,
                    "department_id": row.DepartmentID,
                    "section_id": row.SectionID,
                    "administration_id": row.AdministrationID,
                    "is_manager": bool(row.IsManager) if row.IsManager else False,
                    "is_active": bool(row.IsActive)
                }
            }
        else:
            return {
                "success": False,
                "employee": None,
                "error": "Employee not found"
            }
        
    except Exception as e:
        return {
            "success": False,
            "employee": None,
            "error": f"Failed to get employee: {str(e)}"
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
