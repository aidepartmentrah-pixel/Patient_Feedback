"""
Search Service
Provides search functionality for patients, doctors, and employees.
Used by the insert page to search and select entities from the database.
"""

from typing import List, Dict, Any
from core.database import get_connection
from core.table_config import DOCTORS_TABLE, HR_EMPLOYEES_TABLE
from api.services import patient_directory_service


def search_patients(search_text: str, limit: int = 20) -> Dict[str, Any]:
    """
    Search for patients by name (free-text only — this is the endpoint
    behind the incident-creation autocomplete, GET /api/records/search/patients).

    SESSION C1: merges HCAT's reserve table (APP_RESERVE_PATIENT) with the
    Hospital Directory API instead of the old hospital view — see
    api.services.patient_directory_service for the merge/normalization logic.
    """
    return patient_directory_service.search_patients_insert_flow(search_text, limit)


def search_doctors(search_text: str, limit: int = 20) -> Dict[str, Any]:
    """
    Search for doctors by name.
    
    DUAL-SOURCE PATTERN: Merges results from both hospital view
    (VW_Doctors) and reserve table (APP_RESERVE_DOCTOR).
    
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
        cursor.execute(f"""
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
                FROM {DOCTORS_TABLE}
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
        
        cursor.execute(f"""
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
            FROM {HR_EMPLOYEES_TABLE}
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


def get_patient_by_id(patient_admission_id) -> Dict[str, Any]:
    """
    Get a specific patient by id — verifies patient selection. Confirmed
    unreachable from the current frontend (no caller found in
    Front_End_Feedback_Analysis), kept correct for API completeness.

    SESSION C1: patient_admission_id is now either a reserve
    PatientAdmissionID or an opaque external id (see
    hospital_directory_client.encode_external_patient_id) — routes to
    reserve SQL or the Hospital Directory API accordingly.
    """
    return patient_directory_service.get_patient_by_id_insert_shape(patient_admission_id)


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
            cursor.execute(f"""
                SELECT 
                    DoctorID,
                    Name,
                    SpecialityID,
                    SpecialityName,
                    IsActive,
                    IsAdmitted,
                    IsClinic,
                    'hospital' as Source
                FROM {DOCTORS_TABLE}
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
        
        cursor.execute(f"""
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
            FROM {HR_EMPLOYEES_TABLE}
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
