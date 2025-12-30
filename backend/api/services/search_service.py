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
    
    Args:
        search_text: Text to search for in patient names or document numbers
        limit: Maximum number of results to return (default: 20)
    
    Returns:
        Dictionary containing list of matching patients with their details
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Search in FullName, FirstName, LastName, and DocumentNumber
        search_pattern = f"%{search_text}%"
        
        cursor.execute("""
            SELECT TOP (?)
                PatientAdmissionID,
                FullName,
                FirstName,
                LastName,
                DocumentNumber,
                PhoneNumber1,
                BirthDate,
                SEX,
                MedicalFileNumber,
                AdmissionDate
            FROM APP_VIEWTABLE_PATIENT_ADMISSION
            WHERE 
                FullName LIKE ? 
                OR FirstName LIKE ? 
                OR LastName LIKE ? 
                OR DocumentNumber LIKE ?
                OR MedicalFileNumber LIKE ?
            ORDER BY AdmissionDate DESC
        """, (limit, search_pattern, search_pattern, search_pattern, search_pattern, search_pattern))
        
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
                "admission_date": row.AdmissionDate.isoformat() if row.AdmissionDate else None
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
    Returns doctors with their speciality information.
    
    Args:
        search_text: Text to search for in doctor names
        limit: Maximum number of results to return (default: 20)
    
    Returns:
        Dictionary containing list of matching doctors with speciality
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        search_pattern = f"%{search_text}%"
        
        cursor.execute("""
            SELECT TOP (?)
                DoctorID,
                Name,
                SpecialityID,
                SpecialityName,
                IsActive,
                IsAdmitted,
                IsClinic
            FROM APP_VIEWTABLE_VW_DOCTORS
            WHERE 
                Name LIKE ?
                AND IsActive = 1
            ORDER BY Name
        """, (limit, search_pattern))
        
        doctors = []
        for row in cursor.fetchall():
            doctors.append({
                "doctor_id": row.DoctorID,
                "name": row.Name,
                "speciality_id": row.SpecialityID,
                "speciality_name": row.SpecialityName,
                "is_active": bool(row.IsActive),
                "is_admitted": bool(row.IsAdmitted) if row.IsAdmitted else False,
                "is_clinic": bool(row.IsClinic) if row.IsClinic else False
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
    
    Args:
        patient_admission_id: The patient admission ID
    
    Returns:
        Dictionary containing patient details or error
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
                LastName,
                DocumentNumber,
                PhoneNumber1,
                BirthDate,
                SEX,
                MedicalFileNumber,
                AdmissionDate
            FROM APP_VIEWTABLE_PATIENT_ADMISSION
            WHERE PatientAdmissionID = ?
        """, (patient_admission_id,))
        
        row = cursor.fetchone()
        
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
                    "admission_date": row.AdmissionDate.isoformat() if row.AdmissionDate else None
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
    
    Args:
        doctor_id: The doctor ID
    
    Returns:
        Dictionary containing doctor details or error
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                DoctorID,
                Name,
                SpecialityID,
                SpecialityName,
                IsActive,
                IsAdmitted,
                IsClinic
            FROM APP_VIEWTABLE_VW_DOCTORS
            WHERE DoctorID = ?
        """, (doctor_id,))
        
        row = cursor.fetchone()
        
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
                    "is_clinic": bool(row.IsClinic) if row.IsClinic else False
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
