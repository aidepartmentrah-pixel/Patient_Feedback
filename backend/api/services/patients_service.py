"""
Patients Service Layer
Business logic for patient history operations.
"""

from typing import List, Dict, Any, Optional
import csv
import io
import re
from datetime import datetime, date

from ..db_layer.patients_db import (
    search_patients,
    get_patient_profile,
    get_patient_incidents,
    get_incident_details,
    get_patient_incidents_for_export,
    create_patient
)


# ==================== CREATE PATIENT ====================

def create_patient_service(
    first_name: str,
    middle_name: Optional[str] = None,
    last_name: Optional[str] = None,
    mother_name: Optional[str] = None,
    phone_number: Optional[str] = None,
    phone_number2: Optional[str] = None,
    birth_date: Optional[str] = None,
    sex: Optional[str] = None,
    document_number: Optional[str] = None,
    medical_file_number: Optional[str] = None,
    spouse: Optional[str] = None,
    address_line1: Optional[str] = None,
    address_line2: Optional[str] = None,
    created_by_user_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a new patient with validation and sanitization.
    
    This service layer adds business logic validation on top of DB layer.
    All input is sanitized and validated before passing to DB layer.
    
    Validation Rules:
    - FirstName: Required, 2-150 chars, alphanumeric + spaces + Arabic chars
    - MiddleName/LastName/MotherName: Optional, max 150 chars
    - PhoneNumber: Optional, max 50 chars, basic format check
    - BirthDate: Optional, valid date format (YYYY-MM-DD), not in future, age < 150
    - SEX: Optional, must be M/F/Male/Female (case-insensitive)
    - DocumentNumber: Optional, max 100 chars, alphanumeric
    - MedicalFileNumber: Optional, max 100 chars, alphanumeric
    - Spouse: Optional, max 150 chars
    - AddressLine1/2: Optional, max 300 chars
    
    Args:
        first_name: Patient's first name (REQUIRED)
        ... (other optional fields)
        created_by_user_id: User ID who is creating this patient
    
    Returns:
        Dict with created patient data
    
    Raises:
        ValueError: If validation fails with descriptive message
        Exception: If database operation fails
    """
    
    # ============== VALIDATION ==============
    
    # 1. FirstName validation (REQUIRED)
    if not first_name or not first_name.strip():
        raise ValueError("FirstName is required and cannot be empty")
    
    first_name = first_name.strip()
    
    if len(first_name) < 2:
        raise ValueError("FirstName must be at least 2 characters long")
    
    if len(first_name) > 150:
        raise ValueError("FirstName must not exceed 150 characters")
    
    # Check for valid characters (letters, spaces, Arabic, hyphens, apostrophes)
    if not re.match(r'^[\w\s\u0600-\u06FF\'-]+$', first_name, re.UNICODE):
        raise ValueError("FirstName contains invalid characters. Only letters, spaces, hyphens, and apostrophes allowed")
    
    # 2. MiddleName validation (OPTIONAL)
    if middle_name:
        middle_name = middle_name.strip()
        if len(middle_name) > 150:
            raise ValueError("MiddleName must not exceed 150 characters")
        if not re.match(r'^[\w\s\u0600-\u06FF\'-]+$', middle_name, re.UNICODE):
            raise ValueError("MiddleName contains invalid characters")
    else:
        middle_name = None
    
    # 3. LastName validation (OPTIONAL)
    if last_name:
        last_name = last_name.strip()
        if len(last_name) > 150:
            raise ValueError("LastName must not exceed 150 characters")
        if not re.match(r'^[\w\s\u0600-\u06FF\'-]+$', last_name, re.UNICODE):
            raise ValueError("LastName contains invalid characters")
    else:
        last_name = None
    
    # 4. MotherName validation (OPTIONAL)
    if mother_name:
        mother_name = mother_name.strip()
        if len(mother_name) > 150:
            raise ValueError("MotherName must not exceed 150 characters")
        if not re.match(r'^[\w\s\u0600-\u06FF\'-]+$', mother_name, re.UNICODE):
            raise ValueError("MotherName contains invalid characters")
    else:
        mother_name = None
    
    # 5. PhoneNumber validation (OPTIONAL)
    if phone_number:
        phone_number = phone_number.strip()
        if len(phone_number) > 50:
            raise ValueError("PhoneNumber must not exceed 50 characters")
        # Basic phone format check: digits, spaces, +, -, (, )
        if not re.match(r'^[\d\s\+\-\(\)]+$', phone_number):
            raise ValueError("PhoneNumber contains invalid characters. Only digits, spaces, +, -, (, ) allowed")
        # Must have at least 7 digits
        digits_only = re.sub(r'\D', '', phone_number)
        if len(digits_only) < 7:
            raise ValueError("PhoneNumber must contain at least 7 digits")
    else:
        phone_number = None
    
    # 6. PhoneNumber2 validation (OPTIONAL)
    if phone_number2:
        phone_number2 = phone_number2.strip()
        if len(phone_number2) > 50:
            raise ValueError("PhoneNumber2 must not exceed 50 characters")
        if not re.match(r'^[\d\s\+\-\(\)]+$', phone_number2):
            raise ValueError("PhoneNumber2 contains invalid characters")
        digits_only = re.sub(r'\D', '', phone_number2)
        if len(digits_only) < 7:
            raise ValueError("PhoneNumber2 must contain at least 7 digits")
    else:
        phone_number2 = None
    
    # 7. BirthDate validation (OPTIONAL)
    if birth_date:
        birth_date = birth_date.strip()
        
        # Validate format YYYY-MM-DD
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', birth_date):
            raise ValueError("BirthDate must be in format YYYY-MM-DD")
        
        # Parse and validate date
        try:
            birth_date_obj = datetime.strptime(birth_date, '%Y-%m-%d').date()
        except ValueError:
            raise ValueError("BirthDate is not a valid date")
        
        # Check not in future
        today = date.today()
        if birth_date_obj > today:
            raise ValueError("BirthDate cannot be in the future")
        
        # Check reasonable age (< 150 years)
        age_in_years = (today - birth_date_obj).days / 365.25
        if age_in_years > 150:
            raise ValueError("BirthDate indicates age over 150 years, please verify")
        
        # Check not too recent (at least born)
        if age_in_years < 0:
            raise ValueError("BirthDate is invalid")
    else:
        birth_date = None
    
    # 8. SEX validation (OPTIONAL)
    if sex:
        sex = sex.strip().upper()
        valid_sex_values = ['M', 'F', 'MALE', 'FEMALE']
        if sex not in valid_sex_values:
            raise ValueError("SEX must be one of: M, F, Male, Female")
        # Normalize to single letter
        if sex == 'MALE':
            sex = 'M'
        elif sex == 'FEMALE':
            sex = 'F'
    else:
        sex = None
    
    # 9. DocumentNumber validation (OPTIONAL)
    if document_number:
        document_number = document_number.strip()
        if len(document_number) > 100:
            raise ValueError("DocumentNumber must not exceed 100 characters")
        # Alphanumeric + hyphens
        if not re.match(r'^[\w\-]+$', document_number):
            raise ValueError("DocumentNumber contains invalid characters. Only alphanumeric and hyphens allowed")
    else:
        document_number = None
    
    # 10. MedicalFileNumber validation (OPTIONAL)
    if medical_file_number:
        medical_file_number = medical_file_number.strip()
        if len(medical_file_number) > 100:
            raise ValueError("MedicalFileNumber must not exceed 100 characters")
        # Alphanumeric + hyphens
        if not re.match(r'^[\w\-]+$', medical_file_number):
            raise ValueError("MedicalFileNumber contains invalid characters. Only alphanumeric and hyphens allowed")
    else:
        medical_file_number = None
    
    # 11. Spouse validation (OPTIONAL)
    if spouse:
        spouse = spouse.strip()
        if len(spouse) > 150:
            raise ValueError("Spouse name must not exceed 150 characters")
        if not re.match(r'^[\w\s\u0600-\u06FF\'-]+$', spouse, re.UNICODE):
            raise ValueError("Spouse name contains invalid characters")
    else:
        spouse = None
    
    # 12. AddressLine1 validation (OPTIONAL)
    if address_line1:
        address_line1 = address_line1.strip()
        if len(address_line1) > 300:
            raise ValueError("AddressLine1 must not exceed 300 characters")
    else:
        address_line1 = None
    
    # 13. AddressLine2 validation (OPTIONAL)
    if address_line2:
        address_line2 = address_line2.strip()
        if len(address_line2) > 300:
            raise ValueError("AddressLine2 must not exceed 300 characters")
    else:
        address_line2 = None
    
    # ============== CALL DB LAYER ==============
    
    try:
        patient = create_patient(
            first_name=first_name,
            middle_name=middle_name,
            last_name=last_name,
            mother_name=mother_name,
            phone_number=phone_number,
            phone_number2=phone_number2,
            birth_date=birth_date,
            sex=sex,
            document_number=document_number,
            medical_file_number=medical_file_number,
            spouse=spouse,
            address_line1=address_line1,
            address_line2=address_line2,
            created_by_user_id=created_by_user_id
        )
        
        return patient
        
    except ValueError as ve:
        # Re-raise validation errors from DB layer (duplicates, etc.)
        raise ve
    except Exception as e:
        # Wrap other exceptions
        raise Exception(f"Failed to create patient: {str(e)}")


# ==================== SEARCH PATIENTS ====================

def search_patients_service(
    query: Optional[str] = None,
    mrn: Optional[str] = None,
    phone: Optional[str] = None,
    date_of_birth: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Search for patients.
    
    Returns:
        Dict with patients list and total count
    """
    # Validation
    if limit > 100:
        limit = 100  # Cap at 100 for privacy
    
    if not any([query, mrn, phone, date_of_birth]):
        return {"patients": [], "total": 0, "message": "At least one search criterion required"}
    
    try:
        patients = search_patients(
            query=query,
            mrn=mrn,
            phone=phone,
            date_of_birth=date_of_birth,
            limit=limit
        )
        
        return {
            "patients": patients,
            "total": len(patients)
        }
    
    except Exception as e:
        raise Exception(f"Patient search failed: {str(e)}")


# ==================== GET PATIENT PROFILE ====================

def get_patient_profile_service(patient_id: int) -> Dict[str, Any]:
    """
    Get patient profile with all details.
    
    Returns:
        Patient profile dict
    """
    try:
        profile = get_patient_profile(patient_id)
        
        if not profile:
            raise Exception(f"Patient {patient_id} not found")
        
        return profile
    
    except Exception as e:
        raise Exception(f"Failed to get patient profile: {str(e)}")


# ==================== GET PATIENT INCIDENTS ====================

def get_patient_incidents_service(
    patient_id: int,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    department: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Get patient incidents with filtering and pagination.
    
    Returns:
        Dict with incidents list and pagination info
    """
    try:
        # Validation
        if limit > 100:
            limit = 100
        if offset < 0:
            offset = 0
        
        # Get patient name for incident query
        profile = get_patient_profile(patient_id)
        patient_name = profile['PatientName'] if profile else None
        
        result = get_patient_incidents(
            patient_id=patient_id,
            patient_name=patient_name,
            from_date=from_date,
            to_date=to_date,
            department=department,
            severity=severity,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return result
    
    except Exception as e:
        raise Exception(f"Failed to get patient incidents: {str(e)}")


# ==================== GET INCIDENT DETAILS ====================

def get_incident_details_service(patient_id: int, incident_id: int) -> Dict[str, Any]:
    """
    Get full incident details.
    
    Returns:
        Full incident details dict
    """
    try:
        incident = get_incident_details(patient_id, incident_id)
        
        if not incident:
            raise Exception(f"Incident {incident_id} not found for patient {patient_id}")
        
        return incident
    
    except Exception as e:
        raise Exception(f"Failed to get incident details: {str(e)}")


# ==================== FULL HISTORY (Combined) ====================

def get_patient_full_history_service(
    patient_id: int,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    department: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Get patient profile and incidents in single response.
    
    Returns:
        Dict with profile and incidents
    """
    try:
        profile = get_patient_profile_service(patient_id)
        
        incidents_data = get_patient_incidents_service(
            patient_id=patient_id,
            from_date=from_date,
            to_date=to_date,
            department=department,
            severity=severity,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return {
            "profile": profile,
            "incidents": incidents_data
        }
    
    except Exception as e:
        raise Exception(f"Failed to get full patient history: {str(e)}")


# ==================== EXPORT ====================

def export_patient_history_service(
    patient_id: int,
    format_type: str = "json",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    include_profile: bool = True
) -> Dict[str, Any]:
    """
    Generate patient history export in CSV or JSON format.
    
    Args:
        patient_id: Patient ID
        format_type: "csv" or "json"
        from_date: Optional start date filter
        to_date: Optional end date filter
        include_profile: Include patient profile
    
    Returns:
        For CSV: Dict with filename and csv_content
        For JSON: JSON export dict
    """
    try:
        # Get export data
        export_data = get_patient_incidents_for_export(
            patient_id=patient_id,
            from_date=from_date,
            to_date=to_date,
            include_profile=include_profile
        )
        
        if format_type == "csv":
            return _generate_csv_export(export_data, patient_id)
        else:
            return export_data
    
    except Exception as e:
        raise Exception(f"Failed to export patient history: {str(e)}")


def _generate_csv_export(export_data: Dict[str, Any], patient_id: int) -> Dict[str, Any]:
    """
    Generate CSV export from export data.
    
    Returns:
        Dict with filename and csv_content
    """
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        if export_data['patient']:
            writer.writerow(['PATIENT PROFILE'])
            writer.writerow(['Patient ID', export_data['patient'].get('patient_id')])
            writer.writerow(['MRN', export_data['patient'].get('mrn')])
            writer.writerow(['Full Name', export_data['patient'].get('full_name')])
            writer.writerow(['Total Incidents', export_data['patient'].get('total_incidents')])
            writer.writerow([])
        
        # Write incidents header
        writer.writerow(['INCIDENT HISTORY'])
        
        if export_data['incidents']:
            headers = export_data['incidents'][0].keys()
            writer.writerow(headers)
            
            # Write incidents
            for incident in export_data['incidents']:
                writer.writerow(incident.values())
        
        csv_content = output.getvalue()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"patient_{patient_id}_history_{timestamp}.csv"
        
        return {
            "filename": filename,
            "content": csv_content,
            "content_type": "text/csv"
        }
    
    except Exception as e:
        raise Exception(f"Failed to generate CSV: {str(e)}")
