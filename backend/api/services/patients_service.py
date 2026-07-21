"""
Patients Service Layer
Business logic for patient history operations.
"""

from typing import List, Dict, Any, Optional
import csv
import io
import re
from datetime import datetime, date

from .seasonal_word_adapter import SeasonalWordAdapter
from . import patient_directory_service
from .patient_directory_service import ExternalPatientUnavailableError
from ..db_layer.patients_db import (
    get_patient_incidents,
    get_incident_details,
    get_patient_incidents_for_export,
    get_patient_metrics,
    create_patient,
    get_all_reserve_patients,
    get_reserve_patient_by_id,
    count_incidents_by_patient,
    deactivate_reserve_patient,
    hard_delete_reserve_patient,
    reactivate_reserve_patient
)
from ..db_layer.satisfaction_db import get_satisfactions_by_cases


class PatientServiceError(Exception):
    """
    Raised for a distinctly external-API failure so routers can return a
    clear 503-style "external search unavailable" response instead of a
    generic 500 or a misleading 404. Carries the underlying
    hospital_directory_client status string in `.status`.
    """

    def __init__(self, status: str, message: str):
        self.status = status
        super().__init__(message)


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
    Search for patients (Patient History page). Merges HCAT's reserve table
    with the Hospital Directory API — see patient_directory_service.

    Returns:
        Dict with success flag, patients list, count, and external_status/
        external_message describing whether the external half of the search
        succeeded, was skipped (unsupported filter), or failed.
    """
    # Validation
    if limit > 100:
        limit = 100  # Cap at 100 for privacy

    if not any([query, mrn, phone, date_of_birth]):
        return {"success": True, "patients": [], "count": 0, "message": "At least one search criterion required"}

    try:
        result = patient_directory_service.search_patients_history(
            query=query,
            mrn=mrn,
            phone=phone,
            date_of_birth=date_of_birth,
            limit=limit
        )

        return {
            "success": True,
            "patients": result["patients"],
            "count": len(result["patients"]),
            "external_status": result["external_status"],
            "external_message": result["external_message"],
        }

    except Exception as e:
        raise Exception(f"Patient search failed: {str(e)}")


# ==================== GET ALL RESERVE PATIENTS ====================

def get_all_reserve_patients_service(
    limit: int = 100,
    offset: int = 0,
    order_by: str = 'created_at'
) -> Dict[str, Any]:
    """
    Get all reserve patients (user-created patients only).
    
    Args:
        limit: Maximum number of records per page (default: 100, max: 200)
        offset: Number of records to skip (default: 0)
        order_by: Sort field - 'created_at' (newest first) or 'name' (alphabetical)
    
    Returns:
        Dict with patients list, total count, and pagination info
    
    Raises:
        Exception: If retrieval fails
    """
    # Validation
    if limit > 200:
        limit = 200  # Cap at 200
    if limit < 1:
        limit = 100
    if offset < 0:
        offset = 0
    
    # Map order_by parameter
    db_order_by = 'SystemTime' if order_by == 'created_at' else 'FullName'
    
    try:
        result = get_all_reserve_patients(
            limit=limit,
            offset=offset,
            order_by=db_order_by
        )
        
        return result
    
    except Exception as e:
        raise Exception(f"Failed to retrieve reserve patients: {str(e)}")


# ==================== GET PATIENT PROFILE ====================

def get_patient_profile_service(patient_id) -> Dict[str, Any]:
    """
    Get patient profile with all details — routes to the reserve table or
    the Hospital Directory API depending on the id shape (see
    patient_directory_service.resolve_patient_profile).

    Raises:
        PatientServiceError: the id is external but the API is unreachable/
            unauthorized/etc — distinct from "not found" so the router can
            return a clear "external search unavailable" response.
    """
    try:
        profile = patient_directory_service.resolve_patient_profile(patient_id)
    except ExternalPatientUnavailableError as e:
        raise PatientServiceError(e.status, e.message)

    if not profile:
        raise Exception(f"Patient {patient_id} not found")

    return profile


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
        try:
            profile = patient_directory_service.resolve_patient_profile(patient_id)
        except ExternalPatientUnavailableError as e:
            raise PatientServiceError(e.status, e.message)
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

    except PatientServiceError:
        raise
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
        Dict with profile, incidents, metrics, and meta (V2 schema compatible)
    """
    try:
        # Get profile
        profile = get_patient_profile_service(patient_id)
        
        # Get incidents
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
        
        # Get aggregated metrics
        patient_name = profile.get('PatientName') if profile else None
        metrics = get_patient_metrics(
            patient_id=patient_id,
            patient_name=patient_name,
            from_date=from_date,
            to_date=to_date
        )
        
        items = incidents_data.get('incidents', [])
        
        # Enrich items with satisfaction data
        if items:
            case_ids = [item.get('incident_id') or item.get('record_id') for item in items if item.get('incident_id') or item.get('record_id')]
            if case_ids:
                satisfactions_map = get_satisfactions_by_cases(case_ids)
                for item in items:
                    case_id = item.get('incident_id') or item.get('record_id')
                    if case_id and case_id in satisfactions_map:
                        item['satisfaction'] = {
                            'exists': True,
                            **satisfactions_map[case_id]
                        }
                    else:
                        item['satisfaction'] = {'exists': False}
        
        # Return normalized V2 schema format
        return {
            "profile": profile,
            "metrics": metrics,
            "items": items,
            "meta": {
                "entity_type": "patient",
                "entity_id": patient_id,
                "period_from": from_date,
                "period_to": to_date,
                "total": incidents_data.get('total', 0),
                "limit": limit,
                "offset": offset
            }
        }

    except PatientServiceError:
        raise
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
    Generate patient history export in CSV, JSON, or Word format.
    
    Args:
        patient_id: Patient ID
        format_type: "csv", "json", or "word"
        from_date: Optional start date filter
        to_date: Optional end date filter
        include_profile: Include patient profile
    
    Returns:
        For CSV: Dict with filename and csv_content
        For Word: Dict with filename and word_bytes
        For JSON: JSON export dict
    """
    try:
        # Resolve profile once (reserve or external) and hand it to the
        # DB-layer export builder — it no longer re-derives it itself.
        try:
            profile = patient_directory_service.resolve_patient_profile(patient_id)
        except ExternalPatientUnavailableError as e:
            raise PatientServiceError(e.status, e.message)

        export_data = get_patient_incidents_for_export(
            patient_id=patient_id,
            profile=profile,
            from_date=from_date,
            to_date=to_date,
            include_profile=include_profile
        )

        if format_type == "csv":
            return _generate_csv_export(export_data, patient_id)
        elif format_type == "word":
            return _generate_word_export(export_data, patient_id)
        else:
            return export_data

    except PatientServiceError:
        raise
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


def _generate_word_export(export_data: Dict[str, Any], patient_id: int) -> Dict[str, Any]:
    """
    Generate Word document export from export data.
    
    Returns:
        Dict with filename and word_bytes
    """
    try:
        # Generate Word document using SeasonalWordAdapter
        word_bytes = SeasonalWordAdapter.generate_patient_word_report(export_data)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"patient_{patient_id}_report_{timestamp}.docx"
        
        return {
            "filename": filename,
            "content": word_bytes,
            "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        }
    except Exception as e:
        raise Exception(f"Failed to generate Word: {str(e)}")


# ==================== DELETE PATIENT SERVICE ====================

def delete_patient_service(
    patient_id: int,
    force: bool = False
) -> Dict[str, Any]:
    """
    Delete a patient from the reserve table (APP_RESERVE_PATIENT) ONLY.

    This function does NOT touch external (Hospital Directory API) patients.
    External patients cannot be deleted - they are read-only.
    
    Behavior:
    - If patient has associated incidents: soft-delete (deactivate by setting IsActive=0)
    - If patient has no incidents and force=True: hard-delete (permanently remove)
    - If patient has no incidents and force=False: soft-delete (deactivate)
    
    Args:
        patient_id: The patient's PatientAdmissionID to delete (must be from reserve table)
        force: If True and no incidents, permanently delete
    
    Returns:
        Dict with deletion result:
            - success: True if operation succeeded
            - id: Patient ID
            - is_active: Current active state (False if soft-deleted)
            - deleted: True if hard-deleted
            - incident_count: Number of associated incidents
            - action: 'soft_delete' or 'hard_delete'
            - message: Description of what was done
            - message_ar: Arabic description
            
    Raises:
        ValueError: If patient not found in reserve table or is a hospital patient
    """
    try:
        # ============================================
        # STEP 1: Verify patient exists in RESERVE table only
        # ============================================
        reserve_patient = get_reserve_patient_by_id(patient_id)

        if not reserve_patient:
            # Check if it's an external (Hospital Directory API) patient,
            # read-only. In practice this endpoint's path param is typed
            # `int`, so an external id (an "ext__..." string) can never
            # reach here — this check just documents/preserves the intent.
            try:
                external_patient = patient_directory_service.resolve_patient_profile(patient_id)
            except ExternalPatientUnavailableError:
                external_patient = None
            if external_patient and external_patient.get('source') == 'external':
                raise ValueError(
                    f"Patient ID {patient_id} is from the Hospital Directory system and cannot be deleted. "
                    "External patients are read-only."
                )
            raise ValueError(f"Patient with ID {patient_id} not found in reserve table")
        
        patient_name = reserve_patient.get('full_name', f'Patient {patient_id}')
        
        # ============================================
        # STEP 2: Check for associated incidents
        # ============================================
        incident_count = count_incidents_by_patient(patient_id)
        
        if incident_count > 0:
            # Has incidents - must soft-delete (deactivate)
            deactivate_reserve_patient(patient_id)
            
            return {
                'success': True,
                'id': patient_id,
                'is_active': False,
                'deleted': False,
                'incident_count': incident_count,
                'action': 'soft_delete',
                'message': f"Patient '{patient_name}' has {incident_count} associated incident(s). "
                           "Patient has been deactivated (soft-deleted) to preserve data integrity.",
                'message_ar': f"تم إلغاء تفعيل المريض '{patient_name}' لوجود {incident_count} حالة مرتبطة."
            }
        
        # ============================================
        # STEP 3: No incidents - check force flag
        # ============================================
        if force:
            # Hard delete
            hard_delete_reserve_patient(patient_id)
            
            return {
                'success': True,
                'id': patient_id,
                'is_active': None,  # No longer exists
                'deleted': True,
                'incident_count': 0,
                'action': 'hard_delete',
                'message': f"Patient '{patient_name}' has been permanently deleted.",
                'message_ar': f"تم حذف المريض '{patient_name}' بشكل دائم."
            }
        else:
            # Soft delete (deactivate)
            deactivate_reserve_patient(patient_id)
            
            return {
                'success': True,
                'id': patient_id,
                'is_active': False,
                'deleted': False,
                'incident_count': 0,
                'action': 'soft_delete',
                'message': f"Patient '{patient_name}' has been deactivated. "
                           "Use force=true to permanently delete.",
                'message_ar': f"تم إلغاء تفعيل المريض '{patient_name}'."
            }
    
    except ValueError:
        raise  # Re-raise validation errors
    except Exception as e:
        raise Exception(f"Failed to delete patient: {str(e)}")
