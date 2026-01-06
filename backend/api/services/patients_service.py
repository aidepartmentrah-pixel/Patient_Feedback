"""
Patients Service Layer
Business logic for patient history operations.
"""

from typing import List, Dict, Any, Optional
import csv
import io
from datetime import datetime

from ..db_layer.patients_db import (
    search_patients,
    get_patient_profile,
    get_patient_incidents,
    get_incident_details,
    get_patient_incidents_for_export
)


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
