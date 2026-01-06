"""
Patients Router
API endpoints for patient history page.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from starlette.responses import StreamingResponse
import io

from ..services.patients_service import (
    search_patients_service,
    get_patient_profile_service,
    get_patient_incidents_service,
    get_incident_details_service,
    get_patient_full_history_service,
    export_patient_history_service
)


router = APIRouter(prefix="/api/patients", tags=["Patients - History"])


# ==================== B.1 SEARCH PATIENTS ====================

@router.get("/search")
async def search_patients_endpoint(
    query: Optional[str] = Query(None, description="Search by patient name"),
    mrn: Optional[str] = Query(None, description="Search by MRN"),
    phone: Optional[str] = Query(None, description="Search by phone"),
    date_of_birth: Optional[str] = Query(None, description="Filter by date of birth (YYYY-MM-DD)"),
    limit: int = Query(50, ge=1, le=100, description="Max results (default: 50, max: 100)")
):
    """
    Search for patients by name, MRN, phone, or date of birth.
    
    **Query Parameters:**
    - `query`: Partial match on patient name (Arabic or English)
    - `mrn`: Exact match on Medical Record Number
    - `phone`: Partial match on phone number
    - `date_of_birth`: Filter by date (YYYY-MM-DD)
    - `limit`: Max results to return (default: 50, max: 100)
    
    **Example Request:**
    ```
    GET /api/patients/search?query=أحمد&limit=20
    GET /api/patients/search?mrn=MRN-123456
    GET /api/patients/search?phone=966
    ```
    
    **Response:**
    ```json
    {
      "patients": [
        {
          "patient_id": "12345",
          "mrn": "MRN-123456",
          "full_name": "أحمد محمد علي",
          "date_of_birth": "1985-05-15",
          "age": 39,
          "gender": "Male",
          "phone": "+966XXXXXXXXX"
        }
      ],
      "total": 1
    }
    ```
    """
    try:
        result = search_patients_service(
            query=query,
            mrn=mrn,
            phone=phone,
            date_of_birth=date_of_birth,
            limit=limit
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ==================== B.2 GET PATIENT PROFILE ====================

@router.get("/{patient_id}/profile")
async def get_patient_profile_endpoint(patient_id: int):
    """
    Get complete patient profile information.
    
    **Path Parameters:**
    - `patient_id`: Patient unique identifier
    
    **Example Request:**
    ```
    GET /api/patients/12345/profile
    ```
    
    **Response:**
    ```json
    {
      "patient_id": "12345",
      "mrn": "MRN-123456",
      "full_name": "أحمد محمد علي",
      "full_name_en": "Ahmed Mohamed Ali",
      "date_of_birth": "1985-05-15",
      "age": 39,
      "gender": "Male",
      "nationality": "Saudi Arabia",
      "phone": "+966XXXXXXXXX",
      "email": "ahmed@example.com",
      "address": "الرياض، السعودية",
      "emergency_contact": "فاطمة علي",
      "emergency_phone": "+966YYYYYYYYY",
      "total_incidents": 5,
      "last_visit_date": "2024-11-15",
      "registration_date": "2020-03-10"
    }
    ```
    
    **Error Responses:**
    - 404: Patient not found
    - 500: Server error
    """
    try:
        profile = get_patient_profile_service(patient_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        return profile
    except Exception as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get profile: {str(e)}")


# ==================== B.3 GET PATIENT INCIDENTS ====================

@router.get("/{patient_id}/incidents")
async def get_patient_incidents_endpoint(
    patient_id: int,
    from_date: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)"),
    department: Optional[str] = Query(None, description="Filter by department"),
    severity: Optional[str] = Query(None, description="Filter by severity (High, Medium, Low)"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=100, description="Max results (default: 100)"),
    offset: int = Query(0, ge=0, description="Pagination offset (default: 0)")
):
    """
    Get all feedback/incident records for a patient.
    
    **Path Parameters:**
    - `patient_id`: Patient unique identifier
    
    **Query Parameters:**
    - `from_date`: Filter incidents from this date (YYYY-MM-DD)
    - `to_date`: Filter incidents to this date (YYYY-MM-DD)
    - `department`: Filter by department name
    - `severity`: Filter by severity (High, Medium, Low)
    - `status`: Filter by case status
    - `limit`: Max results per page (default: 100, max: 100)
    - `offset`: Pagination offset (default: 0)
    
    **Example Request:**
    ```
    GET /api/patients/12345/incidents?severity=High&limit=50&offset=0
    GET /api/patients/12345/incidents?from_date=2024-01-01&to_date=2024-12-31
    ```
    
    **Response:**
    ```json
    {
      "patient_id": "12345",
      "patient_name": "أحمد محمد علي",
      "incidents": [
        {
          "incident_id": 1,
          "record_id": "C-2024-0015",
          "date": "2024-11-15",
          "feedback_received_date": "2024-11-15",
          "department": "Emergency Department",
          "department_ar": "قسم الطوارئ",
          "category": "Delayed Diagnosis",
          "category_ar": "تأخر في التشخيص",
          "severity": "High",
          "doctor_name": "د. خالد حسن",
          "status": "Closed",
          "description": "تأخر كبير في تشخيص الحالة...",
          "is_red_flag": false,
          "is_never_event": false
        }
      ],
      "total": 5,
      "limit": 100,
      "offset": 0
    }
    ```
    """
    try:
        result = get_patient_incidents_service(
            patient_id=patient_id,
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
        raise HTTPException(status_code=500, detail=f"Failed to get incidents: {str(e)}")


# ==================== B.4 GET INCIDENT DETAILS ====================

@router.get("/{patient_id}/incidents/{incident_id}")
async def get_incident_details_endpoint(patient_id: int, incident_id: int):
    """
    Get full details for a specific incident.
    
    **Path Parameters:**
    - `patient_id`: Patient unique identifier
    - `incident_id`: Incident unique identifier
    
    **Example Request:**
    ```
    GET /api/patients/12345/incidents/1
    ```
    
    **Response:**
    ```json
    {
      "incident_id": 1,
      "record_id": "C-2024-0015",
      "date": "2024-11-15",
      "feedback_received_date": "2024-11-15",
      "patient_id": "12345",
      "patient_name": "أحمد محمد علي",
      "department": "Emergency Department",
      "target_department": "Emergency Department",
      "category": "Delayed Diagnosis",
      "category_ar": "تأخر في التشخيص",
      "classification": "Clinical > Delayed Diagnosis > Emergency",
      "severity": "High",
      "harm_level": "Minor",
      "stage": "Admission",
      "doctor_name": "د. خالد حسن",
      "status": "Closed",
      "complaint_text": "تأخر كبير في تشخيص الحالة الطارئة مما أدى إلى تفاقم الحالة",
      "immediate_action": "تم توفير الرعاية الفورية",
      "taken_action": "تم متابعة الحالة",
      "is_red_flag": false,
      "is_never_event": false,
      "created_at": "2024-11-15T10:30:00",
      "last_updated_at": "2024-11-20T14:00:00"
    }
    ```
    
    **Error Responses:**
    - 404: Incident not found
    - 500: Server error
    """
    try:
        incident = get_incident_details_service(patient_id, incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
        return incident
    except Exception as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get incident details: {str(e)}")


# ==================== B.6 FULL HISTORY (Combined) ====================

@router.get("/{patient_id}/full-history")
async def get_patient_full_history_endpoint(
    patient_id: int,
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Get patient profile and incidents in a single request.
    
    **Combines endpoints B.2 and B.3 for efficiency.**
    
    **Example Request:**
    ```
    GET /api/patients/12345/full-history?severity=High
    ```
    
    **Response:**
    ```json
    {
      "profile": {
        "patient_id": "12345",
        "mrn": "MRN-123456",
        "full_name": "أحمد محمد علي",
        ...
      },
      "incidents": {
        "patient_id": "12345",
        "patient_name": "أحمد محمد علي",
        "incidents": [...],
        "total": 5
      }
    }
    ```
    """
    try:
        result = get_patient_full_history_service(
            patient_id=patient_id,
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
        raise HTTPException(status_code=500, detail=f"Failed to get full history: {str(e)}")


# ==================== B.5 EXPORT ====================

@router.get("/{patient_id}/export")
async def export_patient_history_endpoint(
    patient_id: int,
    format: str = Query("json", description="Export format: 'csv' or 'json'"),
    from_date: Optional[str] = Query(None, description="Export from date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="Export to date (YYYY-MM-DD)"),
    include_profile: bool = Query(True, description="Include patient profile (default: true)")
):
    """
    Export patient history in CSV or JSON format.
    
    **Path Parameters:**
    - `patient_id`: Patient unique identifier
    
    **Query Parameters:**
    - `format`: Export format - 'csv' or 'json' (default: json)
    - `from_date`: Export incidents from this date (YYYY-MM-DD)
    - `to_date`: Export incidents to this date (YYYY-MM-DD)
    - `include_profile`: Include patient profile in export (default: true)
    
    **Example Requests:**
    ```
    GET /api/patients/12345/export?format=csv
    GET /api/patients/12345/export?format=json&from_date=2024-01-01&to_date=2024-12-31
    ```
    
    **CSV Response:**
    ```
    Content-Type: text/csv
    Content-Disposition: attachment; filename="patient_12345_history_2024-12-17.csv"
    
    PATIENT PROFILE
    Patient ID,12345
    MRN,MRN-123456
    Full Name,أحمد محمد علي
    Total Incidents,5
    
    INCIDENT HISTORY
    Record ID,Date,Department,Category,Severity,Doctor,Status,Complaint
    C-2024-0015,2024-11-15,Emergency,Delayed Diagnosis,High,د. خالد,Closed,تأخر كبير...
    ```
    
    **JSON Response:**
    ```json
    {
      "export_date": "2024-12-17T15:30:00",
      "format": "json",
      "patient": {
        "patient_id": "12345",
        "mrn": "MRN-123456",
        "full_name": "أحمد محمد علي"
      },
      "incidents": [...]
    }
    ```
    
    **Error Responses:**
    - 400: Invalid format (must be 'csv' or 'json')
    - 404: Patient not found
    - 500: Server error
    """
    try:
        # Validate format
        if format.lower() not in ["csv", "json"]:
            raise HTTPException(status_code=400, detail="Format must be 'csv' or 'json'")
        
        # Get export data
        export_data = export_patient_history_service(
            patient_id=patient_id,
            format_type=format.lower(),
            from_date=from_date,
            to_date=to_date,
            include_profile=include_profile
        )
        
        # Return based on format
        if format.lower() == "csv":
            # Stream CSV file
            csv_content = export_data['content']
            filename = export_data['filename']
            
            return StreamingResponse(
                iter([csv_content]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        else:
            # Return JSON
            return export_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
