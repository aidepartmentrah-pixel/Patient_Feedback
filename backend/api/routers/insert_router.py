"""
Insert Router
API endpoints for creating new incident/feedback records.
"""

from fastapi import APIRouter, HTTPException, Body, Query
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date

from ..services.insert_service import create_record
from ..services.search_service import (
    search_patients,
    search_doctors,
    search_employees,
    get_patient_by_id,
    get_doctor_by_id,
    get_employee_by_id
)


router = APIRouter(prefix="/api/records", tags=["Records"])


# ==================== REQUEST/RESPONSE MODELS ====================

class CreateRecordRequest(BaseModel):
    """Request model for creating a new record."""
    
    # Required fields
    complaint_text: str = Field(..., min_length=1, description="Full complaint/incident description")
    feedback_received_date: date = Field(..., description="Date the feedback was received")
    domain_id: int = Field(..., gt=0, description="Domain ID")
    category_id: int = Field(..., gt=0, description="Category ID")
    severity_id: int = Field(..., gt=0, description="Severity level ID")
    
    # Optional text fields
    immediate_action: Optional[str] = Field(None, description="Immediate actions taken")
    taken_action: Optional[str] = Field(None, description="Follow-up actions taken")
    
    # Optional metadata
    issuing_department_id: Optional[int] = Field(None, gt=0, description="Issuing department ID")
    target_department_id: Optional[int] = Field(None, gt=0, description="Target department ID")
    source_id: Optional[int] = Field(None, gt=0, description="Feedback source ID")
    is_inpatient: Optional[bool] = Field(True, description="Is inpatient (True) or outpatient (False). Default: True")
    worker_type: Optional[str] = Field(None, description="Worker type involved")
    
    # Optional entity data
    patient_name: Optional[str] = Field(None, description="Patient name")
    
    # Optional classification hierarchy
    subcategory_id: Optional[int] = Field(None, gt=0, description="Subcategory ID")
    classification_id: Optional[int] = Field(None, gt=0, description="Classification ID")
    
    # Optional severity/impact
    stage_id: Optional[int] = Field(None, gt=0, description="Care stage ID")
    harm_id: Optional[int] = Field(None, gt=0, description="Harm level ID")
    improvement_type: Optional[int] = Field(0, ge=0, le=1, description="Improvement opportunity (0=No, 1=Yes)")


# ==================== ENDPOINTS ====================

@router.post("/add")
async def add_record(request: CreateRecordRequest = Body(...)):
    """
    Create a new incident/feedback record.
    
    **Required Fields:**
    - complaint_text
    - feedback_received_date
    - domain_id
    - category_id
    - severity_id
    
    **Validates:**
    - All required fields are present
    - Foreign key references exist in database
    - Hierarchical relationships (category belongs to domain, etc.)
    
    **Returns:**
    - success: true/false
    - record_id: Generated record ID (e.g., REC-2024-0156)
    - id: Database ID
    - status_id: 3 (In Progress)
    - created_at: Timestamp
    
    **Example Request:**
    ```json
    {
      "complaint_text": "تأخر كبير في تشخيص الحالة الطارئة",
      "feedback_received_date": "2024-12-15",
      "domain_id": 1,
      "category_id": 12,
      "severity_id": 2,
      "immediate_action": "تم توفير الرعاية الفورية",
      "patient_name": "أحمد محمد"
    }
    ```
    
    **Example Response:**
    ```json
    {
      "success": true,
      "message": "Record created successfully",
      "record_id": "REC-2024-0156",
      "id": 156,
      "status_id": 3,
      "created_at": "2024-12-17T15:30:00"
    }
    ```
    """
    
    try:
        # Convert request to dictionary
        data = request.model_dump()
        
        # Call service to create record
        result = create_record(data)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": result.get("error", "CREATE_FAILED"),
                    "message": result.get("message", "Failed to create record"),
                    "message_ar": result.get("message_ar", "فشل في إنشاء السجل"),
                    "field": result.get("field")
                }
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"An error occurred: {str(e)}",
                "message_ar": f"حدث خطأ: {str(e)}"
            }
        )


@router.get("/test")
async def test_records_endpoint():
    """
    Test endpoint to verify records service is operational.
    """
    return {
        "status": "operational",
        "service": "records",
        "message": "Records service is running. Use POST /api/records/add to create new records."
    }


# ==================== SEARCH ENDPOINTS ====================

@router.get("/search/patients")
async def search_patients_endpoint(
    q: str = Query(..., min_length=1, description="Search query (patient name, document number, or medical file number)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results (1-100)")
):
    """
    Search for patients by name, document number, or medical file number.
    Only patients found in the database can be selected for incident records.
    
    **Query Parameters:**
    - `q`: Search text (required, min 1 character)
    - `limit`: Maximum results to return (default: 20, max: 100)
    
    **Examples:**
    - `/api/records/search/patients?q=أحمد` - Search for patients named أحمد
    - `/api/records/search/patients?q=123456&limit=10` - Search by document number
    
    **Returns:**
    ```json
    {
      "success": true,
      "patients": [
        {
          "patient_admission_id": 12345,
          "full_name": "أحمد محمد علي",
          "first_name": "أحمد",
          "last_name": "علي",
          "document_number": "123456789",
          "phone_number": "0501234567",
          "birth_date": "1990-05-15",
          "sex": "M",
          "medical_file_number": "MF-2024-001",
          "admission_date": "2024-12-15T10:30:00"
        }
      ],
      "count": 1
    }
    ```
    """
    result = search_patients(q, limit)
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SEARCH_FAILED",
                "message": result.get("error", "Failed to search patients")
            }
        )
    
    return result


@router.get("/search/doctors")
async def search_doctors_endpoint(
    q: str = Query(..., min_length=1, description="Search query (doctor name)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results (1-100)")
):
    """
    Search for active doctors by name.
    Returns doctors with their speciality information.
    Multiple doctors can be selected for an incident record.
    
    **Query Parameters:**
    - `q`: Search text (required, min 1 character)
    - `limit`: Maximum results to return (default: 20, max: 100)
    
    **Examples:**
    - `/api/records/search/doctors?q=خالد` - Search for doctors named خالد
    - `/api/records/search/doctors?q=أحمد&limit=10` - Limit results to 10
    
    **Returns:**
    ```json
    {
      "success": true,
      "doctors": [
        {
          "doctor_id": 45,
          "name": "د. خالد حسن",
          "speciality_id": 3,
          "speciality_name": "طب الطوارئ",
          "is_active": true,
          "is_admitted": true,
          "is_clinic": false
        }
      ],
      "count": 1
    }
    ```
    """
    result = search_doctors(q, limit)
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SEARCH_FAILED",
                "message": result.get("error", "Failed to search doctors")
            }
        )
    
    return result


@router.get("/search/employees")
async def search_employees_endpoint(
    q: str = Query(..., min_length=1, description="Search query (employee name)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results (1-100)")
):
    """
    Search for active employees by name.
    Returns employees with their job title information.
    Multiple employees can be selected for an incident record.
    
    **Query Parameters:**
    - `q`: Search text (required, min 1 character)
    - `limit`: Maximum results to return (default: 20, max: 100)
    
    **Examples:**
    - `/api/records/search/employees?q=محمد` - Search for employees named محمد
    - `/api/records/search/employees?q=أحمد&limit=10` - Limit results to 10
    
    **Returns:**
    ```json
    {
      "success": true,
      "employees": [
        {
          "employee_id": 789,
          "full_name": "محمد أحمد السعيد",
          "job_title": "ممرض",
          "job_id": 12,
          "department_id": 5,
          "section_id": 8,
          "administration_id": 2,
          "is_manager": false,
          "is_active": true
        }
      ],
      "count": 1
    }
    ```
    """
    result = search_employees(q, limit)
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SEARCH_FAILED",
                "message": result.get("error", "Failed to search employees")
            }
        )
    
    return result


@router.get("/patient/{patient_admission_id}")
async def get_patient_endpoint(patient_admission_id: int):
    """
    Get a specific patient by PatientAdmissionID.
    Used to verify patient selection.
    
    **Path Parameters:**
    - `patient_admission_id`: The patient admission ID
    
    **Returns:**
    ```json
    {
      "success": true,
      "patient": {
        "patient_admission_id": 12345,
        "full_name": "أحمد محمد علي",
        "document_number": "123456789",
        ...
      }
    }
    ```
    """
    result = get_patient_by_id(patient_admission_id)
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "NOT_FOUND",
                "message": result.get("error", "Patient not found")
            }
        )
    
    return result


@router.get("/doctor/{doctor_id}")
async def get_doctor_endpoint(doctor_id: int):
    """
    Get a specific doctor by DoctorID.
    Used to verify doctor selection.
    
    **Path Parameters:**
    - `doctor_id`: The doctor ID
    
    **Returns:**
    ```json
    {
      "success": true,
      "doctor": {
        "doctor_id": 45,
        "name": "د. خالد حسن",
        "speciality_name": "طب الطوارئ",
        ...
      }
    }
    ```
    """
    result = get_doctor_by_id(doctor_id)
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "NOT_FOUND",
                "message": result.get("error", "Doctor not found")
            }
        )
    
    return result


@router.get("/employee/{employee_id}")
async def get_employee_endpoint(employee_id: int):
    """
    Get a specific employee by EmployeeID.
    Used to verify employee selection.
    
    **Path Parameters:**
    - `employee_id`: The employee ID
    
    **Returns:**
    ```json
    {
      "success": true,
      "employee": {
        "employee_id": 789,
        "full_name": "محمد أحمد السعيد",
        "job_title": "ممرض",
        ...
      }
    }
    ```
    """
    result = get_employee_by_id(employee_id)
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "NOT_FOUND",
                "message": result.get("error", "Employee not found")
            }
        )
    
    return result
