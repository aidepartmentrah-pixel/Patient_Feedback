"""
Insert Router
API endpoints for creating new incident/feedback records.
"""

from fastapi import APIRouter, HTTPException, Body, Query, Path, Depends
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime

from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_logged_in
from ..services.insert_service import create_record
from ..services.table_view_service import get_complaint_by_id
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
    issuing_department_id: int = Field(..., gt=0, description="Issuing department ID (required)")
    domain_id: int = Field(..., gt=0, description="Domain ID")
    category_id: int = Field(..., gt=0, description="Category ID")
    subcategory_id: int = Field(..., gt=0, description="Subcategory ID")
    classification_id: int = Field(..., gt=0, description="Classification ID")
    severity_id: int = Field(..., gt=0, description="Severity level ID")
    stage_id: int = Field(..., gt=0, description="Care stage ID")
    harm_id: int = Field(..., gt=0, description="Harm level ID")
    clinical_risk_type_id: int = Field(..., ge=1, le=3, description="Clinical risk type (1=Standard, 2=Red Flag, 3=Never Event)")
    feedback_intent_type_id: int = Field(..., ge=1, description="Feedback intent type ID (required)")
    requires_explanation: bool = Field(..., description="Whether the case requires explanation (required)")
    immediate_action: str = Field(..., description="Immediate actions taken (required)")
    taken_action: str = Field(..., description="Follow-up actions taken (required)")
    patient_name: str = Field(..., description="Patient name (required)")
    is_inpatient: bool = Field(..., description="Is inpatient (True) or outpatient (False) - required")
    source_id: int = Field(..., gt=0, description="Feedback source ID (required)")
    
    # Optional text fields - REMOVED (now required above)
    
    # Optional metadata
    target_department_ids: Optional[list[int]] = Field(None, description="List of target department IDs")
    building_id: Optional[int] = Field(None, gt=0, description="Building ID")
    building: Optional[str] = Field(None, description="Building code (alternative to building_id)")
    explanation_status_id: Optional[int] = Field(None, gt=0, description="Explanation status ID")
    in_out: Optional[str] = Field(None, description="Alternative to is_inpatient: 'IN' or 'OUT'")
    worker_type: Optional[str] = Field(None, description="Worker type involved")
    
    # Optional entity data
    doctors: Optional[list[dict]] = Field(None, description="List of doctors (doctor_id, doctor_name)")
    employees: Optional[list[dict]] = Field(None, description="List of employees (employee_id, employee_name)")
    
    # Optional classification hierarchy
    # Note: subcategory_id and classification_id moved to required fields above
    
    # Optional severity/impact
    improvement_type: Optional[int] = Field(0, ge=0, le=1, description="Improvement opportunity (0=No, 1=Yes)")
    
    # Clinical Risk Classification
    # Note: clinical_risk_type_id and feedback_intent_type_id moved to required fields above
    
    # ML Training Fields (NEW - for embedding generation)
    feedback_type: Optional[int] = Field(None, ge=1, le=4, description="Feedback type (1=improvement Opportunity, 2=notice, 3=Critique Suggestion, 4=Other)")
    improvement_opportunity_type: Optional[int] = Field(None, ge=1, le=3, description="Improvement type (1=Ordinary, 2=RedFlag, 3=NeverEvent)")
    classification_ar: Optional[float] = Field(None, ge=0, le=10, description="Arabic classification confidence score")
    classification_en: Optional[int] = Field(None, ge=0, description="English classification code")


# ==================== ENDPOINTS ====================

@router.post("/add")
async def add_record(
    request: CreateRecordRequest = Body(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Create a new incident/feedback record.
    
    **Required Fields (ALL MUST BE PROVIDED):**
    - complaint_text: Full complaint/incident description
    - feedback_received_date: Date the feedback was received (YYYY-MM-DD)
    - issuing_department_id: Issuing department ID
    - domain_id: Domain ID (1=Clinical, 2=Management, 3=Relational)
    - category_id: Category ID
    - subcategory_id: Subcategory ID
    - classification_id: Classification ID
    - severity_id: Severity level ID (1=High, 2=Low, 3=Medium)
    - stage_id: Care stage ID (1=Examination & Diagnosis, 2=Admissions, 4=Care on the Ward, 6=Discharge/Transfer, 8=Operation/Procedure, 9=Unspecified)
    - harm_id: Harm level ID (1=Severe Harm, 2=Death, 3=High Severe, 4=Minor Harm, 5=Moderate Harm, 6=No Harm)
    - clinical_risk_type_id: Clinical risk type (1=Standard, 2=Red Flag, 3=Never Event)
    - feedback_intent_type_id: Feedback intent type ID
    
    **Optional Fields:**
    - immediate_action: Immediate actions taken
    - taken_action: Follow-up actions taken
    - patient_name: Patient name
    - target_department_ids: List of target department IDs
    - source_id: Feedback source ID
    - building_id: Building ID
    - explanation_status_id: Explanation status ID
    - is_inpatient: Boolean (True=inpatient, False=outpatient, default=True)
    - doctors: List of doctors [{doctor_id, doctor_name}, ...]
    - employees: List of employees [{employee_id, employee_name}, ...]
    - feedback_type: Feedback type (1=Improvement, 2=Notice, 3=Critique, 4=Other)
    - improvement_opportunity_type: Improvement type (1=Ordinary, 2=RedFlag, 3=NeverEvent)
    - classification_ar: Arabic classification confidence (0-10)
    - classification_en: English classification code
    
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
      "issuing_department_id": 5,
      "domain_id": 1,
      "category_id": 2,
      "subcategory_id": 6,
      "classification_id": 10,
      "severity_id": 2,
      "stage_id": 1,
      "harm_id": 4,
      "clinical_risk_type_id": 2,
      "feedback_intent_type_id": 1,
      "immediate_action": "تم توفير الرعاية الفورية",
      "taken_action": "تم اتخاذ إجراءات تصحيحية",
      "patient_name": "أحمد محمد",
      "target_department_ids": [3, 4],
      "source_id": 1,
      "building_id": 2,
      "is_inpatient": true,
      "doctors": [
        {"doctor_id": 101, "doctor_name": "د. محمد علي"}
      ],
      "feedback_type": 1,
      "improvement_opportunity_type": 2,
      "classification_ar": 8.5,
      "classification_en": 78
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
    require_logged_in(current_user)
    
    try:
        # Convert request to dictionary
        data = request.model_dump()
        
        # Map frontend field names to backend field names
        # NOTE: is_inpatient must be boolean (true/false) - no "IN"/"OUT" strings
        
        if 'building' in data and isinstance(data['building'], str):
            # Convert building code to building_id if needed
            data['building_code'] = data['building']
            del data['building']
        
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


@router.get("/{record_id}")
async def get_record(
    record_id: int = Path(..., gt=0, description="Record ID to retrieve"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get record details for editing.
    
    Returns all fields of a record including classification and metadata.
    
    **Example Request:**
    ```
    GET /api/records/156
    ```
    
    **Returns:**
    - All 12 required fields + optional fields
    - With database IDs and names
    - Formatted dates
    """
    require_logged_in(current_user)
    
    try:
        result = get_complaint_by_id(record_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NOT_FOUND",
                    "message": f"Record {record_id} not found",
                    "message_ar": f"لم يتم العثور على السجل {record_id}"
                }
            )
        
        return {
            "success": True,
            "record": result
        }
        
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


@router.put("/{record_id}")
async def update_record(
    record_id: int = Path(..., gt=0, description="Record ID to update"),
    request: CreateRecordRequest = Body(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Update an existing record.
    
    Accepts same payload as POST /api/records/add.
    Updates all fields and revalidates foreign keys.
    
    **Example Request:**
    ```
    PUT /api/records/156
    {
      "complaint_text": "Updated complaint text",
      "feedback_received_date": "2024-12-15",
      "issuing_department_id": 5,
      ...all 12 required fields...
    }
    ```
    
    **Returns:**
    - success: true/false
    - message: Update confirmation
    - record_id: Updated record ID
    - updated_at: Update timestamp
    """
    require_logged_in(current_user)
    
    try:
        # Get existing record first
        existing_record = get_complaint_by_id(record_id)
        
        if not existing_record:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NOT_FOUND",
                    "message": f"Record {record_id} not found",
                    "message_ar": f"لم يتم العثور على السجل {record_id}"
                }
            )
        
        # Convert request to dictionary
        data = request.model_dump()
        
        # Map frontend field names to backend field names
        # NOTE: is_inpatient must be boolean (true/false) - no "IN"/"OUT" strings
        
        if 'building' in data and isinstance(data['building'], str):
            # Convert building code to building_id if needed
            # For now, just rename the field (building_code expected by service)
            data['building_code'] = data['building']
            del data['building']
        
        # Add record_id to data for update operation
        data['id'] = record_id
        
        # Import update function
        from ..services.insert_service import update_record as update_record_service
        
        # Call service to update record
        result = update_record_service(record_id, data)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": result.get("error", "UPDATE_FAILED"),
                    "message": result.get("message", "Failed to update record"),
                    "message_ar": result.get("message_ar", "فشل في تحديث السجل"),
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
async def test_records_endpoint(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Test endpoint to verify records service is operational.
    """
    require_logged_in(current_user)
    
    return {
        "status": "operational",
        "service": "records",
        "message": "Records service is running. Use POST /api/records/add to create new records."
    }


# ==================== SEARCH ENDPOINTS ====================

@router.get("/search/patients")
async def search_patients_endpoint(
    q: str = Query(..., min_length=1, description="Search query (patient name, document number, or medical file number)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results (1-100)"),
    current_user: CurrentUser = Depends(get_current_user)
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
    require_logged_in(current_user)
    
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
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results (1-100)"),
    current_user: CurrentUser = Depends(get_current_user)
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
    require_logged_in(current_user)
    
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
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results (1-100)"),
    current_user: CurrentUser = Depends(get_current_user)
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
    require_logged_in(current_user)
    
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
async def get_patient_endpoint(
    patient_admission_id: int,
    current_user: CurrentUser = Depends(get_current_user)
):
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
    require_logged_in(current_user)
    
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
async def get_doctor_endpoint(
    doctor_id: int,
    current_user: CurrentUser = Depends(get_current_user)
):
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
    require_logged_in(current_user)
    
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
async def get_employee_endpoint(
    employee_id: int,
    current_user: CurrentUser = Depends(get_current_user)
):
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
    require_logged_in(current_user)
    
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
