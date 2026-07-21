"""
Patients Router
API endpoints for patient history page.
"""

from fastapi import APIRouter, HTTPException, Query, status
from typing import Optional
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field
import io

from ..services.patients_service import (
    search_patients_service,
    get_patient_profile_service,
    get_patient_incidents_service,
    get_incident_details_service,
    get_patient_full_history_service,
    export_patient_history_service,
    create_patient_service,
    get_all_reserve_patients_service,
    PatientServiceError
)


def _external_unavailable_http_exception(e: "PatientServiceError") -> HTTPException:
    """
    A patient_id resolved to an EXTERNAL (Hospital Directory API) identity,
    but the API call failed for a reason distinct from "not found" — surface
    this as 503, never silently as a 404/empty result (see Session C1
    requirements: a network failure must be a clear "external search
    unavailable" state).
    """
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": "EXTERNAL_PATIENT_UNAVAILABLE",
            "message": f"Hospital Directory API unavailable ({e.status}): {str(e)}",
        },
    )


router = APIRouter(prefix="/api/patients", tags=["Patients - History"])


# ==================== REQUEST/RESPONSE MODELS ====================

class CreatePatientRequest(BaseModel):
    """Request model for creating a new patient."""
    first_name: str = Field(
        ...,
        min_length=2,
        max_length=150,
        description="Patient's first name (required, 2-150 characters)",
        example="Ahmed"
    )
    middle_name: Optional[str] = Field(
        None,
        max_length=150,
        description="Patient's middle name (optional, max 150 characters)",
        example="Mohammed"
    )
    last_name: Optional[str] = Field(
        None,
        max_length=150,
        description="Patient's last name (optional, max 150 characters)",
        example="Al-Rashid"
    )
    mother_name: Optional[str] = Field(
        None,
        max_length=150,
        description="Patient's mother name (optional, max 150 characters)",
        example="Fatima"
    )
    phone_number: Optional[str] = Field(
        None,
        max_length=50,
        description="Primary phone number (optional, min 7 digits)",
        example="0501234567"
    )
    phone_number2: Optional[str] = Field(
        None,
        max_length=50,
        description="Secondary phone number (optional, min 7 digits)",
        example="0509876543"
    )
    birth_date: Optional[str] = Field(
        None,
        description="Date of birth in YYYY-MM-DD format (optional)",
        example="1985-05-15"
    )
    sex: Optional[str] = Field(
        None,
        description="Patient sex: M, F, Male, or Female (optional, normalized to M/F)",
        example="M"
    )
    document_number: Optional[str] = Field(
        None,
        max_length=100,
        description="ID/Document number (optional, alphanumeric + hyphens)",
        example="1234567890"
    )
    medical_file_number: Optional[str] = Field(
        None,
        max_length=100,
        description="Medical record number (optional, alphanumeric + hyphens)",
        example="MRN-2026-001234"
    )
    spouse: Optional[str] = Field(
        None,
        max_length=150,
        description="Spouse name (optional, max 150 characters)",
        example="Sara Al-Ahmad"
    )
    address_line1: Optional[str] = Field(
        None,
        max_length=300,
        description="Primary address line (optional, max 300 characters)",
        example="123 King Fahd Road, Riyadh"
    )
    address_line2: Optional[str] = Field(
        None,
        max_length=300,
        description="Secondary address line (optional, max 300 characters)",
        example="Building 5, Apartment 201"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "first_name": "Ahmed",
                "middle_name": "Mohammed",
                "last_name": "Al-Rashid",
                "mother_name": "Fatima",
                "phone_number": "0501234567",
                "phone_number2": "0509876543",
                "birth_date": "1985-05-15",
                "sex": "M",
                "document_number": "1234567890",
                "medical_file_number": "MRN-2026-001234",
                "spouse": "Sara Al-Ahmad",
                "address_line1": "123 King Fahd Road, Riyadh",
                "address_line2": "Building 5, Apartment 201"
            }
        }


class PatientResponse(BaseModel):
    """Patient object in responses."""
    patient_admission_id: int
    full_name: str
    first_name: str
    middle_name: Optional[str]
    last_name: Optional[str]
    mother_name: Optional[str]
    phone_number: Optional[str]
    phone_number2: Optional[str]
    birth_date: Optional[str]
    sex: Optional[str]
    document_number: Optional[str]
    medical_file_number: Optional[str]
    spouse: Optional[str]
    address_line1: Optional[str]
    address_line2: Optional[str]
    source: str


class CreatePatientResponse(BaseModel):
    """Response model for patient creation."""
    success: bool
    message: str
    message_ar: str
    patient: PatientResponse


router = APIRouter(prefix="/api/patients", tags=["Patients - History"])


# ==================== B.1 SEARCH PATIENTS ====================

# ==================== CREATE PATIENT ====================

@router.post(
    "/create",
    response_model=CreatePatientResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create New Patient",
    responses={
        201: {
            "description": "Patient created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "Patient 'Ahmed Mohammed Al-Rashid' created successfully with ID 100001",
                        "message_ar": "تم إنشاء المريض 'Ahmed Mohammed Al-Rashid' بنجاح بالرقم 100001",
                        "patient": {
                            "patient_admission_id": 100001,
                            "full_name": "Ahmed Mohammed Al-Rashid",
                            "first_name": "Ahmed",
                            "middle_name": "Mohammed",
                            "last_name": "Al-Rashid",
                            "mother_name": "Fatima",
                            "phone_number": "0501234567",
                            "phone_number2": "0509876543",
                            "birth_date": "1985-05-15",
                            "sex": "M",
                            "document_number": "1234567890",
                            "medical_file_number": "MRN-2026-001234",
                            "spouse": "Sara Al-Ahmad",
                            "address_line1": "123 King Fahd Road, Riyadh",
                            "address_line2": "Building 5, Apartment 201",
                            "source": "reserve"
                        }
                    }
                }
            }
        },
        400: {"description": "Validation error - Invalid input"},
        409: {"description": "Conflict - Patient already exists"},
        500: {"description": "Internal server error"}
    }
)
async def create_patient(request: CreatePatientRequest):
    """
    Create a new patient in the reserve table.
    
    This endpoint allows you to add patients that don't exist in the hospital's
    main database. Created patients are stored in a separate reserve table and
    will appear in all search and profile queries alongside hospital patients.
    
    **Request Body:**
    - `first_name`: **Required**. Patient's first name (2-150 characters)
    - `middle_name`: Optional. Patient's middle name (max 150 characters)
    - `last_name`: Optional. Patient's last name (max 150 characters)
    - `mother_name`: Optional. Patient's mother name (max 150 characters)
    - `phone_number`: Optional. Primary phone (min 7 digits, max 50 characters)
    - `phone_number2`: Optional. Secondary phone (min 7 digits, max 50 characters)
    - `birth_date`: Optional. Date of birth in YYYY-MM-DD format
    - `sex`: Optional. Patient sex (M, F, Male, or Female - normalized to M/F)
    - `document_number`: Optional. ID/Document number (alphanumeric + hyphens, max 100 chars)
    - `medical_file_number`: Optional. Medical record number (alphanumeric + hyphens, max 100 chars)
    - `spouse`: Optional. Spouse name (max 150 characters)
    - `address_line1`: Optional. Primary address (max 300 characters)
    - `address_line2`: Optional. Secondary address (max 300 characters)
    
    **Validation Rules:**
    - FirstName is required (2-150 characters)
    - Names must contain only letters, spaces, hyphens, apostrophes (supports Arabic)
    - Phone numbers must have at least 7 digits
    - BirthDate must be YYYY-MM-DD format, not in future, age < 150 years
    - SEX normalized: "Male"→"M", "Female"→"F"
    - Duplicate detection by FullName, DocumentNumber, or MedicalFileNumber
    - All whitespace automatically trimmed
    
    **Response:**
    - Returns created patient with auto-generated ID (starting at 100000)
    - Includes success message in English and Arabic
    - Patient will have source='reserve' to distinguish from hospital patients
    
    **Example Request:**
    ```json
    {
      "first_name": "Ahmed",
      "middle_name": "Mohammed",
      "last_name": "Al-Rashid",
      "mother_name": "Fatima",
      "phone_number": "0501234567",
      "birth_date": "1985-05-15",
      "sex": "M",
      "document_number": "1234567890",
      "medical_file_number": "MRN-2026-001234"
    }
    ```
    
    **Example Success Response (201):**
    ```json
    {
      "success": true,
      "message": "Patient 'Ahmed Mohammed Al-Rashid' created successfully with ID 100001",
      "message_ar": "تم إنشاء المريض 'Ahmed Mohammed Al-Rashid' بنجاح بالرقم 100001",
      "patient": {
        "patient_admission_id": 100001,
        "full_name": "Ahmed Mohammed Al-Rashid",
        "first_name": "Ahmed",
        "sex": "M",
        "source": "reserve",
        ...
      }
    }
    ```
    
    **Error Response (409 - Duplicate):**
    ```json
    {
      "detail": {
        "error": "DUPLICATE_PATIENT",
        "message": "Patient with FullName 'Ahmed Mohammed Al-Rashid' already exists in reserve",
        "message_ar": "المريض موجود بالفعل"
      }
    }
    ```
    
    **Error Response (400 - Validation):**
    ```json
    {
      "detail": {
        "error": "VALIDATION_ERROR",
        "message": "FirstName is required and must be at least 2 characters",
        "message_ar": "خطأ في التحقق من الصحة"
      }
    }
    ```
    """
    try:
        # Call service layer (handles all validation)
        patient = create_patient_service(
            first_name=request.first_name,
            middle_name=request.middle_name,
            last_name=request.last_name,
            mother_name=request.mother_name,
            phone_number=request.phone_number,
            phone_number2=request.phone_number2,
            birth_date=request.birth_date,
            sex=request.sex,
            document_number=request.document_number,
            medical_file_number=request.medical_file_number,
            spouse=request.spouse,
            address_line1=request.address_line1,
            address_line2=request.address_line2
        )
        
        # Format response
        return CreatePatientResponse(
            success=True,
            message=f"Patient '{patient['FullName']}' created successfully with ID {patient['PatientAdmissionID']}",
            message_ar=f"تم إنشاء المريض '{patient['FullName']}' بنجاح بالرقم {patient['PatientAdmissionID']}",
            patient=PatientResponse(
                patient_admission_id=patient['PatientAdmissionID'],
                full_name=patient['FullName'],
                first_name=patient['FirstName'],
                middle_name=patient.get('MiddleName'),
                last_name=patient.get('LastName'),
                mother_name=patient.get('MotherName'),
                phone_number=patient.get('PhoneNumber1'),
                phone_number2=patient.get('PhoneNumber2'),
                birth_date=patient.get('BirthDate'),
                sex=patient.get('SEX'),
                document_number=patient.get('DocumentNumber'),
                medical_file_number=patient.get('MedicalFileNumber'),
                spouse=patient.get('Spouse'),
                address_line1=patient.get('AddressLine1'),
                address_line2=patient.get('AddressLine2'),
                source=patient['Source']
            )
        )
        
    except ValueError as e:
        # Validation errors (400) or duplicate errors (409)
        error_msg = str(e)
        if "already exists" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": "DUPLICATE_PATIENT",
                    "message": error_msg,
                    "message_ar": "المريض موجود بالفعل"
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "VALIDATION_ERROR",
                    "message": error_msg,
                    "message_ar": "خطأ في التحقق من الصحة"
                }
            )
    except Exception as e:
        # Unexpected errors (500)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Failed to create patient: {str(e)}",
                "message_ar": "خطأ داخلي في النظام"
            }
        )


# ==================== GET ALL RESERVE PATIENTS ====================

@router.get(
    "/reserve",
    summary="Get All Reserve Patients",
    responses={
        200: {
            "description": "Successfully retrieved reserve patients",
            "content": {
                "application/json": {
                    "example": {
                        "patients": [
                            {
                                "patient_admission_id": 100001,
                                "full_name": "Ahmed Mohammed Al-Rashid",
                                "first_name": "Ahmed",
                                "middle_name": "Mohammed",
                                "last_name": "Al-Rashid",
                                "mother_name": "Fatima",
                                "phone_number": "0501234567",
                                "phone_number2": "0509876543",
                                "birth_date": "1985-05-15",
                                "sex": "M",
                                "document_number": "1234567890",
                                "medical_file_number": "MRN-2026-001234",
                                "spouse": "Sara",
                                "address_line1": "123 King Fahd Road",
                                "address_line2": "Building 5, Apt 201",
                                "created_at": "2026-01-21 10:30:00",
                                "source": "reserve"
                            }
                        ],
                        "total": 42,
                        "count": 1,
                        "limit": 100,
                        "offset": 0
                    }
                }
            }
        },
        500: {"description": "Internal server error"}
    }
)
async def get_all_reserve_patients_endpoint(
    limit: int = Query(100, ge=1, le=200, description="Maximum records per page (default: 100, max: 200)"),
    offset: int = Query(0, ge=0, description="Number of records to skip for pagination (default: 0)"),
    order_by: str = Query('created_at', description="Sort order: 'created_at' (newest first) or 'name' (alphabetical)")
):
    """
    Get all patients from the reserve table (user-created patients only).
    
    This endpoint returns ONLY patients that were manually added through the 
    create patient endpoint. It does NOT include hospital patients from the 
    main database.
    
    **Query Parameters:**
    - `limit`: Maximum number of records per page (default: 100, max: 200)
    - `offset`: Number of records to skip for pagination (default: 0)
    - `order_by`: Sort order
        - `'created_at'` - Newest patients first (default)
        - `'name'` - Alphabetical by full name
    
    **Use Cases:**
    - Display list of all manually added patients
    - Frontend patient management page
    - Export/review user-created patients
    - Pagination for large datasets
    
    **Pagination Example:**
    ```
    # First page (records 1-100)
    GET /api/patients/reserve?limit=100&offset=0
    
    # Second page (records 101-200)
    GET /api/patients/reserve?limit=100&offset=100
    
    # Third page (records 201-300)
    GET /api/patients/reserve?limit=100&offset=200
    ```
    
    **Response Structure:**
    ```json
    {
      "patients": [
        {
          "patient_admission_id": 100001,
          "full_name": "Ahmed Mohammed Al-Rashid",
          "first_name": "Ahmed",
          "middle_name": "Mohammed",
          "last_name": "Al-Rashid",
          "mother_name": "Fatima",
          "phone_number": "0501234567",
          "phone_number2": "0509876543",
          "birth_date": "1985-05-15",
          "sex": "M",
          "document_number": "1234567890",
          "medical_file_number": "MRN-2026-001234",
          "spouse": "Sara",
          "address_line1": "123 King Fahd Road",
          "address_line2": "Building 5, Apt 201",
          "created_at": "2026-01-21 10:30:00",
          "source": "reserve"
        }
      ],
      "total": 42,          // Total number of reserve patients in database
      "count": 1,           // Number of patients in this response
      "limit": 100,         // Applied limit
      "offset": 0           // Applied offset
    }
    ```
    
    **Notes:**
    - All patients will have `source: "reserve"`
    - Patient IDs start at 100000 to distinguish from hospital patients
    - Empty fields will be `null`
    - `created_at` is when the patient was added to the system
    - Total count is always returned for pagination logic
    
    **Example Requests:**
    ```
    # Get first 50 patients (newest first)
    GET /api/patients/reserve?limit=50&order_by=created_at
    
    # Get patients alphabetically
    GET /api/patients/reserve?order_by=name
    
    # Get second page of 100 patients
    GET /api/patients/reserve?limit=100&offset=100
    ```
    """
    try:
        result = get_all_reserve_patients_service(
            limit=limit,
            offset=offset,
            order_by=order_by
        )
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Failed to retrieve reserve patients: {str(e)}",
                "message_ar": "فشل في استرجاع المرضى المحفوظين"
            }
        )


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
async def get_patient_profile_endpoint(patient_id: str):
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
    except HTTPException:
        raise
    except PatientServiceError as e:
        raise _external_unavailable_http_exception(e)
    except Exception as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get profile: {str(e)}")


# ==================== B.3 GET PATIENT INCIDENTS ====================

@router.get("/{patient_id}/incidents")
async def get_patient_incidents_endpoint(
    patient_id: str,
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
    except PatientServiceError as e:
        raise _external_unavailable_http_exception(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get incidents: {str(e)}")


# ==================== B.4 GET INCIDENT DETAILS ====================

@router.get("/{patient_id}/incidents/{incident_id}")
async def get_incident_details_endpoint(patient_id: str, incident_id: int):
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
    patient_id: str,
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
    except PatientServiceError as e:
        raise _external_unavailable_http_exception(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get full history: {str(e)}")


# ==================== B.5 EXPORT ====================

@router.get("/{patient_id}/export")
async def export_patient_history_endpoint(
    patient_id: str,
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
    except PatientServiceError as e:
        raise _external_unavailable_http_exception(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
