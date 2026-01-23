"""
Doctors Router
API endpoints for doctor profiles, statistics, and analytics.
"""

from fastapi import APIRouter, Query, Path, HTTPException, status
from typing import Optional
from pydantic import BaseModel, Field

from ..services.doctors_service import DoctorService


router = APIRouter(prefix="/api/doctors", tags=["Doctors"])


# ==================== REQUEST/RESPONSE MODELS ====================

class CreateDoctorRequest(BaseModel):
    """Request model for creating a new doctor."""
    doctor_name: str = Field(
        ..., 
        min_length=3, 
        max_length=200,
        description="Doctor's full name (3-200 characters)",
        example="Dr. Ahmed Al-Mansour"
    )
    specialty: Optional[str] = Field(
        None,
        max_length=200,
        description="Medical specialty (optional)",
        example="Interventional Cardiology"
    )
    is_active: bool = Field(
        True,
        description="Active status (default: true)",
        example=True
    )
    source_system: str = Field(
        "MANUAL",
        max_length=100,
        description="Source system identifier (default: MANUAL)",
        example="MANUAL"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "doctor_name": "Dr. Ahmed Al-Mansour",
                "specialty": "Interventional Cardiology",
                "is_active": True,
                "source_system": "MANUAL"
            }
        }


class DoctorResponse(BaseModel):
    """Doctor object in responses."""
    id: int
    name_en: str
    name_ar: str
    specialty: str
    status: str
    source: str
    source_system: str
    last_synced_at: str


class CreateDoctorResponse(BaseModel):
    """Response model for doctor creation."""
    success: bool
    message: str
    message_ar: str
    doctor: DoctorResponse


class DoctorSearchResponse(BaseModel):
    """Response model for doctor search."""
    id: int
    name_en: str
    name_ar: str
    specialty: str
    status: str
    source: str


# ==================== A.1: CREATE DOCTOR ====================

@router.post(
    "",
    response_model=CreateDoctorResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new doctor",
    responses={
        201: {
            "description": "Doctor created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "Doctor 'Dr. Ahmed Al-Mansour' created successfully",
                        "message_ar": "تم إنشاء الطبيب 'Dr. Ahmed Al-Mansour' بنجاح",
                        "doctor": {
                            "id": 1,
                            "name_en": "Dr. Ahmed Al-Mansour",
                            "name_ar": "Dr. Ahmed Al-Mansour",
                            "specialty": "Interventional Cardiology",
                            "status": "active",
                            "source": "reserve",
                            "source_system": "MANUAL",
                            "last_synced_at": "2026-01-20 14:30:00"
                        }
                    }
                }
            }
        },
        400: {"description": "Validation error - Invalid input"},
        409: {"description": "Conflict - Doctor already exists"},
        500: {"description": "Internal server error"}
    }
)
async def create_doctor(request: CreateDoctorRequest):
    """
    Create a new doctor in the reserve table.
    
    This endpoint allows you to add doctors that don't exist in the hospital's
    main database. Created doctors are stored in a separate reserve table and
    will appear in all search and profile queries alongside hospital doctors.
    
    **Request Body:**
    - `doctor_name`: Required. Doctor's full name (3-200 characters)
    - `specialty`: Optional. Medical specialty (max 200 characters)
    - `is_active`: Optional. Active status (default: true)
    - `source_system`: Optional. Source identifier (default: "MANUAL")
    
    **Validation Rules:**
    - Doctor name must be 3-200 characters
    - Doctor name must be unique (not already in reserve table)
    - Specialty max 200 characters
    - Whitespace is automatically trimmed
    
    **Response:**
    - Returns created doctor with generated ID
    - Includes success message in English and Arabic
    - Doctor will have source='reserve' to distinguish from hospital doctors
    
    **Example Request:**
    ```json
    {
      "doctor_name": "Dr. Ahmed Al-Mansour",
      "specialty": "Interventional Cardiology",
      "is_active": true,
      "source_system": "MANUAL"
    }
    ```
    
    **Example Success Response (201):**
    ```json
    {
      "success": true,
      "message": "Doctor 'Dr. Ahmed Al-Mansour' created successfully",
      "message_ar": "تم إنشاء الطبيب 'Dr. Ahmed Al-Mansour' بنجاح",
      "doctor": {
        "id": 1,
        "name_en": "Dr. Ahmed Al-Mansour",
        "name_ar": "Dr. Ahmed Al-Mansour",
        "specialty": "Interventional Cardiology",
        "status": "active",
        "source": "reserve",
        "source_system": "MANUAL",
        "last_synced_at": "2026-01-20 14:30:00"
      }
    }
    ```
    
    **Error Response (409 - Duplicate):**
    ```json
    {
      "error": "DUPLICATE_DOCTOR",
      "message": "Doctor with name 'Dr. Ahmed Al-Mansour' already exists",
      "message_ar": "الطبيب موجود بالفعل"
    }
    ```
    
    **Error Response (400 - Validation):**
    ```json
    {
      "error": "VALIDATION_ERROR",
      "message": "Doctor name must be at least 3 characters",
      "message_ar": "خطأ في التحقق من الصحة"
    }
    ```
    """
    try:
        result = DoctorService.create_doctor(
            doctor_name=request.doctor_name,
            specialty=request.specialty,
            is_active=request.is_active,
            source_system=request.source_system
        )
        return result
        
    except ValueError as ve:
        # Validation errors or duplicates
        error_msg = str(ve)
        
        if "already exists" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": "DUPLICATE_DOCTOR",
                    "message": error_msg,
                    "message_ar": "الطبيب موجود بالفعل",
                    "field": "doctor_name"
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Failed to create doctor: {str(e)}",
                "message_ar": "فشل في إنشاء الطبيب"
            }
        )


# ==================== A.2: GET RESERVE DOCTORS ====================

@router.get(
    "/reserve",
    summary="Get all reserve doctors",
    responses={
        200: {
            "description": "Successfully retrieved reserve doctors",
            "content": {
                "application/json": {
                    "example": {
                        "doctors": [
                            {
                                "id": 1,
                                "name_en": "Dr. Ahmed Al-Mansour",
                                "name_ar": "Dr. Ahmed Al-Mansour",
                                "specialty": "Interventional Cardiology",
                                "status": "active",
                                "source": "reserve",
                                "source_system": "MANUAL",
                                "last_synced_at": "2026-01-20 14:30:00"
                            }
                        ],
                        "total": 1,
                        "message": "Found 1 reserve doctor(s)",
                        "message_ar": "تم العثور على 1 طبيب في الجدول الاحتياطي"
                    }
                }
            }
        },
        500: {"description": "Internal server error"}
    }
)
async def get_reserve_doctors(
    limit: int = Query(100, ge=1, le=500, description="Max results to return")
):
    """
    Get all doctors from the reserve table only.
    
    This endpoint returns ONLY user-created doctors (those added via POST /api/doctors).
    Hospital doctors are NOT included in this list.
    
    **Query Parameters:**
    - `limit`: Max results (default: 100, max: 500)
    
    **Response:**
    - Returns array of reserve doctors with full details
    - Each doctor has `source='reserve'` to indicate it's user-created
    - Ordered by most recently created first
    
    **Example Response:**
    ```json
    {
      "doctors": [
        {
          "id": 1,
          "name_en": "Dr. Ahmed Al-Mansour",
          "name_ar": "Dr. Ahmed Al-Mansour",
          "specialty": "Interventional Cardiology",
          "status": "active",
          "source": "reserve",
          "source_system": "MANUAL",
          "last_synced_at": "2026-01-20 14:30:00"
        }
      ],
      "total": 1,
      "message": "Found 1 reserve doctor(s)",
      "message_ar": "تم العثور على 1 طبيب في الجدول الاحتياطي"
    }
    ```
    """
    try:
        result = DoctorService.get_reserve_doctors(limit=limit)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "FETCH_RESERVE_DOCTORS_FAILED",
                "message": f"Failed to fetch reserve doctors: {str(e)}",
                "message_ar": "فشل في جلب الأطباء من الجدول الاحتياطي"
            }
        )


# ==================== B.1: DOCTOR SEARCH / LIST ====================

@router.get("")
async def search_doctors(
    query: Optional[str] = Query(None, description="Search by name or employee ID"),
    department: Optional[str] = Query(None, description="Filter by department"),
    status: Optional[str] = Query(None, description="Filter by status: active, inactive, suspended"),
    limit: int = Query(50, ge=1, le=500, description="Max results to return")
):
    """
    Search for doctors by name, employee ID, or department.
    
    Supports partial matching on name and employee ID.
    Returns limited fields for performance in search/autocomplete.
    
    **Query Parameters:**
    - `query`: Search term (name or employee ID)
    - `department`: Department name filter
    - `status`: active, inactive, or suspended
    - `limit`: Max results (default: 50, max: 500)
    
    **Example Response:**
    ```json
    {
      "doctors": [
        {
          "id": 1,
          "employee_id": "DOC-2024-001",
          "name_en": "Dr. Ahmed Mohamed",
          "name_ar": "د. أحمد محمد",
          "department": "Cardiology",
          "specialty": "Cardiologist",
          "hire_date": "2020-01-15",
          "status": "active"
        }
      ],
      "total": 45
    }
    ```
    """
    try:
        result = DoctorService.search_doctors(
            query=query,
            department=department,
            status=status,
            limit=limit
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "SEARCH_FAILED",
            "message": str(e),
            "message_ar": f"فشل البحث: {str(e)}"
        })


# ==================== B.2: DOCTOR PROFILE ====================

@router.get("/{doctor_id}/profile")
async def get_doctor_profile(
    doctor_id: int = Path(..., gt=0, description="Doctor ID")
):
    """
    Fetch detailed profile information for a specific doctor.
    
    Returns comprehensive doctor metadata including contact info and years of service.
    
    **Path Parameters:**
    - `doctor_id`: Doctor's internal ID
    
    **Example Response:**
    ```json
    {
      "id": 1,
      "employee_id": "DOC-2024-001",
      "name_en": "Dr. Ahmed Mohamed",
      "name_ar": "د. أحمد محمد",
      "department": "Cardiology",
      "specialty": "Cardiologist",
      "hire_date": "2020-01-15",
      "status": "active",
      "years_of_service": 5,
      "email": "ahmed.mohamed@hospital.local",
      "phone": "+966XXXXXXXXX",
      "license_number": "LIC-12345"
    }
    ```
    """
    try:
        result = DoctorService.get_doctor_profile(doctor_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail={
            "error": "DOCTOR_NOT_FOUND",
            "message": str(e),
            "message_ar": "الدكتور غير موجود"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "PROFILE_FETCH_FAILED",
            "message": str(e),
            "message_ar": f"فشل جلب الملف الشخصي: {str(e)}"
        })


# ==================== B.3: DOCTOR STATISTICS ====================

@router.get("/{doctor_id}/statistics")
async def get_doctor_statistics(
    doctor_id: int = Path(..., gt=0, description="Doctor ID"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD), default: 6 months ago"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD), default: today")
):
    """
    Fetch aggregated incident statistics for a doctor.
    
    Returns counts of incidents by severity and red flags.
    Default period is last 6 months if not specified.
    
    **Path Parameters:**
    - `doctor_id`: Doctor's internal ID
    
    **Query Parameters:**
    - `from_date`: Start date (YYYY-MM-DD, optional)
    - `to_date`: End date (YYYY-MM-DD, optional)
    
    **Example Response:**
    ```json
    {
      "statistics": {
        "total": 12,
        "high": 3,
        "medium": 7,
        "low": 2,
        "red_flags": 1
      },
      "period": {
        "from": "2024-06-01",
        "to": "2024-12-17"
      }
    }
    ```
    """
    try:
        result = DoctorService.get_doctor_statistics(
            doctor_id=doctor_id,
            from_date=from_date,
            to_date=to_date
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 400, detail={
            "error": "INVALID_REQUEST",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "STATISTICS_FETCH_FAILED",
            "message": str(e),
            "message_ar": f"فشل جلب الإحصائيات: {str(e)}"
        })


# ==================== B.4: DOCTOR ANALYTICS ====================

@router.get("/{doctor_id}/analytics")
async def get_doctor_analytics(
    doctor_id: int = Path(..., gt=0, description="Doctor's ID"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD, optional)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD, optional)")
):
    """
    Get doctor analytics: category breakdown and monthly trends.
    
    Combines category-wise incident distribution with monthly trend data.
    Useful for analyzing patterns over time.
    
    **Path Parameters:**
    - `doctor_id`: Doctor's internal ID
    
    **Query Parameters:**
    - `from_date`: Start date (YYYY-MM-DD, optional, defaults to 6 months ago)
    - `to_date`: End date (YYYY-MM-DD, optional, defaults to today)
    
    **Example Response:**
    ```json
    {
      "categoryBreakdown": [
        {"category": "Medication Error", "count": 5},
        {"category": "Patient Safety", "count": 3}
      ],
      "monthlyTrend": [
        {"month": "Jun", "count": 2},
        {"month": "Jul", "count": 3},
        {"month": "Aug", "count": 0}
      ],
      "period": {
        "from": "2024-06-01",
        "to": "2024-12-17"
      }
    }
    ```
    """
    try:
        result = DoctorService.get_doctor_analytics(
            doctor_id=doctor_id,
            from_date=from_date,
            to_date=to_date
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 400, detail={
            "error": "INVALID_REQUEST",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "ANALYTICS_FETCH_FAILED",
            "message": str(e),
            "message_ar": f"فشل جلب التحليلات: {str(e)}"
        })


# ==================== B.5: DOCTOR INCIDENTS ====================

@router.get("/{doctor_id}/incidents")
async def get_doctor_incidents(
    doctor_id: int = Path(..., gt=0, description="Doctor's ID"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD, optional)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD, optional)"),
    severity: Optional[str] = Query(None, description="Filter by severity: HIGH, MEDIUM, LOW"),
    status: Optional[str] = Query(None, description="Filter by status"),
    red_flags_only: bool = Query(False, description="Only incidents with red flags"),
    limit: int = Query(100, ge=1, le=500, description="Max incidents per page"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """
    Get doctor's incidents with filtering and pagination.
    
    Retrieve detailed list of incidents involving a doctor with optional filters.
    
    **Path Parameters:**
    - `doctor_id`: Doctor's internal ID
    
    **Query Parameters:**
    - `from_date`: Start date (YYYY-MM-DD, optional, defaults to 6 months ago)
    - `to_date`: End date (YYYY-MM-DD, optional, defaults to today)
    - `severity`: Filter by HIGH, MEDIUM, LOW (optional)
    - `status`: Filter by case status (optional)
    - `red_flags_only`: Only return incidents with red flags (default: false)
    - `limit`: Max results per page (default: 100, max: 500)
    - `offset`: Pagination offset (default: 0)
    
    **Example Response:**
    ```json
    {
      "incidents": [
        {
          "id": 123,
          "incident_number": "INC-2024-001",
          "patient_id": "P12345",
          "category": "Medication Error",
          "severity": "HIGH",
          "status": "Open",
          "has_red_flag": true,
          "reported_date": "2024-06-15"
        }
      ],
      "total": 15,
      "limit": 100,
      "offset": 0
    }
    ```
    """
    try:
        result = DoctorService.get_doctor_incidents(
            doctor_id=doctor_id,
            from_date=from_date,
            to_date=to_date,
            severity=severity,
            status=status,
            red_flags_only=red_flags_only,
            limit=limit,
            offset=offset
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 400, detail={
            "error": "INVALID_REQUEST",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "INCIDENTS_FETCH_FAILED",
            "message": str(e),
            "message_ar": f"فشل جلب الحوادث: {str(e)}"
        })


# ==================== B.6: DOCTOR FULL REPORT ====================

@router.get("/{doctor_id}/full-report")
async def get_doctor_full_report(
    doctor_id: int = Path(..., gt=0, description="Doctor's ID"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD, optional)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD, optional)"),
    severity: Optional[str] = Query(None, description="Filter incidents by severity: HIGH, MEDIUM, LOW"),
    status: Optional[str] = Query(None, description="Filter incidents by status"),
    red_flags_only: bool = Query(False, description="Only incidents with red flags"),
    limit: int = Query(100, ge=1, le=500, description="Max incidents per page"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """
    Get comprehensive doctor report with all data in one call.
    
    Combines profile, statistics, analytics, and incidents into a single response.
    Useful for complete dashboard views or detailed doctor reports.
    
    **Path Parameters:**
    - `doctor_id`: Doctor's internal ID
    
    **Query Parameters:**
    - `from_date`: Start date for incidents and analytics (YYYY-MM-DD, optional)
    - `to_date`: End date for incidents and analytics (YYYY-MM-DD, optional)
    - `severity`: Filter incidents by HIGH, MEDIUM, LOW (optional)
    - `status`: Filter incidents by status (optional)
    - `red_flags_only`: Only return incidents with red flags (default: false)
    - `limit`: Max incidents per page (default: 100, max: 500)
    - `offset`: Pagination offset (default: 0)
    
    **Example Response:**
    ```json
    {
      "profile": {
        "id": 42,
        "employee_id": "EMP001",
        "name_en": "Dr. John Smith",
        "name_ar": "د. جون سميث",
        "email": "john@hospital.com",
        "phone": "+966501234567",
        "department": "Cardiology",
        "specialty": "Interventional Cardiology",
        "license_number": "LIC123456",
        "hire_date": "2018-05-15",
        "status": "active",
        "years_of_service": 6
      },
      "statistics": {
        "total": 12,
        "high": 3,
        "medium": 7,
        "low": 2,
        "red_flags": 1
      },
      "analytics": {
        "categoryBreakdown": [
          {"category": "Medication Error", "count": 5},
          {"category": "Patient Safety", "count": 3}
        ],
        "monthlyTrend": [
          {"month": "Jun", "count": 2},
          {"month": "Jul", "count": 3}
        ]
      },
      "incidents": [
        {
          "id": 123,
          "incident_number": "INC-2024-001",
          "patient_id": "P12345",
          "category": "Medication Error",
          "severity": "HIGH",
          "status": "Open",
          "has_red_flag": true,
          "reported_date": "2024-06-15"
        }
      ],
      "incidentsPagination": {
        "total": 15,
        "limit": 100,
        "offset": 0
      },
      "period": {
        "from": "2024-06-01",
        "to": "2024-12-17"
      }
    }
    ```
    """
    try:
        result = DoctorService.get_doctor_full_report(
            doctor_id=doctor_id,
            from_date=from_date,
            to_date=to_date,
            severity=severity,
            status=status,
            red_flags_only=red_flags_only,
            limit=limit,
            offset=offset
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 400, detail={
            "error": "INVALID_REQUEST",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "FULL_REPORT_GENERATION_FAILED",
            "message": str(e),
            "message_ar": f"فشل إنشاء التقرير الكامل: {str(e)}"
        })


# ==================== HEALTH CHECK ====================

@router.get("/health-check/check")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Doctors API is running"
    }
