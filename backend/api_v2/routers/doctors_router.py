"""
Doctor Router (API v2)
Phase B — B-B1 — Doctor router exposed under API v2 namespace.

This router exposes all existing doctor endpoints under /api/v2/doctors
without rewriting business logic. Thin wrapper that delegates to service layer.

Endpoints exposed:
- POST /api/v2/doctors - Create doctor
- GET /api/v2/doctors/reserve - Get reserve doctors
- GET /api/v2/doctors - Search doctors
- GET /api/v2/doctors/{doctor_id}/profile - Doctor profile
- GET /api/v2/doctors/{doctor_id}/statistics - Doctor statistics
- GET /api/v2/doctors/{doctor_id}/analytics - Doctor analytics
- GET /api/v2/doctors/{doctor_id}/incidents - Doctor incidents
- GET /api/v2/doctors/{doctor_id}/full-report - Full doctor report
- GET /api/v2/doctors/{doctor_id}/full-history - Full doctor history (mirrors patient)
- GET /api/v2/doctors/{doctor_id}/export - Export doctor history (csv/json/word)
- GET /api/v2/doctors/health-check/check - Health check

Security: All endpoints protected by authentication (where applicable).
"""

from fastapi import APIRouter, Query, Path, HTTPException, status
from fastapi.responses import StreamingResponse, Response
from typing import Optional, List
from pydantic import BaseModel, Field
from backend.api.services.doctors_service import DoctorService, get_doctor_full_history_service, export_doctor_history_service
from backend.api.services.search_service import search_doctors as search_doctors_service
from backend.api_v2.schemas.profile_schemas import DoctorProfileV2Response, EntityMeta


# ============================================================
# ROUTER DEFINITION (Phase B — B-B1)
# ============================================================
router = APIRouter(prefix="/api/v2/doctors", tags=["Doctors V2"])


# ==================== REQUEST/RESPONSE MODELS ====================
# Reuse models from V1 router to maintain contract consistency

class CreateDoctorRequest(BaseModel):
    """Request model for creating a new doctor."""
    doctor_name: str = Field(..., min_length=3, max_length=200)
    specialty: Optional[str] = Field(None, max_length=200)
    is_active: bool = Field(True)
    source_system: str = Field("MANUAL", max_length=100)


class DoctorSearchItem(BaseModel):
    """Individual doctor item in search results."""
    doctor_id: Optional[int] = Field(None, description="Doctor ID (if available)")
    employeeId: Optional[str] = Field(None, description="Employee ID")
    full_name: str = Field(..., description="Doctor full name")
    nameEn: Optional[str] = Field(None, description="English name")
    name: str = Field(..., description="Doctor name")
    specialty: Optional[str] = Field(None, description="Doctor specialty")
    department: Optional[str] = Field(None, description="Department name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "doctor_id": 12345,
                "employeeId": "E67890",
                "full_name": "Dr. Ahmed Mohammed",
                "nameEn": "Dr. Ahmed Mohammed",
                "name": "Dr. Ahmed Mohammed",
                "specialty": "Cardiology",
                "department": "Cardiac Care"
            }
        }


class DoctorSearchResponse(BaseModel):
    """Response model for doctor search endpoint."""
    success: bool = Field(True, description="Request success status")
    items: List[DoctorSearchItem] = Field(..., description="List of matching doctors")
    total: int = Field(..., description="Total number of results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "items": [
                    {
                        "doctor_id": 12345,
                        "employeeId": "E67890",
                        "full_name": "Dr. Ahmed Mohammed",
                        "nameEn": "Dr. Ahmed Mohammed",
                        "name": "Dr. Ahmed Mohammed",
                        "specialty": "Cardiology",
                        "department": "Cardiac Care"
                    }
                ],
                "total": 1
            }
        }



# ==================== ENDPOINTS ====================
# Thin wrappers that delegate to service layer (reusing business logic)

@router.post("", status_code=status.HTTP_201_CREATED, summary="Create a new doctor")
async def create_doctor(request: CreateDoctorRequest):
    """Create a new doctor in the reserve table."""
    try:
        result = DoctorService.create_doctor(
            doctor_name=request.doctor_name,
            specialty=request.specialty,
            is_active=request.is_active,
            source_system=request.source_system
        )
        return result
    except ValueError as ve:
        error_msg = str(ve)
        if "already exists" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": "DUPLICATE_DOCTOR",
                    "message": error_msg,
                    "message_ar": "الطبيب موجود بالفعل"
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


@router.get("/reserve", summary="Get all reserve doctors")
async def get_reserve_doctors(
    limit: int = Query(100, ge=1, le=500, description="Max results to return")
):
    """Get all doctors from the reserve table only."""
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


@router.get("/search", response_model=DoctorSearchResponse, summary="Search doctors by name or ID")
async def search_doctors_endpoint(
    q: str = Query(..., min_length=2, description="Search query (doctor name or ID, minimum 2 characters)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results (1-100)")
):
    """
    Search for doctors by name (English/Arabic) or ID.
    
    This endpoint searches across both hospital system and reserve tables
    to find matching doctors. Supports partial matching and is case-insensitive.
    
    **Query Parameters:**
    - `q`: Search text (required, min 2 characters) - searches name (Arabic/English) or ID
    - `limit`: Maximum results to return (default: 20, max: 100)
    
    **Search Behavior:**
    - Searches doctor name, employee ID, doctor ID
    - Case-insensitive partial matching
    - Searches both hospital view and reserve table (dual-source pattern)
    - Only returns active doctors
    
    **Response Fields:**
    - `success`: Always true for successful requests
    - `items`: Array of matching doctors
    - `total`: Number of doctors returned
    
    **Example Requests:**
    ```
    GET /api/v2/doctors/search?q=ahmed&limit=20
    GET /api/v2/doctors/search?q=أحمد&limit=10
    GET /api/v2/doctors/search?q=D123&limit=20
    ```
    
    **Example Response:**
    ```json
    {
      "success": true,
      "items": [
        {
          "doctor_id": 12345,
          "employeeId": "E67890",
          "full_name": "Dr. Ahmed Mohammed",
          "nameEn": "Dr. Ahmed Mohammed",
          "name": "Dr. Ahmed Mohammed",
          "specialty": "Cardiology",
          "department": "Cardiac Care"
        }
      ],
      "total": 1
    }
    ```
    """
    try:
        # Call existing search service (reusing logic)
        result = search_doctors_service(search_text=q, limit=limit)
        
        # Check for service-level errors
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "SEARCH_FAILED",
                    "message": result.get("error", "Failed to search doctors"),
                    "message_ar": "فشل في البحث عن الأطباء"
                }
            )
        
        # Normalize response format: "doctors" -> "items"
        doctors = result.get("doctors", [])
        
        # Map to DoctorSearchItem schema
        items = []
        for doc in doctors:
            # Map fields to expected format
            doctor_id = doc.get("doctor_id")
            name = doc.get("name", "")
            specialty_name = doc.get("speciality_name", "")
            
            items.append(DoctorSearchItem(
                doctor_id=doctor_id,
                employeeId=str(doctor_id) if doctor_id else None,  # Use doctor_id as employeeId
                full_name=name,
                nameEn=name,
                name=name,
                specialty=specialty_name if specialty_name else None,
                department=None  # Not available in current schema
            ))
        
        # Return normalized V2 response
        return DoctorSearchResponse(
            success=True,
            items=items,
            total=len(items)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SEARCH_ERROR",
                "message": f"Search operation failed: {str(e)}",
                "message_ar": "فشلت عملية البحث"
            }
        )



@router.get("", summary="Search for doctors")
async def search_doctors(
    query: Optional[str] = Query(None, description="Search by name or employee ID"),
    department: Optional[str] = Query(None, description="Filter by department"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=500, description="Max results to return")
):
    """Search for doctors by name, employee ID, or department."""
    try:
        result = DoctorService.search_doctors(
            query=query,
            department=department,
            status=status,
            limit=limit
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SEARCH_FAILED",
                "message": str(e),
                "message_ar": f"فشل البحث: {str(e)}"
            }
        )


@router.get("/{doctor_id}/profile", response_model=DoctorProfileV2Response, summary="Get doctor profile")
async def get_doctor_profile(
    doctor_id: int = Path(..., gt=0, description="Doctor ID")
):
    """
    Fetch detailed profile information for a specific doctor.
    
    Phase B — V2 profile contract normalized.
    """
    try:
        # Get profile from service layer (reusing existing logic)
        result = DoctorService.get_doctor_profile(doctor_id)
        
        # Normalize to V2 contract: profile, metrics, items, meta
        return DoctorProfileV2Response(
            profile=result,  # Doctor identity/basic info
            metrics={},       # No metrics in simple profile endpoint
            items=[],         # No items in simple profile endpoint
            meta=EntityMeta(
                entity_type="doctor",
                entity_id=doctor_id,
                period_from=None,
                period_to=None
            )
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "DOCTOR_NOT_FOUND",
                "message": str(e),
                "message_ar": "الدكتور غير موجود"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PROFILE_FETCH_FAILED",
                "message": str(e),
                "message_ar": f"فشل جلب الملف الشخصي: {str(e)}"
            }
        )


@router.get("/{doctor_id}/statistics", summary="Get doctor statistics")
async def get_doctor_statistics(
    doctor_id: int = Path(..., gt=0, description="Doctor ID"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Fetch aggregated incident statistics for a doctor."""
    try:
        result = DoctorService.get_doctor_statistics(
            doctor_id=doctor_id,
            from_date=from_date,
            to_date=to_date
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=404 if "not found" in str(e).lower() else 400,
            detail={
                "error": "INVALID_REQUEST",
                "message": str(e),
                "message_ar": str(e)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "STATISTICS_FETCH_FAILED",
                "message": str(e),
                "message_ar": f"فشل جلب الإحصائيات: {str(e)}"
            }
        )


@router.get("/{doctor_id}/analytics", summary="Get doctor analytics")
async def get_doctor_analytics(
    doctor_id: int = Path(..., gt=0, description="Doctor's ID"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get doctor analytics: category breakdown and monthly trends."""
    try:
        result = DoctorService.get_doctor_analytics(
            doctor_id=doctor_id,
            from_date=from_date,
            to_date=to_date
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=404 if "not found" in str(e).lower() else 400,
            detail={
                "error": "INVALID_REQUEST",
                "message": str(e),
                "message_ar": str(e)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ANALYTICS_FETCH_FAILED",
                "message": str(e),
                "message_ar": f"فشل جلب التحليلات: {str(e)}"
            }
        )


@router.get("/{doctor_id}/incidents", summary="Get doctor incidents")
async def get_doctor_incidents(
    doctor_id: int = Path(..., gt=0, description="Doctor's ID"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    status: Optional[str] = Query(None, description="Filter by status"),
    red_flags_only: bool = Query(False, description="Only incidents with red flags"),
    limit: int = Query(100, ge=1, le=500, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """Get doctor's incidents with filtering and pagination."""
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
        raise HTTPException(
            status_code=404 if "not found" in str(e).lower() else 400,
            detail={
                "error": "INVALID_REQUEST",
                "message": str(e),
                "message_ar": str(e)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INCIDENTS_FETCH_FAILED",
                "message": str(e),
                "message_ar": f"فشل جلب الحوادث: {str(e)}"
            }
        )


@router.get("/{doctor_id}/full-report", summary="Get comprehensive doctor report")
async def get_doctor_full_report(
    doctor_id: int = Path(..., gt=0, description="Doctor's ID"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    status: Optional[str] = Query(None, description="Filter by status"),
    red_flags_only: bool = Query(False, description="Only incidents with red flags"),
    limit: int = Query(100, ge=1, le=500, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """Get comprehensive doctor report with all data in one call."""
    import logging
    logger = logging.getLogger("doctor_full_report_endpoint")
    try:
        logger.info(f"Fetching full report for doctor_id={doctor_id}")
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
        logger.info(f"Full report for doctor_id={doctor_id} generated successfully")
        return result
    except ValueError as e:
        logger.warning(f"ValueError for doctor_id={doctor_id}: {e}")
        raise HTTPException(
            status_code=404 if "not found" in str(e).lower() else 400,
            detail={
                "error": "INVALID_REQUEST",
                "message": str(e),
                "message_ar": str(e)
            }
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Full report failed for doctor_id={doctor_id}: {e}\n{tb}")
        print(f"[DOCTOR FULL REPORT ERROR] doctor_id={doctor_id}: {e}\n{tb}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "FULL_REPORT_GENERATION_FAILED",
                "message": str(e),
                "message_ar": f"فشل إنشاء التقرير الكامل: {str(e)}",
                "traceback": tb
            }
        )


@router.get("/{doctor_id}/full-history", summary="Get doctor full history")
async def get_doctor_full_history(
    doctor_id: int = Path(..., gt=0, description="Doctor's ID"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    status: Optional[str] = Query(None, description="Filter by status"),
    red_flags_only: bool = Query(False, description="Only incidents with red flags"),
    limit: int = Query(100, ge=1, le=500, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """
    Get doctor profile and incidents in a single request.
    Returns unified structure matching patient full-history endpoint.
    """
    try:
        result = get_doctor_full_history_service(
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
        raise HTTPException(
            status_code=404 if "not found" in str(e).lower() else 400,
            detail={
                "error": "INVALID_REQUEST",
                "message": str(e),
                "message_ar": str(e)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "FULL_HISTORY_FAILED",
                "message": str(e),
                "message_ar": f"فشل جلب التاريخ الكامل: {str(e)}"
            }
        )


@router.get("/{doctor_id}/export", summary="Export doctor history")
async def export_doctor_history(
    doctor_id: int = Path(..., gt=0, description="Doctor's ID"),
    format: str = Query("json", description="Export format: 'csv', 'json', or 'word'"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    include_profile: bool = Query(True, description="Include doctor profile in export")
):
    """Export doctor history in CSV, JSON, or Word format."""
    try:
        # Validate format
        if format.lower() not in ["csv", "json", "word"]:
            raise HTTPException(
                status_code=400,
                detail="Format must be 'csv', 'json', or 'word'"
            )
        
        # Get export data
        export_data = export_doctor_history_service(
            doctor_id=doctor_id,
            format_type=format.lower(),
            from_date=from_date,
            to_date=to_date,
            include_profile=include_profile
        )
        
        # Return based on format
        if format.lower() == "csv":
            csv_content = export_data['content']
            filename = export_data['filename']
            return StreamingResponse(
                iter([csv_content]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        elif format.lower() == "word":
            word_bytes = export_data['content']
            filename = export_data['filename']
            return Response(
                content=word_bytes,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        else:
            return export_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "EXPORT_FAILED",
                "message": str(e),
                "message_ar": f"فشل تصدير البيانات: {str(e)}"
            }
        )


@router.get("/health-check/check", summary="Health check")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Doctors API V2 is running"
    }
