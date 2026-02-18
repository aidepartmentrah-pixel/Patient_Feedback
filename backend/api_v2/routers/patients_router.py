"""
Patient Router (API v2)
Phase B — B-B2 — Patient router exposed under API v2 namespace.

This router exposes all existing patient endpoints under /api/v2/patients
without rewriting business logic. Thin wrapper that delegates to service layer.

Endpoints exposed:
- POST /api/v2/patients - Create patient
- GET /api/v2/patients/reserve - Get reserve patients
- GET /api/v2/patients/search - Search patients
- GET /api/v2/patients/{patient_id}/profile - Patient profile
- GET /api/v2/patients/{patient_id}/incidents - Patient incidents
- GET /api/v2/patients/{patient_id}/incidents/{incident_id} - Incident details
- GET /api/v2/patients/{patient_id}/full-history - Full history
- GET /api/v2/patients/{patient_id}/export - Export patient history

Security: All endpoints protected by authentication (where applicable).
"""

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response
from typing import Optional
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field
import io

from backend.api.services.patients_service import (
    search_patients_service,
    get_patient_profile_service,
    get_patient_incidents_service,
    get_incident_details_service,
    get_patient_full_history_service,
    export_patient_history_service,
    create_patient_service,
    get_all_reserve_patients_service
)
from backend.api_v2.schemas.profile_schemas import PatientProfileV2Response, EntityMeta


# ============================================================
# ROUTER DEFINITION (Phase B — B-B2)
# ============================================================
router = APIRouter(prefix="/api/v2/patients", tags=["Patients V2"])


# ==================== REQUEST/RESPONSE MODELS ====================
# Reuse models from V1 router to maintain contract consistency

class CreatePatientRequest(BaseModel):
    """Request model for creating a new patient."""
    first_name: str = Field(..., min_length=2, max_length=150)
    middle_name: Optional[str] = Field(None, max_length=150)
    last_name: Optional[str] = Field(None, max_length=150)
    mother_name: Optional[str] = Field(None, max_length=150)
    phone_number: Optional[str] = Field(None, max_length=50)
    phone_number2: Optional[str] = Field(None, max_length=50)
    birth_date: Optional[str] = Field(None)
    sex: Optional[str] = Field(None)
    document_number: Optional[str] = Field(None, max_length=100)
    medical_file_number: Optional[str] = Field(None, max_length=100)
    spouse: Optional[str] = Field(None, max_length=150)
    address_line1: Optional[str] = Field(None, max_length=300)
    address_line2: Optional[str] = Field(None, max_length=300)


# ==================== ENDPOINTS ====================
# Thin wrappers that delegate to service layer (reusing business logic)

@router.post("", status_code=status.HTTP_201_CREATED, summary="Create a new patient")
async def create_patient(request: CreatePatientRequest):
    """Create a new patient in the reserve table."""
    try:
        result = create_patient_service(
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
        return result
    except ValueError as ve:
        error_msg = str(ve)
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Failed to create patient: {str(e)}",
                "message_ar": "فشل في إنشاء المريض"
            }
        )


@router.get("/reserve", summary="Get all reserve patients")
async def get_reserve_patients(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order_by: str = Query("PatientAdmissionID DESC")
):
    """Get all patients from the reserve table only."""
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


@router.get("/search", summary="Search for patients")
async def search_patients(
    q: Optional[str] = Query(None, alias="q", description="Search by patient name (also accepts 'query')"),
    query: Optional[str] = Query(None, description="Search by patient name (legacy param name)"),
    mrn: Optional[str] = Query(None, description="Search by MRN"),
    phone: Optional[str] = Query(None, description="Search by phone"),
    date_of_birth: Optional[str] = Query(None, description="Filter by date of birth"),
    limit: int = Query(50, ge=1, le=100)
):
    """Search for patients by name, MRN, phone, or date of birth.
    
    Accepts search text via either `q` or `query` parameter for compatibility.
    """
    # Support both ?q=xxx (frontend) and ?query=xxx (legacy) parameter names
    search_text = q or query
    try:
        result = search_patients_service(
            query=search_text,
            mrn=mrn,
            phone=phone,
            date_of_birth=date_of_birth,
            limit=limit
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/{patient_id}/profile", response_model=PatientProfileV2Response, summary="Get patient profile")
async def get_patient_profile(patient_id: int):
    """
    Get complete patient profile information.
    
    Phase B — V2 profile contract normalized.
    """
    try:
        # Get profile from service layer (reusing existing logic)
        profile_data = get_patient_profile_service(patient_id)
        if not profile_data:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {patient_id} not found"
            )
        
        # Extract metrics from profile data
        metrics = {}
        if "TotalIncidents" in profile_data:
            metrics["TotalIncidents"] = profile_data["TotalIncidents"]
        if "LastVisitDate" in profile_data:
            metrics["LastVisitDate"] = profile_data["LastVisitDate"]
        
        # Normalize to V2 contract: profile, metrics, items, meta
        return PatientProfileV2Response(
            profile=profile_data,  # Full patient demographics
            metrics=metrics,        # Extracted metrics
            items=[],               # No items in simple profile endpoint
            meta=EntityMeta(
                entity_type="patient",
                entity_id=patient_id,
                period_from=None,
                period_to=None
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get profile: {str(e)}"
        )


@router.get("/{patient_id}/incidents", summary="Get patient incidents")
async def get_patient_incidents(
    patient_id: int,
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get all feedback/incident records for a patient."""
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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get incidents: {str(e)}"
        )


@router.get("/{patient_id}/incidents/{incident_id}", summary="Get incident details")
async def get_incident_details(patient_id: int, incident_id: int):
    """Get full details for a specific incident."""
    try:
        incident = get_incident_details_service(patient_id, incident_id)
        if not incident:
            raise HTTPException(
                status_code=404,
                detail=f"Incident {incident_id} not found"
            )
        return incident
    except Exception as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get incident details: {str(e)}"
        )


@router.get("/{patient_id}/full-history", summary="Get patient full history")
async def get_patient_full_history(
    patient_id: int,
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get patient profile and incidents in a single request."""
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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get full history: {str(e)}"
        )


@router.get("/{patient_id}/export", summary="Export patient history")
async def export_patient_history(
    patient_id: int,
    format: str = Query("json", description="Export format: 'csv', 'json', or 'word'"),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    include_profile: bool = Query(True)
):
    """Export patient history in CSV, JSON, or Word format."""
    try:
        # Validate format
        if format.lower() not in ["csv", "json", "word"]:
            raise HTTPException(
                status_code=400,
                detail="Format must be 'csv', 'json', or 'word'"
            )
        
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
        elif format.lower() == "word":
            # Return Word file as direct response
            word_bytes = export_data['content']
            filename = export_data['filename']
            
            return Response(
                content=word_bytes,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        else:
            # Return JSON
            return export_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {str(e)}"
        )
