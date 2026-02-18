"""
D-B9: Seasonal Person Report Endpoint
D-B10: Export Contract Alignment
D-B11: Permission Guard Integration

Router endpoints for exporting doctor and worker seasonal reports as Word documents.
Orchestrates builder services and Word adapter to generate downloadable reports.

Endpoints:
- GET /api/person-reports/doctor/{doctor_id}/seasonal-word
- GET /api/person-reports/worker/{employee_id}/seasonal-word

Architecture:
- Router layer (this file) → Builder services → Word adapter
- No DB logic in router
- No scoring logic in router
- Uses StreamingResponse pattern for file downloads

Export contract aligned with reports_router pattern.
Scope + role guard applied per reporting security policy.
"""

from io import BytesIO
from datetime import date, datetime
from fastapi import APIRouter, Path, Query, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any

from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..services.doctor_seasonal_reporting_service import DoctorSeasonalReportingService
from ..services.worker_seasonal_reporting_service import WorkerSeasonalReportingService
from ..services.person_report_word_adapter import generate_person_seasonal_word_report
from ..utils.guards import (
    require_doctor_report_access,
    require_worker_report_access,
    require_unit_in_scope,
    require_role
)
from ..db_layer.worker_reporting_db import get_worker_identity
from core.database import get_connection
from core.constants.roles import SOFTWARE_ADMIN, WORKER, COMPLAINT_SUPERVISOR


router = APIRouter(prefix="/api/person-reports", tags=["Person Reports"])


def _build_person_report_filename(person_type: str, person_id: int) -> str:
    """
    Build standardized filename for person seasonal reports.
    
    Export contract aligned with reports_router pattern.
    
    Args:
        person_type: "doctor" or "worker"
        person_id: ID of the person
    
    Returns:
        Filename in format: doctor_seasonal_{id}.docx or worker_seasonal_{id}.docx
    """
    return f"{person_type}_seasonal_{person_id}.docx"


@router.get("/doctor/{doctor_id}/seasonal-word")
async def export_doctor_seasonal_word(
    doctor_id: int = Path(..., gt=0, description="Doctor's employee ID"),
    season_start: date = Query(..., description="Season start date (YYYY-MM-DD)"),
    season_end: date = Query(..., description="Season end date (YYYY-MM-DD)"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Export doctor seasonal performance report as Word document.
    
    Downloads a professionally formatted Word document containing:
    - Doctor identity and specialty
    - Seasonal performance metrics
    - Incident statistics
    - Top incident categories
    
    **Authentication:** Required (uses session-based auth)
    
    **Query Parameters:**
    - season_start: Start date of reporting period (YYYY-MM-DD format)
    - season_end: End date of reporting period (YYYY-MM-DD format)
    
    **Returns:**
    - Word document (.docx) with Content-Disposition header for download
    - Filename: doctor_{id}_seasonal.docx
    
    **Errors:**
    - 401: Not authenticated
    - 403: Forbidden (insufficient role or out of scope)
    - 404: Doctor not found
    - 400: Invalid date format or date range
    - 500: Report generation failed
    """
    try:
        # Scope + role guard applied per reporting security policy
        
        # Step 1: Role guard - verify user has permission to access doctor reports
        require_doctor_report_access(current_user)
        
        # Step 2: Scope validation - resolve doctor's org unit and verify access
        employee_identity = get_worker_identity(doctor_id)
        if not employee_identity:
            raise HTTPException(
                status_code=404,
                detail=f"Doctor {doctor_id} not found in employee records"
            )
        
        # Use section_id as primary org unit for scope validation
        if employee_identity.get('section_id'):
            require_unit_in_scope(current_user, employee_identity['section_id'])
        
        # Step 3: Build doctor seasonal report data
        payload = DoctorSeasonalReportingService.build_doctor_seasonal_report_data(
            doctor_id=doctor_id,
            season_start=season_start,
            season_end=season_end
        )
        
        # Step 4: Generate Word document bytes using adapter
        word_bytes = generate_person_seasonal_word_report(
            person_type="doctor",
            payload=payload
        )
        
        # Step 5: Wrap bytes in BytesIO for StreamingResponse
        word_file = BytesIO(word_bytes)
        word_file.seek(0)
        
        # Step 6: Prepare filename (using standardized helper)
        filename = _build_person_report_filename("doctor", doctor_id)
        
        # Step 7: Return as StreamingResponse (export contract aligned with reports_router pattern)
        return StreamingResponse(
            word_file,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    
    except ValueError as e:
        # Doctor not found or validation error
        if 'not found' in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate doctor seasonal report: {str(e)}"
        )


@router.get("/worker/{employee_id}/seasonal-word")
async def export_worker_seasonal_word(
    employee_id: int = Path(..., gt=0, description="Worker's employee ID"),
    season_start: date = Query(..., description="Season start date (YYYY-MM-DD)"),
    season_end: date = Query(..., description="Season end date (YYYY-MM-DD)"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Export worker seasonal performance report as Word document.
    
    Downloads a professionally formatted Word document containing:
    - Worker identity and job title
    - Seasonal performance metrics
    - Action item statistics
    - Explanation workflow metrics
    
    **Authentication:** Required (uses session-based auth)
    
    **Query Parameters:**
    - season_start: Start date of reporting period (YYYY-MM-DD format)
    - season_end: End date of reporting period (YYYY-MM-DD format)
    
    **Returns:**
    - Word document (.docx) with Content-Disposition header for download
    - Filename: worker_{id}_seasonal.docx
    
    **Errors:**
    - 401: Not authenticated
    - 403: Forbidden (insufficient role or out of scope)
    - 404: Worker not found
    - 400: Invalid date format or date range
    - 500: Report generation failed
    """
    try:
        # Scope + role guard applied per reporting security policy
        
        # Step 1: Role guard - verify user has permission to access worker reports
        require_worker_report_access(current_user)
        
        # Step 2: Scope validation - resolve worker's org unit and verify access
        employee_identity = get_worker_identity(employee_id)
        if not employee_identity:
            raise HTTPException(
                status_code=404,
                detail=f"Worker {employee_id} not found in employee records"
            )
        
        # Use section_id as primary org unit for scope validation
        if employee_identity.get('section_id'):
            require_unit_in_scope(current_user, employee_identity['section_id'])
        
        # Step 3: Build worker seasonal report data
        payload = WorkerSeasonalReportingService.build_worker_seasonal_report_data(
            employee_id=employee_id,
            season_start=season_start,
            season_end=season_end
        )
        
        # Step 4: Generate Word document bytes using adapter
        word_bytes = generate_person_seasonal_word_report(
            person_type="worker",
            payload=payload
        )
        
        # Step 5: Wrap bytes in BytesIO for StreamingResponse
        word_file = BytesIO(word_bytes)
        word_file.seek(0)
        
        # Step 6: Prepare filename (using standardized helper)
        filename = _build_person_report_filename("worker", employee_id)
        
        # Step 7: Return as StreamingResponse (export contract aligned with reports_router pattern)
        return StreamingResponse(
            word_file,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    
    except ValueError as e:
        # Worker not found or validation error
        if 'not found' in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate worker seasonal report: {str(e)}"
        )


# ==================== AGGREGATE REPORT ENDPOINTS ====================

@router.get("/doctors/all-seasonal-word")
async def export_all_doctors_seasonal_word(
    season_start: date = Query(..., description="Season start date (YYYY-MM-DD)"),
    season_end: date = Query(..., description="Season end date (YYYY-MM-DD)"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Export seasonal performance report for ALL active doctors as Word document.
    
    This endpoint generates a comprehensive report covering all doctors
    in the system for the specified seasonal period.
    
    **Authorization:** Only SOFTWARE_ADMIN, WORKER, or COMPLAINT_SUPERVISOR
    
    **Document Contents:**
    - Title page with season summary
    - Executive summary with totals across all doctors
    - Individual sections for each doctor with metrics and performance
    
    **Query Parameters:**
    - season_start: Start date of reporting period (YYYY-MM-DD format)
    - season_end: End date of reporting period (YYYY-MM-DD format)
    
    **Returns:**
    - Word document (.docx) with Content-Disposition header for download
    - Filename: doctors_seasonal_report_YYYY-MM-DD_to_YYYY-MM-DD.docx
    
    **Errors:**
    - 401: Not authenticated
    - 403: Forbidden (insufficient role)
    - 404: No doctors found in system
    - 400: Invalid date range
    - 500: Report generation failed
    """
    try:
        # Authorization check: Only SOFTWARE_ADMIN, WORKER, or COMPLAINT_SUPERVISOR
        require_role(current_user, [SOFTWARE_ADMIN, WORKER, COMPLAINT_SUPERVISOR])
        
        # Validate dates
        if season_start >= season_end:
            raise HTTPException(
                status_code=400,
                detail="season_start must be before season_end"
            )
        
        # Import the aggregate report generator
        from ..services.aggregate_seasonal_report_service import generate_all_doctors_seasonal_word
        
        # Generate Word document
        word_bytes = generate_all_doctors_seasonal_word(
            season_start=season_start,
            season_end=season_end
        )
        
        # Wrap bytes in BytesIO for StreamingResponse
        word_file = BytesIO(word_bytes)
        word_file.seek(0)
        
        # Prepare filename
        filename = f"doctors_seasonal_report_{season_start.strftime('%Y-%m-%d')}_to_{season_end.strftime('%Y-%m-%d')}.docx"
        
        # Return as StreamingResponse
        return StreamingResponse(
            word_file,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    except ValueError as e:
        # No doctors found or validation error
        if 'not found' in str(e).lower() or 'no active doctors' in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate aggregate doctor seasonal report: {str(e)}"
        )


@router.get("/workers/all-seasonal-word")
async def export_all_workers_seasonal_word(
    season_start: date = Query(..., description="Season start date (YYYY-MM-DD)"),
    season_end: date = Query(..., description="Season end date (YYYY-MM-DD)"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Export seasonal performance report for ALL active workers as Word document.
    
    This endpoint generates a comprehensive report covering all workers
    in the complaint department for the specified seasonal period.
    
    **Authorization:** Only SOFTWARE_ADMIN, WORKER, or COMPLAINT_SUPERVISOR
    
    **Document Contents:**
    - Title page with season summary
    - Executive summary with totals across all workers
    - Individual sections for each worker with metrics and action items
    
    **Query Parameters:**
    - season_start: Start date of reporting period (YYYY-MM-DD format)
    - season_end: End date of reporting period (YYYY-MM-DD format)
    
    **Returns:**
    - Word document (.docx) with Content-Disposition header for download
    - Filename: workers_seasonal_report_YYYY-MM-DD_to_YYYY-MM-DD.docx
    
    **Errors:**
    - 401: Not authenticated
    - 403: Forbidden (insufficient role)
    - 404: No workers found in system
    - 400: Invalid date range
    - 500: Report generation failed
    """
    try:
        # Authorization check: Only SOFTWARE_ADMIN, WORKER, or COMPLAINT_SUPERVISOR
        require_role(current_user, [SOFTWARE_ADMIN, WORKER, COMPLAINT_SUPERVISOR])
        
        # Validate dates
        if season_start >= season_end:
            raise HTTPException(
                status_code=400,
                detail="season_start must be before season_end"
            )
        
        # Import the aggregate report generator
        from ..services.aggregate_seasonal_report_service import generate_all_workers_seasonal_word
        
        # Generate Word document
        word_bytes = generate_all_workers_seasonal_word(
            season_start=season_start,
            season_end=season_end
        )
        
        # Wrap bytes in BytesIO for StreamingResponse
        word_file = BytesIO(word_bytes)
        word_file.seek(0)
        
        # Prepare filename
        filename = f"workers_seasonal_report_{season_start.strftime('%Y-%m-%d')}_to_{season_end.strftime('%Y-%m-%d')}.docx"
        
        # Return as StreamingResponse
        return StreamingResponse(
            word_file,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    except ValueError as e:
        # No workers found or validation error
        if 'not found' in str(e).lower() or 'no active workers' in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate aggregate worker seasonal report: {str(e)}"
        )
