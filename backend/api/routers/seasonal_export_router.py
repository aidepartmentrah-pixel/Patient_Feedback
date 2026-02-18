"""
D-B9: Seasonal Export Endpoints
D-B10: Permission Guards for Seasonal Exports

REST API endpoints for exporting doctor and worker seasonal reports as Word documents.
Orchestrates data aggregation (D-B6/D-B7) and Word generation (D-B8).
Applies role-based authorization guards (D-B10).

Endpoints:
- GET /api/doctors/{doctor_id}/seasonal-report (requires admin/supervisor role)
- GET /api/workers/{employee_id}/seasonal-report (requires admin/supervisor role)
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import Response
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND
from typing import Optional
from datetime import datetime

from ..services.doctor_seasonal_reporting_service import DoctorSeasonalReportingService
from ..services.worker_seasonal_reporting_service import WorkerSeasonalReportingService
from ..services.seasonal_word_adapter import SeasonalWordAdapter
from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_doctor_report_access, require_worker_report_access


router = APIRouter(prefix="/api", tags=["seasonal_exports"])


@router.get("/doctors/{doctor_id}/seasonal-report")
def export_doctor_seasonal_report(
    doctor_id: int,
    season_start: str = Query(..., description="Season start date (YYYY-MM-DD)"),
    season_end: str = Query(..., description="Season end date (YYYY-MM-DD)"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Export doctor seasonal performance report as Word document.
    
    **Authorization:** Requires admin or supervisor role (SOFTWARE_ADMIN, ADMINISTRATION_ADMIN,
    DEPARTMENT_ADMIN, SECTION_ADMIN, or COMPLAINT_SUPERVISOR).
    
    Generates a comprehensive Word report including:
    - Doctor identity information
    - Performance score and praise/risk levels
    - Metrics (incidents, actions, explanations)
    - Category breakdown and monthly trends
    - High severity incidents summary
    
    Args:
        doctor_id: Doctor's employee ID
        season_start: Start date of reporting period (YYYY-MM-DD)
        season_end: End date of reporting period (YYYY-MM-DD)
        current_user: Authenticated user (injected by auth dependency)
    
    Returns:
        Word document (.docx) with proper content-disposition header
    
    Raises:
        401: Not authenticated
        403: Insufficient permissions (not admin/supervisor)
        400: Invalid date format or date range
        404: Doctor not found
        500: Report generation failed
    """
    # D-B10: Apply permission guard
    require_doctor_report_access(current_user)
    
    try:
        # Validate date format (will raise ValueError if invalid)
        try:
            datetime.strptime(season_start, '%Y-%m-%d')
            datetime.strptime(season_end, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD"
            )
        
        # Build report data
        try:
            report_data = DoctorSeasonalReportingService.build_doctor_seasonal_report_data(
                doctor_id=doctor_id,
                season_start=season_start,
                season_end=season_end
            )
        except ValueError as e:
            # Doctor not found or validation error
            if 'not found' in str(e).lower():
                raise HTTPException(
                    status_code=HTTP_404_NOT_FOUND,
                    detail=f"Doctor {doctor_id} not found"
                )
            else:
                raise HTTPException(
                    status_code=HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
        
        # Generate Word document
        word_bytes = SeasonalWordAdapter.generate_doctor_seasonal_word(report_data)
        
        # Prepare filename
        season_label = report_data['period'].get('season_name', f"{season_start}_to_{season_end}")
        season_label_safe = str(season_label).replace(' ', '_')
        filename = f"Doctor_{doctor_id}_Seasonal_Report_{season_label_safe}.docx"
        
        # Return as downloadable file
        return Response(
            content=word_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}"
        )


@router.get("/workers/{employee_id}/seasonal-report")
def export_worker_seasonal_report(
    employee_id: int,
    season_start: str = Query(..., description="Season start date (YYYY-MM-DD)"),
    season_end: str = Query(..., description="Season end date (YYYY-MM-DD)"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Export worker seasonal performance report as Word document.
    
    **Authorization:** Requires admin or supervisor role (SOFTWARE_ADMIN, ADMINISTRATION_ADMIN,
    DEPARTMENT_ADMIN, SECTION_ADMIN, or COMPLAINT_SUPERVISOR).
    
    Generates a comprehensive Word report including:
    - Worker identity information
    - Performance score and praise/risk levels
    - Metrics (incidents, action items, explanations)
    - Action completion rate
    
    Args:
        employee_id: Worker's employee ID
        season_start: Start date of reporting period (YYYY-MM-DD)
        season_end: End date of reporting period (YYYY-MM-DD)
        current_user: Authenticated user (injected by auth dependency)
    
    Returns:
        Word document (.docx) with proper content-disposition header
    
    Raises:
        401: Not authenticated
        403: Insufficient permissions (not admin/supervisor)
        400: Invalid date format or date range
        404: Worker not found
        500: Report generation failed
    """
    # D-B10: Apply permission guard
    require_worker_report_access(current_user)
    
    try:
        # Validate date format (will raise ValueError if invalid)
        try:
            datetime.strptime(season_start, '%Y-%m-%d')
            datetime.strptime(season_end, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD"
            )
        
        # Build report data
        try:
            report_data = WorkerSeasonalReportingService.build_worker_seasonal_report_data(
                employee_id=employee_id,
                season_start=season_start,
                season_end=season_end
            )
        except ValueError as e:
            # Worker not found or validation error
            if 'not found' in str(e).lower():
                raise HTTPException(
                    status_code=HTTP_404_NOT_FOUND,
                    detail=f"Worker {employee_id} not found"
                )
            else:
                raise HTTPException(
                    status_code=HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
        
        # Generate Word document
        word_bytes = SeasonalWordAdapter.generate_worker_seasonal_word(report_data)
        
        # Prepare filename
        season_label = report_data['period'].get('season_name', f"{season_start}_to_{season_end}")
        season_label_safe = str(season_label).replace(' ', '_')
        filename = f"Worker_{employee_id}_Seasonal_Report_{season_label_safe}.docx"
        
        # Return as downloadable file
        return Response(
            content=word_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}"
        )
