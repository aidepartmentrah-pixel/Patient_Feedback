"""
Table View Router
FastAPI endpoints for the TableView page (main operational data table).
"""

from datetime import datetime
from fastapi import APIRouter, Query, HTTPException, Body, Path, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel

from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..services.table_view_service import (
    get_complaints_paginated,
    get_filter_options,
    get_complaint_by_id,
    get_complaints_count,
    export_complaints,
    export_complaints_excel,
    get_table_views,
)
from ..db_layer.incident_case import soft_delete_incident_case, hard_delete_incident_case


router = APIRouter(prefix="/api/complaints", tags=["Table View"])

# Roles that see all records with no org-unit restriction
_FULL_ACCESS_ROLES = {"SOFTWARE_ADMIN", "COMPLAINT_SUPERVISOR", "WORKER"}


# ==================== REQUEST/RESPONSE MODELS ====================

class ExportRequest(BaseModel):
    """Request model for complaint export."""
    format: Literal['csv', 'json']
    filters: Dict[str, Any] = {}
    columns: List[str]
    include_patient_identifiers: bool = False
    language: str = 'en'


# ==================== ENDPOINTS ====================

@router.get("")
async def get_complaints(
    search: Optional[str] = Query(None, description="Free-text search across complaint_number, patient_name, complaint_text"),
    issuing_org_unit_id: Optional[int] = Query(None, description="Filter by issuing organizational unit ID"),
    target_department_id: Optional[int] = Query(None, description="Filter by target section ID"),
    target_dept_parent_id: Optional[int] = Query(None, description="Filter by target department (Type=2) ID"),
    target_admin_id: Optional[int] = Query(None, description="Filter by target administration (Type=1) ID"),
    domain_id: Optional[str] = Query(None, description="Filter by HCAT domain ID(s), comma-separated for multi-select"),
    category_id: Optional[str] = Query(None, description="Filter by category ID(s), comma-separated for multi-select"),
    subcategory_id: Optional[str] = Query(None, description="Filter by subcategory ID(s), comma-separated for multi-select"),
    classification_id: Optional[str] = Query(None, description="Filter by classification ID(s), comma-separated for multi-select"),
    severity_id: Optional[str] = Query(None, description="Filter by severity level ID(s), comma-separated for multi-select"),
    stage_id: Optional[str] = Query(None, description="Filter by stage ID(s), comma-separated for multi-select"),
    harm_level_id: Optional[str] = Query(None, description="Filter by harm level ID(s), comma-separated for multi-select"),
    case_status_id: Optional[str] = Query(None, description="Filter by case status ID(s), comma-separated for multi-select"),
    clinical_risk_type_id: Optional[str] = Query(None, description="Filter by clinical risk type ID(s), comma-separated for multi-select"),
    feedback_intent_type_id: Optional[str] = Query(None, description="Filter by feedback intent type ID(s), comma-separated for multi-select"),
    source_id: Optional[str] = Query(None, description="Filter by source ID(s), comma-separated for multi-select"),
    year: Optional[int] = Query(None, description="Filter by year (YYYY)"),
    month: Optional[int] = Query(None, ge=1, le=12, description="Filter by month (1-12)"),
    start_date: Optional[str] = Query(None, description="Filter by received_date >= start_date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filter by received_date <= end_date (YYYY-MM-DD)"),
    sort_by: str = Query("FeedbackRecievedDate", description="Sort field"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order (asc or desc)"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=500, description="Results per page (1-500)"),
    view: str = Query("complete", description="View preset (complete, simplified)"),
    tab: Optional[str] = Query(None, description="Tab filter: 'preparation' | 'workflow'"),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Fetch paginated complaints with search and filtering.
    
    This is the primary endpoint for the TableView page.
    Supports extensive filtering, searching, sorting, and pagination.
    
    **Example Request:**
    ```
    GET /api/complaints?status=open&issuing_dept_id=12&page=1&page_size=50
    ```
    
    **Returns:**
    - `complaints`: Array of complaint records (masked patient data)
    - `pagination`: Pagination metadata (page, page_size, total_records, total_pages)
    - `filters_applied`: Echo of applied filters
    - `view`: Current view preset
    """
    try:
        # Org-scoped roles may only see their own unit subtree
        user_roles = set(current_user.roles)
        is_full_access = bool(user_roles & _FULL_ACCESS_ROLES)
        restrict_to_unit_ids = None if is_full_access else current_user.allowed_unit_ids

        result = get_complaints_paginated(
            search=search,
            issuing_org_unit_id=issuing_org_unit_id,
            target_department_id=target_department_id,
            target_dept_parent_id=target_dept_parent_id,
            target_admin_id=target_admin_id,
            domain_id=domain_id,
            category_id=category_id,
            subcategory_id=subcategory_id,
            classification_id=classification_id,
            severity_id=severity_id,
            stage_id=stage_id,
            harm_level_id=harm_level_id,
            case_status_id=case_status_id,
            clinical_risk_type_id=clinical_risk_type_id,
            feedback_intent_type_id=feedback_intent_type_id,
            source_id=source_id,
            year=year,
            month=month,
            start_date=start_date,
            end_date=end_date,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            page_size=page_size,
            view=view,
            tab=tab,
            restrict_to_unit_ids=restrict_to_unit_ids,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": "invalid_parameters",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "internal_server_error",
            "message": f"An error occurred while fetching complaints: {str(e)}",
            "message_ar": f"حدث خطأ أثناء جلب الشكاوى: {str(e)}"
        })


@router.get("/filter-options")
async def get_filter_dropdown_options(
    include_counts: bool = Query(False, description="Include record counts for each option")
):
    """
    Fetch available filter options for dropdown population.
    
    Returns reference data for:
    - Issuing departments
    - Target departments
    - Sources (patient, staff, family, external)
    - Statuses (open, in_progress, closed)
    - Severities (Low, Medium, High)
    - Domains (HCAT domains)
    - Categories (HCAT categories)
    
    **Example Request:**
    ```
    GET /api/complaints/filter-options?include_counts=true
    ```
    
    **Returns:**
    Dictionary with arrays for each filter type.
    If `include_counts=true`, each option includes `count` field.
    """
    try:
        result = get_filter_options(include_counts=include_counts)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "internal_server_error",
            "message": f"An error occurred while fetching filter options: {str(e)}",
            "message_ar": f"حدث خطأ أثناء جلب خيارات التصفية: {str(e)}"
        })


@router.get("/count")
async def get_filtered_count(
    search: Optional[str] = Query(None),
    issuing_org_unit_id: Optional[int] = Query(None),
    domain_id: Optional[str] = Query(None),
    category_id: Optional[str] = Query(None),
    subcategory_id: Optional[str] = Query(None),
    classification_id: Optional[str] = Query(None),
    severity_id: Optional[str] = Query(None),
    stage_id: Optional[str] = Query(None),
    harm_level_id: Optional[str] = Query(None),
    case_status_id: Optional[str] = Query(None),
    clinical_risk_type_id: Optional[str] = Query(None),
    feedback_intent_type_id: Optional[str] = Query(None),
    source_id: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    month: Optional[int] = Query(None, ge=1, le=12),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """
    Get total count of complaints matching current filters.
    
    Used for export preview ("X records found") and summary statistics.
    Accepts same filter parameters as GET /api/complaints, but excludes pagination.
    
    **Example Request:**
    ```
    GET /api/complaints/count?status=open&issuing_dept_id=12
    ```
    
    **Returns:**
    - `total_count`: Total number of matching records
    - `filters_applied`: Echo of applied filters
    """
    try:
        result = get_complaints_count(
            search=search,
            issuing_org_unit_id=issuing_org_unit_id,
            domain_id=domain_id,
            category_id=category_id,
            subcategory_id=subcategory_id,
            classification_id=classification_id,
            severity_id=severity_id,
            stage_id=stage_id,
            harm_level_id=harm_level_id,
            case_status_id=case_status_id,
            clinical_risk_type_id=clinical_risk_type_id,
            feedback_intent_type_id=feedback_intent_type_id,
            source_id=source_id,
            year=year,
            month=month,
            start_date=start_date,
            end_date=end_date
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "internal_server_error",
            "message": f"An error occurred while counting complaints: {str(e)}",
            "message_ar": f"حدث خطأ أثناء حساب الشكاوى: {str(e)}"
        })


@router.get("/export")
async def export_complaints_to_excel(
    search: Optional[str] = Query(None),
    issuing_org_unit_id: Optional[int] = Query(None),
    target_department_id: Optional[int] = Query(None),
    target_dept_parent_id: Optional[int] = Query(None),
    target_admin_id: Optional[int] = Query(None),
    domain_id: Optional[str] = Query(None),
    category_id: Optional[str] = Query(None),
    subcategory_id: Optional[str] = Query(None),
    classification_id: Optional[str] = Query(None),
    severity_id: Optional[str] = Query(None),
    stage_id: Optional[str] = Query(None),
    harm_level_id: Optional[str] = Query(None),
    case_status_id: Optional[str] = Query(None),
    clinical_risk_type_id: Optional[str] = Query(None),
    feedback_intent_type_id: Optional[str] = Query(None),
    source_id: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    month: Optional[int] = Query(None, ge=1, le=12),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    sort_by: str = Query("FeedbackRecievedDate"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    tab: Optional[str] = Query(None),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Export filtered complaints as Excel file.
    
    Accepts same query parameters as /api/complaints endpoint.
    Returns Excel file for download.
    
    **Example Request:**
    ```
    GET /api/complaints/export?domain_id=1&severity_id=2&start_date=2024-01-01
    ```
    
    **Returns:**
    Excel file with filtered complaints data
    
    **Content-Type:**
    application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
    """
    try:
        from datetime import datetime
        
        # Resolve org scope restriction
        user_roles = set(current_user.roles)
        is_full_access = bool(user_roles & _FULL_ACCESS_ROLES)
        restrict_to_unit_ids = None if is_full_access else current_user.allowed_unit_ids

        # Generate Excel file
        excel_file = export_complaints_excel(
            search=search,
            issuing_org_unit_id=issuing_org_unit_id,
            target_department_id=target_department_id,
            target_dept_parent_id=target_dept_parent_id,
            target_admin_id=target_admin_id,
            domain_id=domain_id,
            category_id=category_id,
            subcategory_id=subcategory_id,
            classification_id=classification_id,
            severity_id=severity_id,
            stage_id=stage_id,
            harm_level_id=harm_level_id,
            case_status_id=case_status_id,
            clinical_risk_type_id=clinical_risk_type_id,
            feedback_intent_type_id=feedback_intent_type_id,
            source_id=source_id,
            year=year,
            month=month,
            start_date=start_date,
            end_date=end_date,
            sort_by=sort_by,
            sort_order=sort_order,
            tab=tab,
            restrict_to_unit_ids=restrict_to_unit_ids,
        )

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"Complaints_Export_{timestamp}.xlsx"
        
        # Return as streaming response
        return StreamingResponse(
            excel_file,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "export_failed",
            "message": f"An error occurred while generating export: {str(e)}",
            "message_ar": f"حدث خطأ أثناء إنشاء التصدير: {str(e)}"
        })


@router.get("/{complaint_id}")
async def get_single_complaint(complaint_id: int):
    """
    Fetch full details of a single complaint record.
    
    Returns UNMASKED patient data (full MRN, full name).
    Used when navigating to EditRecord or viewing full details modal.
    
    **Example Request:**
    ```
    GET /api/complaints/1234
    ```
    
    **Returns:**
    - Full complaint record with all fields
    - Patient identifiers are NOT masked
    
    **Errors:**
    - 404: Complaint not found
    """
    try:
        result = get_complaint_by_id(complaint_id)
        
        if result is None:
            raise HTTPException(status_code=404, detail={
                "error": "complaint_not_found",
                "message": f"Complaint with ID {complaint_id} not found",
                "message_ar": f"لم يتم العثور على الشكوى ذات المعرف {complaint_id}"
            })
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "internal_server_error",
            "message": f"An error occurred while fetching complaint: {str(e)}",
            "message_ar": f"حدث خطأ أثناء جلب الشكوى: {str(e)}"
        })


@router.post("/export")
async def export_filtered_complaints(
    request: ExportRequest = Body(...),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Export filtered complaints as CSV or JSON.
    
    Creates an export job and returns metadata with download URL.
    Export uses same filters as current table view.
    
    **Example Request:**
    ```json
    POST /api/complaints/export
    {
      "format": "csv",
      "filters": {
        "status": "open",
        "issuing_dept_id": 12,
        "start_date": "2024-03-01",
        "end_date": "2024-03-31"
      },
      "columns": [
        "complaint_number",
        "received_date",
        "issuing_dept_name",
        "domain_name",
        "severity_name",
        "status"
      ],
      "include_patient_identifiers": false,
      "language": "ar"
    }
    ```
    
    **Returns:**
    - `export_id`: Unique export identifier
    - `file_name`: Generated filename
    - `download_url`: URL to download file (expires in 24 hours)
    - `record_count`: Number of records in export
    - `generated_at`: Export creation timestamp
    - `expires_at`: Export expiration timestamp
    - `status`: Export status (pending, processing, completed, failed)
    
    **Errors:**
    - 400: Invalid format or parameters
    """
    try:
        user_roles = set(current_user.roles)
        is_full_access = bool(user_roles & _FULL_ACCESS_ROLES)
        restrict_to_unit_ids = None if is_full_access else current_user.allowed_unit_ids

        result = export_complaints(
            export_format=request.format,
            filters=request.filters,
            columns=request.columns,
            include_patient_identifiers=request.include_patient_identifiers,
            language=request.language,
            restrict_to_unit_ids=restrict_to_unit_ids,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": "invalid_format",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "internal_server_error",
            "message": f"An error occurred while creating export: {str(e)}",
            "message_ar": f"حدث خطأ أثناء إنشاء التصدير: {str(e)}"
        })


@router.get("/views")
async def get_available_views():
    """
    Get predefined table view configurations.
    
    Returns available view presets with their column configurations.
    Views define which columns are displayed in the table.
    
    **Example Request:**
    ```
    GET /api/complaints/views
    ```
    
    **Returns:**
    - `views`: Array of view configurations
      - Each view has: view_id, view_name, columns, default_sort, preset_filters
    - `default_view`: Default view ID to use on page load
    
    **Available Views:**
    - `complete`: All columns (default)
    - `simplified`: Essential columns only
    - `red_flags_only`: Red flag complaints with preset filter
    """
    try:
        result = get_table_views()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "internal_server_error",
            "message": f"An error occurred while fetching views: {str(e)}",
            "message_ar": f"حدث خطأ أثناء جلب طرق العرض: {str(e)}"
        })


@router.delete("/{complaint_id}")
async def delete_complaint(
    complaint_id: int = Path(..., gt=0, description="Incident/complaint record ID to delete")
):
    """
    Soft delete (close) a complaint record - MAIN DELETE METHOD.
    
    Marks a complaint as closed (CaseStatusID = 3) without permanently deleting data.
    Data remains in the database for archival purposes and can be filtered out in views.
    
    **Example Request:**
    ```
    DELETE /api/complaints/156
    ```
    
    **Returns:**
    ```json
    {
        "success": true,
        "message": "Record marked as closed",
        "message_ar": "تم وضع علامة على السجل كمغلق",
        "complaint_id": 156,
        "case_status_id": 3,
        "closed_at": "2026-01-06T10:30:45.123456"
    }
    ```
    """
    try:
        closed_status_id = 3
        soft_delete_incident_case(
            incident_id=complaint_id,
            closed_status_id=closed_status_id
        )
        
        return {
            "success": True,
            "message": "Record marked as closed",
            "message_ar": "تم وضع علامة على السجل كمغلق",
            "complaint_id": complaint_id,
            "case_status_id": closed_status_id,
            "closed_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "deletion_failed",
            "message": f"Failed to delete record: {str(e)}",
            "message_ar": f"فشل حذف السجل: {str(e)}"
        })


@router.delete("/{complaint_id}/hard-delete")
async def hard_delete_complaint(
    complaint_id: int = Path(..., gt=0, description="Incident/complaint record ID to permanently delete")
):
    """
    Permanently delete a complaint record and all related data - IRREVERSIBLE.
    
    **IMPORTANT:** This is a HARD DELETE. Completely removes the record and all related data.
    Data CANNOT be recovered. Use soft delete instead for normal operations.
    
    Permanently removes:
    - Linked doctors
    - Target departments
    - Feedback records
    - Action items
    - Main case record
    
    **Example Request:**
    ```
    DELETE /api/complaints/156/hard-delete
    ```
    
    **Returns:**
    ```json
    {
        "success": true,
        "message": "Record permanently deleted",
        "message_ar": "تم حذف السجل نهائياً",
        "complaint_id": 156,
        "deleted_permanently": true,
        "deleted_at": "2026-01-06T10:30:45.123456"
    }
    ```
    """
    try:
        hard_delete_incident_case(complaint_id)
        
        return {
            "success": True,
            "message": "Record permanently deleted",
            "message_ar": "تم حذف السجل نهائياً",
            "complaint_id": complaint_id,
            "deleted_permanently": True,
            "deleted_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "permanent_deletion_failed",
            "message": f"Failed to permanently delete record: {str(e)}",
            "message_ar": f"فشل الحذف النهائي للسجل: {str(e)}"
        })


@router.post("/backfill-incidents")
async def backfill_incident_parents():
    """
    Create a parent Incident for every APP_IncidentCase that has no incident_id.
    Safe to call multiple times — only processes cases where incident_id IS NULL.
    Returns the count of cases that were fixed.
    """
    from core.database import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT IncidentRequestCaseID FROM dbo.APP_IncidentCase WHERE incident_id IS NULL")
        orphans = [row[0] for row in cursor.fetchall()]
        if not orphans:
            return {"fixed": 0, "message": "No orphaned cases found"}

        fixed = 0
        for case_id in orphans:
            cursor.execute(
                "SELECT PatientName, FeedbackIntentTypeID, IssuingOrgUnitID, ComplaintText, BuildingID, isINPatient, CreatedByUserID FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?",
                case_id
            )
            row = cursor.fetchone()
            if not row:
                continue
            cursor.execute(
                """INSERT INTO dbo.APP_Incident (patient_name, feedback_intent_type_id, issuing_org_unit_id, complaint_summary, building_id, is_inpatient, created_by_user_id)
                   OUTPUT INSERTED.incident_id
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                row[0], row[1], row[2], row[3], row[4], row[5], row[6]
            )
            incident_id = cursor.fetchone()[0]
            cursor.execute("UPDATE dbo.APP_IncidentCase SET incident_id = ? WHERE IncidentRequestCaseID = ?", incident_id, case_id)
            fixed += 1

        conn.commit()
        return {"fixed": fixed, "message": f"Created Incident parents for {fixed} case(s)"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail={"error": "backfill_failed", "message": str(e)})
    finally:
        cursor.close()
        conn.close()
