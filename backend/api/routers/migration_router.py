"""
PHASE K — K-API-1 — Migration Router (API V1)

API router for legacy case migration operations.

This router provides endpoints for:
- Listing legacy cases
- Viewing legacy case details
- Migrating legacy cases to new system
- Viewing migration progress

AUTHORIZATION
-------------
Required roles: SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR, WORKER

ROUTER PREFIX
-------------
/api/migration
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any
from core.constants.roles import SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR, WORKER
from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_role
from ..services.legacy_case_service import list_legacy_cases_paged, get_legacy_case_detail
from ..services.migration_service import migrate_legacy_case
from ..services.migration_progress_service import get_migration_progress


router = APIRouter(
    prefix="/api/migration",
    tags=["migration"],
)


# =========================================================
# LEGACY CASE LIST ENDPOINT
# =========================================================

@router.get("/legacy/list")
def legacy_list_endpoint(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Records per page"),
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List legacy cases with pagination.
    
    Returns:
        {
            "cases": [
                {
                    "legacy_case_id": int,
                    "complaint_text": str,
                    "patient_name": str,
                    "feedback_received_date": datetime,
                    "case_status_id": int,
                    "created_at": datetime,
                    "migrated": bool
                },
                ...
            ],
            "total": int
        }
    """
    # Authorization guard
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR, WORKER])
    
    try:
        result = list_legacy_cases_paged(page, page_size)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "LEGACY_LIST_FAILED",
                "message": f"Failed to retrieve legacy cases: {str(e)}",
                "message_ar": "فشل في استرجاع الحالات القديمة"
            }
        )


# =========================================================
# LEGACY CASE DETAIL ENDPOINT
# =========================================================

@router.get("/legacy/{legacy_id}")
def legacy_detail_endpoint(
    legacy_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed legacy case record.
    
    Returns:
        Full legacy case record with all fields.
        
    Raises:
        404: If legacy case not found
    """
    # Authorization guard
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR, WORKER])
    
    try:
        case = get_legacy_case_detail(legacy_id)
        
        if case is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "LEGACY_CASE_NOT_FOUND",
                    "message": f"Legacy case with ID {legacy_id} not found",
                    "message_ar": f"لم يتم العثور على الحالة القديمة برقم {legacy_id}"
                }
            )
        
        return case
        
    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "LEGACY_DETAIL_FAILED",
                "message": f"Failed to retrieve legacy case: {str(e)}",
                "message_ar": "فشل في استرجاع تفاصيل الحالة القديمة"
            }
        )


# =========================================================
# MIGRATE LEGACY CASE ENDPOINT
# =========================================================

@router.post("/migrate/{legacy_id}")
def migrate_endpoint(
    legacy_id: int,
    payload: Dict[str, Any],
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Migrate a legacy case to the new system.
    
    Args:
        legacy_id: Legacy case ID to migrate
        payload: Migration payload with case data
        
    Returns:
        {
            "success": True,
            "status": "MIGRATED" | "ALREADY_MIGRATED",
            "legacy_case_id": int,
            "new_case_id": int
        }
        
    Raises:
        400: If migration fails (validation, conflicts, etc.)
    """
    # Authorization guard
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR, WORKER])
    
    try:
        result = migrate_legacy_case(
            legacy_id,
            payload,
            current_user.user_id
        )
        
        # Check if migration failed
        if result.get("success") == False:
            error_code = result.get("error", "MIGRATION_FAILED")
            error_message = result.get("message", "Migration operation failed")
            error_message_ar = result.get("message_ar", "فشلت عملية الترحيل")
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": error_code,
                    "message": error_message,
                    "message_ar": error_message_ar
                }
            )
        
        # Success - return result directly
        return result
        
    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "MIGRATION_ERROR",
                "message": f"Migration operation error: {str(e)}",
                "message_ar": "خطأ في عملية الترحيل"
            }
        )


# =========================================================
# MIGRATION PROGRESS ENDPOINT
# =========================================================

@router.get("/progress")
def progress_endpoint(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get migration progress statistics.
    
    Authorization: SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR, WORKER
    
    Returns:
        {
            "total_legacy": int,    # Total number of legacy cases in database
            "migrated_total": int,  # Number of cases already migrated
            "percent": float        # Percentage complete (rounded to 1 decimal)
        }
    """
    # Authorization guard - SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR, and WORKER
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR, WORKER])
    
    try:
        progress = get_migration_progress()
        
        # Transform response to match API contract
        return {
            "total_legacy": progress["total_cases"],
            "migrated_total": progress["migrated_cases"],
            "percent": progress["percent_complete"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "PROGRESS_FAILED",
                "message": f"Failed to retrieve migration progress: {str(e)}",
                "message_ar": "فشل في استرجاع تقدم الترحيل"
            }
        )
