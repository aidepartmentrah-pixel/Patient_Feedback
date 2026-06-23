"""
Action Log Router (API V2)
Phase F — Action Item Log Extraction endpoint for Word export.

This router provides:
- Export endpoint for Action Log reports (Word format)

Security: All endpoints protected by authentication and authorization guard.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from datetime import date
from backend.api.schemas.auth_models import CurrentUser
from backend.api_v2.guards.action_log_guards import require_action_log_role
from backend.api_v2.services.action_log_report_service import (
    build_action_log_report,
    build_action_log_report_by_date_range,
)
from backend.api_v2.services.action_log_word_generator import generate_action_log_word
from core.database import get_connection


# ============================================================
# ROUTER DEFINITION
# ============================================================
router = APIRouter(prefix="/api/v2/action-log", tags=["Action Log v2"])


# ============================================================
# EXPORT ENDPOINT
# ============================================================

@router.get("/export")
def export_action_log_report(
    season_id: int,
    current_user: CurrentUser = Depends(require_action_log_role)
) -> Response:
    """
    Export Action Log report as Word document.
    
    Phase F Action Log export endpoint — API v2 only.
    
    Generates a Word document containing:
    - Completed action items (DONE, VERIFIED)
    - Not completed action items (all other statuses except CANCELLED)
    - Overdue indicators for items past due date
    - Filtered by season DueDate range
    - Scoped to user's organizational units (Phase 2.5)
    
    Args:
        season_id: Season UniqueID for date range resolution
        current_user: Authenticated user with allowed_unit_ids
        
    Returns:
        Word document (.docx) with Action Log report
        
    Security:
        - Requires authentication (get_current_user dependency)
        - Requires SOFTWARE_ADMIN or WORKER role (require_action_log_role guard)
        - Scope filtering applied by service layer
        
    Raises:
        401: Not authenticated
        403: Not authorized (missing SOFTWARE_ADMIN or WORKER role)
        404: Season not found
        500: Internal server error
    """
    # Get database connection
    conn = get_connection()
    
    try:
        # Get today's date for overdue computation
        today = date.today()
        
        # Build report data (orchestration + scope filtering)
        report_data = build_action_log_report(
            conn=conn,
            season_id=season_id,
            current_user=current_user,
            today=today
        )
        
        # Generate Word document bytes
        doc_bytes = generate_action_log_word(report_data)
        
        # Construct filename
        filename = f"action_log_season_{season_id}.docx"
        
        # Return Word document as download
        return Response(
            content=doc_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
        
    finally:
        # Always close connection
        conn.close()


# ============================================================
# EXPORT BY DATE RANGE ENDPOINT
# ============================================================

@router.get("/export-by-date")
def export_action_log_by_date_range(
    date_from: date,
    date_to: date,
    current_user: CurrentUser = Depends(require_action_log_role)
) -> Response:
    """
    Export Action Log report for a custom date range as Word document.

    Accepts date_from and date_to as YYYY-MM-DD query parameters.
    Includes all action items with DueDate in [date_from, date_to] inclusive.
    Scoped to user's organizational units (Phase 2.5).

    Raises:
        422: date_from is after date_to
        401: Not authenticated
        403: Not authorized
    """
    if date_from > date_to:
        raise HTTPException(
            status_code=422,
            detail="date_from must not be after date_to"
        )

    conn = get_connection()
    try:
        today = date.today()
        report_data = build_action_log_report_by_date_range(
            conn=conn,
            start_date=date_from,
            end_date=date_to,
            current_user=current_user,
            today=today,
        )
        doc_bytes = generate_action_log_word(report_data)
        filename = f"action_log_{date_from}_{date_to}.docx"
        return Response(
            content=doc_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    finally:
        conn.close()
