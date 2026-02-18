"""
Action Log Report Builder Service (API V2)
Orchestration layer for Action Item Accountability Log reports.

PHASE F — ACTION LOG REPORT
Coordinates DB queries, season resolution, scope filtering, and classification.

NO formatting/Word generation here - this is pure data orchestration.
"""

from typing import Dict, Any
from datetime import date

from backend.api_v2.services import season_service
from backend.api_v2.services import action_log_classification_service
from backend.api_v2.db_layer import action_item_subcase_db


# ============================================================================
# REPORT BUILDER
# ============================================================================

def build_action_log_report(
    conn,
    season_id: int,
    current_user,
    today: date
) -> Dict[str, Any]:
    """
    Build Action Log report dataset for a given season and user scope.
    
    This is the main orchestration function for Phase F Action Log reports.
    
    Flow:
    1. Resolve season → start_date, end_date
    2. Query action items by due date range (DB layer)
    3. Apply organizational scope filtering (Phase 2.5)
    4. Classify items → completed vs not completed
    5. Compute overdue status
    6. Return structured report data (ready for Word generator)
    
    Scope Behavior (LOCKED):
    - Uses current_user.allowed_unit_ids from Phase 2.5 scope engine
    - Filters action items to only those in user's authorized org units
    - NO special "only my items" logic - scope is organizational
    
    Status Gate (LOCKED):
    - Only includes action items from subcases >= ADMIN_APPROVED
    - CANCELLED items already excluded by DB query
    
    Args:
        conn: Database connection (passed to avoid multiple connections)
        season_id: Season UniqueID for date range resolution
        current_user: Authenticated user with allowed_unit_ids and display_name
        today: Current date for overdue computation
        
    Returns:
        Dict with structure:
        {
            "meta": {
                "season_id": int,
                "season_name": str | None,
                "start_date": date,
                "end_date": date,
                "generated_at": date,
                "generated_by": str
            },
            "completed_items": list[dict],
            "not_completed_items": list[dict],
            "totals": {
                "completed_count": int,
                "not_completed_count": int,
                "overdue_count": int
            }
        }
        
    Raises:
        ValueError: If season_id not found
        season_service.SeasonNotFoundError: If season doesn't exist
        season_service.InvalidSeasonDataError: If season has invalid dates
    """
    
    # ========================================================================
    # STEP A — RESOLVE SEASON
    # ========================================================================
    
    season_data = season_service.resolve_season_date_range(season_id)
    
    # Extract dates
    start_date = season_data["start_date"]
    end_date = season_data["end_date"]
    season_name = season_data.get("season_name")
    
    # ========================================================================
    # STEP B — FETCH ROWS FROM DB
    # ========================================================================
    
    rows = action_item_subcase_db.get_action_items_by_due_date_range(
        conn,
        start_date,
        end_date
    )
    
    # ========================================================================
    # STEP C — SCOPE FILTERING (SERVICE LEVEL)
    # ========================================================================
    
    # Get user's allowed organizational units (Phase 2.5 scope)
    allowed_unit_ids = set(current_user.allowed_unit_ids)
    
    # Filter rows to only those in user's scope
    scoped_rows = []
    for row in rows:
        target_org_unit_id = row.get("target_org_unit_id")
        if target_org_unit_id in allowed_unit_ids:
            scoped_rows.append(row)
    
    # ========================================================================
    # STEP D — STATUS CLASSIFICATION
    # ========================================================================
    
    classification_result = action_log_classification_service.classify_action_items(
        scoped_rows,
        today
    )
    
    # ========================================================================
    # STEP E — ENRICH METADATA
    # ========================================================================
    
    report_meta = {
        "season_id": season_id,
        "season_name": season_name,
        "start_date": start_date,
        "end_date": end_date,
        "generated_at": today,
        "generated_by": current_user.display_name if hasattr(current_user, 'display_name') else None
    }
    
    # ========================================================================
    # STEP F — RETURN FINAL STRUCTURE
    # ========================================================================
    
    return {
        "meta": report_meta,
        "completed_items": classification_result["completed_items"],
        "not_completed_items": classification_result["not_completed_items"],
        "totals": classification_result["totals"]
    }
