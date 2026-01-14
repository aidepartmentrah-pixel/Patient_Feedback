"""
Action Items Router
Exposes API endpoints for querying and managing action items.
"""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Path

from backend.api.services.action_item_service import (
    get_action_item,
    get_action_items_for_incident,
    get_action_items_for_seasonal_report,
    get_action_items_for_season,
    create_action_item,
    update_action_item,
    mark_action_done,
)


# ============================================================
# ROUTER
# ============================================================
router = APIRouter(prefix="/api/action-items", tags=["Action Items"])


# ============================================================
# QUERY ENDPOINTS
# ============================================================

@router.get("/{action_item_id}", response_model=Dict[str, Any], status_code=200)
def get_action_item_by_id(
    action_item_id: int = Path(..., description="Primary key of the action item")
):
    """
    Retrieve a single action item by ID.
    
    **Path Parameter:**
    - `action_item_id`: Primary key (ActionItemID)
    
    **Returns:**
    - Action item record with all fields
    
    **Errors:**
    - `404`: Action item not found
    """
    action_item = get_action_item(action_item_id)
    
    if not action_item:
        raise HTTPException(
            status_code=404,
            detail=f"Action item with ID {action_item_id} not found"
        )
    
    return action_item


@router.get("/by-incident/{incident_case_id}", response_model=List[Dict[str, Any]], status_code=200)
def get_action_items_by_incident(
    incident_case_id: int = Path(..., description="IncidentRequestCaseID")
):
    """
    List all action items for a specific incident case.
    
    **Path Parameter:**
    - `incident_case_id`: IncidentRequestCaseID
    
    **Returns:**
    - List of action items ordered by CreatedAt DESC
    - Empty list if no action items found (not an error)
    """
    return get_action_items_for_incident(incident_case_id)


@router.get("/by-seasonal-report/{seasonal_report_id}", response_model=List[Dict[str, Any]], status_code=200)
def get_action_items_by_seasonal_report(
    seasonal_report_id: int = Path(..., description="SeasonalReportID")
):
    """
    List all action items for a specific seasonal report.
    
    **Path Parameter:**
    - `seasonal_report_id`: SeasonalReportID from APP_SeasonalOrgUnitReport
    
    **Returns:**
    - List of action items ordered by CreatedAt DESC
    - Empty list if no action items found (not an error)
    
    **Use Case:**
    - Display action items assigned to a seasonal organizational report
    - Track compliance-related action items for a specific season
    """
    return get_action_items_for_seasonal_report(seasonal_report_id)


@router.get("/by-season/{season_case_id}", response_model=List[Dict[str, Any]], status_code=200)
def get_action_items_by_season(
    season_case_id: int = Path(..., description="SeasonCaseID")
):
    """
    List all action items for a specific season case.
    
    **Path Parameter:**
    - `season_case_id`: SeasonCaseID
    
    **Returns:**
    - List of action items ordered by CreatedAt DESC
    - Empty list if no action items found (not an error)
    """
    return get_action_items_for_season(season_case_id)


# ============================================================
# MUTATION ENDPOINTS
# ============================================================

@router.post("/{action_item_id}/mark-done", status_code=200)
def mark_action_item_done(
    action_item_id: int = Path(..., description="Primary key of the action item")
):
    """
    Mark an action item as completed.
    
    **Path Parameter:**
    - `action_item_id`: Primary key (ActionItemID)
    
    **Behavior:**
    - Sets IsDone = 1
    - Sets DateSubmitted = current date
    
    **Returns:**
    - `{"status": "ok"}`
    
    **Errors:**
    - `404`: Action item not found (database will fail silently)
    """
    try:
        mark_action_done(action_item_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to mark action item as done: {str(e)}")
