"""
Action Item Service
Handles business logic for creating and managing action items.

Key Business Rule:
ActionItems must belong to EXACTLY ONE parent:
- IncidentRequestCaseID (incident case)
- SeasonalReportID (seasonal organizational report)
- SeasonCaseID (season case)
"""

from datetime import date
from typing import Dict, Any, Optional

from backend.api.db_layer.action_items import (
    create_action_item as db_create_action_item,
    update_action_item as db_update_action_item,
    get_action_item_by_id as db_get_action_item_by_id,
    list_action_items_for_incident,
    list_action_items_for_seasonal_report,
    list_action_items_for_season,
    mark_action_item_done,
)


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _validate_exactly_one_parent(
    incident_case_id: Optional[int],
    seasonal_report_id: Optional[int],
    season_case_id: Optional[int]
) -> None:
    """
    Enforce critical business invariant: ActionItem must belong to exactly ONE parent.
    
    Args:
        incident_case_id: IncidentRequestCaseID (nullable)
        seasonal_report_id: SeasonalReportID (nullable)
        season_case_id: SeasonCaseID (nullable)
    
    Raises:
        ValueError: If the rule is violated (0 parents or 2+ parents)
    """
    parent_count = sum([
        incident_case_id is not None,
        seasonal_report_id is not None,
        season_case_id is not None
    ])
    
    if parent_count == 0:
        raise ValueError(
            "ActionItem must belong to exactly one parent. "
            "Provide one of: incident_case_id, seasonal_report_id, or season_case_id."
        )
    
    if parent_count > 1:
        raise ValueError(
            "ActionItem must belong to exactly one parent. "
            f"Found {parent_count} non-null parent IDs. "
            "Only one of incident_case_id, seasonal_report_id, or season_case_id is allowed."
        )


# ============================================================
# SERVICE LAYER FUNCTIONS
# ============================================================

def create_action_item(
    *,
    action_title: str,
    created_by_user_id: int,
    incident_case_id: Optional[int] = None,
    seasonal_report_id: Optional[int] = None,
    season_case_id: Optional[int] = None,
    action_description: Optional[str] = None,
    due_date: Optional[date] = None,
) -> int:
    """
    Create a new action item with parent validation.
    
    Business Rules:
    - MUST belong to exactly one parent (incident, seasonal report, or season)
    - action_title is required
    - created_by_user_id is required
    
    Args:
        action_title: Title of the action item (required)
        created_by_user_id: User ID who created the action (required)
        incident_case_id: Link to incident case (optional, mutually exclusive with others)
        seasonal_report_id: Link to seasonal report (optional, mutually exclusive with others)
        season_case_id: Link to season case (optional, mutually exclusive with others)
        action_description: Detailed description of the action (optional)
        due_date: Due date for completion (optional)
    
    Returns:
        int: ActionItemID of the newly created action item
    
    Raises:
        ValueError: If parent validation fails or required fields are missing
    """
    # Validate required fields
    if not action_title or not action_title.strip():
        raise ValueError("action_title is required and cannot be empty")
    
    if not created_by_user_id:
        raise ValueError("created_by_user_id is required")
    
    # Enforce exactly-one-parent rule
    _validate_exactly_one_parent(
        incident_case_id=incident_case_id,
        seasonal_report_id=seasonal_report_id,
        season_case_id=season_case_id
    )
    
    # Delegate to DB layer after validation passes
    return db_create_action_item(
        action_title=action_title,
        created_by_user_id=created_by_user_id,
        incident_case_id=incident_case_id,
        seasonal_report_id=seasonal_report_id,
        season_case_id=season_case_id,
        action_description=action_description,
        due_date=due_date,
    )


def update_action_item(
    action_item_id: int,
    updates: Dict[str, Any]
) -> None:
    """
    Update an existing action item with validation.
    
    Business Rules:
    - Cannot change parent associations (IncidentRequestCaseID, SeasonalReportID, SeasonCaseID)
    - Only editable fields can be updated (ActionTitle, ActionDescription, DueDate, IsDone, DateSubmitted)
    
    Args:
        action_item_id: Primary key of the action item to update
        updates: Dictionary of fields to update
    
    Raises:
        ValueError: If attempting to update parent associations or other immutable fields
    """
    # Block attempts to change parent associations
    FORBIDDEN_FIELDS = {
        "IncidentRequestCaseID",
        "SeasonalReportID",
        "SeasonCaseID",
        "incident_case_id",
        "seasonal_report_id",
        "season_case_id",
    }
    
    forbidden_found = FORBIDDEN_FIELDS & set(updates.keys())
    if forbidden_found:
        raise ValueError(
            f"Cannot update parent associations. "
            f"Attempted to modify forbidden fields: {', '.join(forbidden_found)}"
        )
    
    # Delegate to DB layer
    db_update_action_item(action_item_id=action_item_id, updates=updates)


def get_action_item(action_item_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve an action item by ID.
    
    Args:
        action_item_id: Primary key of the action item
    
    Returns:
        dict: Action item record, or None if not found
    """
    return db_get_action_item_by_id(action_item_id)


def get_action_items_for_incident(incident_case_id: int) -> list[Dict[str, Any]]:
    """
    List all action items for a specific incident case.
    
    Args:
        incident_case_id: IncidentRequestCaseID
    
    Returns:
        list[dict]: Action items ordered by CreatedAt DESC
    """
    return list_action_items_for_incident(incident_case_id)


def get_action_items_for_seasonal_report(seasonal_report_id: int) -> list[Dict[str, Any]]:
    """
    List all action items for a specific seasonal report.
    
    Args:
        seasonal_report_id: SeasonalReportID
    
    Returns:
        list[dict]: Action items ordered by CreatedAt DESC
    """
    return list_action_items_for_seasonal_report(seasonal_report_id)


def get_action_items_for_season(season_case_id: int) -> list[Dict[str, Any]]:
    """
    List all action items for a specific season case.
    
    Args:
        season_case_id: SeasonCaseID
    
    Returns:
        list[dict]: Action items ordered by CreatedAt DESC
    """
    return list_action_items_for_season(season_case_id)


def mark_action_done(action_item_id: int) -> None:
    """
    Mark an action item as completed.
    
    Sets IsDone = 1 and DateSubmitted = current date.
    
    Args:
        action_item_id: Primary key of the action item
    """
    mark_action_item_done(action_item_id)
