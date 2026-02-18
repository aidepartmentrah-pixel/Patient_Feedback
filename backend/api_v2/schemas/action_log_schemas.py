"""
📋 PHASE F — ACTION LOG REPORT SCHEMAS (API V2)

Data contract definitions for Action Item Accountability Log reports.
Used by DB layer, service layer, and Word formatter.

SOURCE OF TRUTH
- APP_SubcaseActionItem (action items table)
- APP_AdministrativeSubcase (parent subcase for scoping)
- APP_Users (for assigned user display names)
- APP_OrganizationalUnit (for target org unit names)
- APP_Lookup_SubcaseActionItemStatus (status definitions)

STATUS CLASSIFICATION RULES (LOCKED BY DECISION BLOCK)
- COMPLETED: Status in (DONE, VERIFIED)
- NOT COMPLETED: All other statuses EXCEPT CANCELLED
- EXCLUDED: CANCELLED status is excluded entirely from report
- SUBCASE GATE: Only include action items where parent subcase >= ADMIN_APPROVED

OVERDUE DEFINITION
- due_date < today
- AND completed_at is NULL
- Computed at runtime, not stored
"""

from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_overdue_fields(
    due_date: Optional[date],
    completed_at: Optional[datetime],
    today: date
) -> tuple[bool, Optional[int]]:
    """
    Compute overdue status and days overdue.
    
    Logic (LOCKED):
    - If due_date is None -> not overdue
    - If completed_at is not None -> not overdue (already completed)
    - If due_date < today AND completed_at is None -> overdue
    
    Args:
        due_date: The action item's due date
        completed_at: Completion timestamp (None if not completed)
        today: Current date for comparison
        
    Returns:
        (is_overdue: bool, days_overdue: int | None)
        days_overdue is None if not overdue, otherwise positive integer
    """
    if due_date is None:
        return False, None
    
    if completed_at is not None:
        return False, None
    
    if due_date < today:
        days_overdue = (today - due_date).days
        return True, days_overdue
    
    return False, None


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class ActionLogReportRequest(BaseModel):
    """
    Request parameters for generating an Action Log report.
    
    The season_id determines the date range filter:
    - Queries Season table for StartDate/EndDate
    - Filters action items by: DueDate BETWEEN StartDate AND EndDate
    """
    season_id: int = Field(..., description="Season/Quarter ID for date range filtering")


# ============================================================================
# DATA ITEM SCHEMAS
# ============================================================================

class ActionLogItem(BaseModel):
    """
    A single action item entry in the report.
    
    Represents one row in the generated Word report tables.
    Contains all data needed for display and classification.
    
    STATUS CLASSIFICATION:
    - DONE or VERIFIED -> Completed table
    - All others (except CANCELLED) -> Not Completed table
    - CANCELLED -> excluded from report entirely
    
    OVERDUE COMPUTATION:
    - Computed via compute_overdue_fields() utility
    - Only relevant for not-completed items
    """
    action_item_id: int
    subcase_id: int
    title: str
    description: Optional[str] = None
    status: str  # StatusCode from APP_Lookup_SubcaseActionItemStatus
    due_date: Optional[date] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    
    # User assignment
    assigned_to_user_id: Optional[int] = None
    assigned_to_display_name: Optional[str] = None  # From APP_Users.DisplayName
    
    # Organizational context
    target_org_unit_id: Optional[int] = None
    target_org_unit_name: Optional[str] = None  # Section/Dept/Admin name
    
    # Computed fields
    is_overdue: bool = Field(..., description="Computed: due_date < today AND completed_at is NULL")
    days_overdue: Optional[int] = Field(None, description="Days past due date, if overdue")


# ============================================================================
# REPORT DATA SCHEMAS
# ============================================================================

class ActionLogReportData(BaseModel):
    """
    Complete report data structure.
    
    Contains all action items grouped by completion status,
    plus metadata about the season and generation time.
    
    TABLE STRUCTURE:
    - Table A (Completed): completed_items list
    - Table B (Not Completed): not_completed_items list
      - Overdue items in Table B are visually emphasized
    
    STATUS GATE:
    - All items come from subcases with status >= ADMIN_APPROVED
    - This matches Follow Up calendar visibility
    """
    # Season context
    season_id: int
    season_name: Optional[str] = None  # e.g., "Q1 2026" or Arabic equivalent
    start_date: date
    end_date: date
    generated_at: datetime
    
    # Grouped action items
    completed_items: list[ActionLogItem] = Field(
        default_factory=list,
        description="Action items with status DONE or VERIFIED"
    )
    not_completed_items: list[ActionLogItem] = Field(
        default_factory=list,
        description="Action items with status not in (DONE, VERIFIED, CANCELLED)"
    )
    
    # Summary statistics
    totals: dict = Field(
        default_factory=dict,
        description="Total counts: completed_count, not_completed_count, overdue_count"
    )


class ActionLogReportResponse(BaseModel):
    """
    API response wrapper for action log report generation.
    
    Used by service layer and router to return:
    - Success case: data populated, error=None
    - Failure case: data=None, error contains message
    """
    success: bool
    data: Optional[ActionLogReportData] = None
    error: Optional[str] = None
