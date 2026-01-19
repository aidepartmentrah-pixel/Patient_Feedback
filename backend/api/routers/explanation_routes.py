"""
Explanation Routes
==================
API endpoints for the explanation workflow.

Handles:
- Retrieving cases needing explanations
- Submitting explanations with action items
- Managing RequiresExplanation flag
- Dashboard statistics
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Path, Query, Body
from pydantic import BaseModel, Field
from datetime import date

from backend.api.services.explanation_service import (
    get_pending_explanations,
    get_explanation_dashboard_statistics,
    get_case_explanation_details,
    submit_explanation,
    validate_explanation_submission,
    toggle_requires_explanation,
    admin_force_close_case,
    check_and_close_case_if_complete,
    mark_action_item_complete_and_check_case,
    get_case_completion_status
)


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class ActionItemCreate(BaseModel):
    """Action item to create with explanation"""
    title: str = Field(..., min_length=3, max_length=200, description="Action item title")
    description: Optional[str] = Field(None, max_length=1000, description="Detailed description")
    due_date: Optional[str] = Field(None, description="Due date in YYYY-MM-DD format")


class SubmitExplanationRequest(BaseModel):
    """Request body for submitting an explanation"""
    explanation_text: str = Field(..., min_length=10, max_length=5000, description="Explanation text")
    action_items: Optional[List[ActionItemCreate]] = Field(default=[], description="Optional action items to create")
    user_id: int = Field(..., description="ID of user submitting explanation")


class UpdateRequiresExplanationRequest(BaseModel):
    """Request body for updating RequiresExplanation flag"""
    requires_explanation: bool = Field(..., description="Whether case requires explanation")
    reason: Optional[str] = Field(None, min_length=10, description="Reason for change (required for admins)")
    user_id: int = Field(..., description="ID of user making the change")


class ForceCloseRequest(BaseModel):
    """Request body for admin force close"""
    reason: str = Field(..., min_length=20, description="Reason for force closing (minimum 20 characters)")
    user_id: int = Field(..., description="ID of admin user")


class MarkActionCompleteRequest(BaseModel):
    """Request body for marking action item complete"""
    action_item_id: int = Field(..., description="Action item ID to mark complete")
    user_id: int = Field(..., description="ID of user marking complete")


# ============================================================
# ROUTER
# ============================================================
router = APIRouter(prefix="/api/explanations", tags=["Explanations"])


# ============================================================
# QUERY ENDPOINTS
# ============================================================

@router.get("/pending", response_model=Dict[str, Any], status_code=200)
def get_pending_explanations_endpoint(
    dept_id: Optional[int] = Query(None, description="Filter by department ID"),
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    case_type: Optional[str] = Query(None, description="Filter by case type"),
    include_red_flags_only: bool = Query(False, description="Only return Red Flag/Never Event cases")
):
    """
    Get all cases that are pending explanation submission.
    
    **Query Parameters:**
    - `dept_id`: Filter by target department
    - `start_date`: Filter cases from this date onward (YYYY-MM-DD)
    - `end_date`: Filter cases up to this date (YYYY-MM-DD)
    - `case_type`: Filter by case type (e.g., 'Red Flag', 'Never Event')
    - `include_red_flags_only`: If true, only return Red Flag/Never Event cases
    
    **Returns:**
    - List of cases awaiting explanation
    - Statistics (total count, by type)
    - Filtered results based on query parameters
    
    **FSM State:** Cases with ExplanationStatus = 'Waiting'
    """
    result = get_pending_explanations(
        department_id=dept_id,
        start_date=start_date,
        end_date=end_date,
        case_type=case_type,
        include_red_flags_only=include_red_flags_only
    )
    
    if not result['success']:
        raise HTTPException(
            status_code=400,
            detail=result.get('error', 'Failed to retrieve pending explanations')
        )
    
    return result


@router.get("/statistics", response_model=Dict[str, Any], status_code=200)
def get_explanation_statistics():
    """
    Get dashboard statistics for explanation workflow.
    
    **Returns:**
    - Counts by explanation status (Waiting, Responded, Forcibly Closed, etc.)
    - Overdue cases (over 7 days, over 30 days)
    - Aggregate totals
    
    **Use Case:** Dashboard widgets and management reporting
    """
    result = get_explanation_dashboard_statistics()
    
    if not result['success']:
        raise HTTPException(
            status_code=500,
            detail=result.get('error', 'Failed to retrieve statistics')
        )
    
    return result


@router.get("/{case_id}", response_model=Dict[str, Any], status_code=200)
def get_case_explanation_details_endpoint(
    case_id: int = Path(..., description="Incident case ID")
):
    """
    Get detailed explanation information for a specific case.
    
    **Path Parameter:**
    - `case_id`: IncidentRequestCaseID
    
    **Returns:**
    - Full case details
    - Validation information (can_submit_explanation, has_existing_explanation, etc.)
    - Current explanation status and FSM state
    
    **Errors:**
    - `404`: Case not found
    """
    result = get_case_explanation_details(case_id)
    
    if not result['success']:
        raise HTTPException(
            status_code=404,
            detail=result.get('error', f'Case {case_id} not found')
        )
    
    return result


@router.get("/{case_id}/completion-status", response_model=Dict[str, Any], status_code=200)
def get_case_completion_status_endpoint(
    case_id: int = Path(..., description="Incident case ID")
):
    """
    Get completion status including action items progress.
    
    **Path Parameter:**
    - `case_id`: IncidentRequestCaseID
    
    **Returns:**
    - Action items completion statistics
    - Percentage complete
    - Whether case can be closed
    
    **Use Case:** Progress tracking UI
    """
    result = get_case_completion_status(case_id)
    
    if not result['success']:
        raise HTTPException(
            status_code=404,
            detail=result.get('error', f'Case {case_id} not found')
        )
    
    return result


# ============================================================
# MUTATION ENDPOINTS
# ============================================================

@router.post("/{case_id}", response_model=Dict[str, Any], status_code=200)
def submit_explanation_endpoint(
    case_id: int = Path(..., description="Incident case ID"),
    request: SubmitExplanationRequest = Body(...)
):
    """
    Submit an explanation for a case with optional action items.
    
    **Path Parameter:**
    - `case_id`: IncidentRequestCaseID
    
    **Request Body:**
    - `explanation_text`: Explanation content (min 10 characters)
    - `action_items`: Optional list of action items to create
    - `user_id`: User submitting the explanation
    
    **FSM Transition:**
    - (Open + Waiting) → (In Progress + Responded)
    
    **Returns:**
    - Updated case information
    - Created action items (if any)
    
    **Errors:**
    - `400`: Validation failed (case doesn't require explanation, is closed, etc.)
    - `404`: Case not found
    """
    # Convert action items to dict format
    action_items_data = None
    if request.action_items:
        action_items_data = [
            {
                "title": item.title,
                "description": item.description,
                "due_date": item.due_date
            }
            for item in request.action_items
        ]
    
    result = submit_explanation(
        case_id=case_id,
        explanation_text=request.explanation_text,
        action_items=action_items_data,
        user_id=request.user_id
    )
    
    if not result['success']:
        raise HTTPException(
            status_code=400,
            detail=result.get('error', 'Failed to submit explanation')
        )
    
    return result


@router.put("/{case_id}/requires-explanation", response_model=Dict[str, Any], status_code=200)
def update_requires_explanation_flag(
    case_id: int = Path(..., description="Incident case ID"),
    request: UpdateRequiresExplanationRequest = Body(...)
):
    """
    Toggle the RequiresExplanation flag for a case.
    
    **Path Parameter:**
    - `case_id`: IncidentRequestCaseID
    
    **Request Body:**
    - `requires_explanation`: New flag value (true/false)
    - `reason`: Reason for change (optional but recommended)
    - `user_id`: User making the change
    
    **Use Case:** 
    - Admin override for ordinary complaints
    - Policy-based flagging of cases needing explanation
    
    **Note:** Red Flag and Never Event cases always require explanation regardless of this flag
    
    **Errors:**
    - `404`: Case not found
    """
    result = toggle_requires_explanation(
        case_id=case_id,
        requires_explanation=request.requires_explanation,
        user_id=request.user_id,
        reason=request.reason
    )
    
    if not result['success']:
        raise HTTPException(
            status_code=400,
            detail=result.get('error', 'Failed to update flag')
        )
    
    return result


@router.post("/{case_id}/force-close", response_model=Dict[str, Any], status_code=200)
def admin_force_close_case_endpoint(
    case_id: int = Path(..., description="Incident case ID"),
    request: ForceCloseRequest = Body(...)
):
    """
    Admin endpoint to force close a case without completing action items.
    
    **Path Parameter:**
    - `case_id`: IncidentRequestCaseID
    
    **Request Body:**
    - `reason`: Detailed reason for force closing (min 20 characters)
    - `user_id`: Admin user ID
    
    **FSM Transition:**
    - (Any + Waiting/Responded) → (Closed + Forcibly Closed)
    
    **Use Case:** 
    - Emergency closure
    - Policy exception
    - Cancelled investigations
    
    **Errors:**
    - `400`: Validation failed (reason too short, etc.)
    - `404`: Case not found
    """
    result = admin_force_close_case(
        case_id=case_id,
        user_id=request.user_id,
        reason=request.reason
    )
    
    if not result['success']:
        raise HTTPException(
            status_code=400,
            detail=result.get('error', 'Failed to force close case')
        )
    
    return result


@router.post("/{case_id}/check-closure", response_model=Dict[str, Any], status_code=200)
def check_case_for_automatic_closure(
    case_id: int = Path(..., description="Incident case ID"),
    user_id: int = Query(..., description="User ID performing the check")
):
    """
    Check if all action items are complete and close case if so.
    
    **Path Parameter:**
    - `case_id`: IncidentRequestCaseID
    
    **Query Parameter:**
    - `user_id`: User performing the check
    
    **Business Logic:**
    - Checks if all action items are marked as done
    - If yes, automatically closes the case
    - If no, returns progress information
    
    **FSM Transition (if all complete):**
    - (In Progress + Responded) → (Closed + Responded)
    
    **Returns:**
    - can_close: Boolean indicating if case can be closed
    - case_closed: Boolean indicating if case was closed
    - Progress statistics
    
    **Errors:**
    - `404`: Case not found
    """
    result = check_and_close_case_if_complete(
        case_id=case_id,
        user_id=user_id
    )
    
    if not result['success']:
        raise HTTPException(
            status_code=400,
            detail=result.get('error', 'Failed to check case closure')
        )
    
    return result


@router.post("/{case_id}/mark-action-complete", response_model=Dict[str, Any], status_code=200)
def mark_action_item_complete_endpoint(
    case_id: int = Path(..., description="Incident case ID"),
    request: MarkActionCompleteRequest = Body(...)
):
    """
    Mark an action item as complete and check if case can be closed.
    
    **Path Parameter:**
    - `case_id`: IncidentRequestCaseID
    
    **Request Body:**
    - `action_item_id`: Action item to mark complete
    - `user_id`: User marking the item complete
    
    **Business Logic:**
    - Marks the specified action item as done
    - Automatically checks if all action items are complete
    - If yes, closes the case automatically
    
    **Returns:**
    - Action item status
    - Case closure status
    - Whether case was automatically closed
    
    **Errors:**
    - `400`: Action item not found or already complete
    - `404`: Case not found
    """
    result = mark_action_item_complete_and_check_case(
        action_item_id=request.action_item_id,
        case_id=case_id,
        user_id=request.user_id
    )
    
    if not result['success']:
        raise HTTPException(
            status_code=400,
            detail=result.get('error', 'Failed to mark action item complete')
        )
    
    return result


# ============================================================
# VALIDATION ENDPOINT
# ============================================================

@router.post("/{case_id}/validate", response_model=Dict[str, Any], status_code=200)
def validate_explanation_endpoint(
    case_id: int = Path(..., description="Incident case ID"),
    explanation_text: str = Body(..., embed=True),
    action_items: Optional[List[ActionItemCreate]] = Body(default=[], embed=True)
):
    """
    Validate an explanation submission without actually submitting it.
    
    **Path Parameter:**
    - `case_id`: IncidentRequestCaseID
    
    **Request Body:**
    - `explanation_text`: Text to validate
    - `action_items`: Optional action items to validate
    
    **Returns:**
    - valid: Boolean indicating if submission would succeed
    - errors: List of validation errors
    - warnings: List of warnings (non-blocking)
    
    **Use Case:** Frontend validation before actual submission
    """
    # Convert action items to dict format
    action_items_data = None
    if action_items:
        action_items_data = [
            {
                "title": item.title,
                "description": item.description,
                "due_date": item.due_date
            }
            for item in action_items
        ]
    
    result = validate_explanation_submission(
        case_id=case_id,
        explanation_text=explanation_text,
        action_items=action_items_data
    )
    
    return result
