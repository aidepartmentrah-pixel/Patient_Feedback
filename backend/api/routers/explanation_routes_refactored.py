"""
Explanation Routes - Refactored Three-Type System
==================================================
Separate API endpoints for three distinct explanation types:

1. RED FLAG / NEVER EVENT
   - POST /api/explanations/red-flag/{case_id}
   - Creates new record in APP_IncidentCaseFeedback
   - Requires comprehensive root cause analysis + preventive actions

2. ORDINARY CASES  
   - POST /api/explanations/ordinary/{case_id}
   - Updates TakenAction field in APP_IncidentCase
   - Simple text explanation only

3. SEASONAL REPORTS
   - POST /api/explanations/seasonal/{report_id}
   - Updates ExplanationText in APP_SeasonalOrgUnitReport
   - Report-level explanation

UNIFIED ENDPOINTS:
   - GET /api/explanations/pending/cases - All cases needing explanation
   - GET /api/explanations/pending/seasonal - Seasonal reports needing explanation
"""

from typing import List, Dict, Any, Optional
from datetime import date
from fastapi import APIRouter, HTTPException, Path, Query, Body
from pydantic import BaseModel, Field

from api.services.explanation_service_refactored import (
    submit_red_flag_explanation,
    get_red_flag_explanation_details,
    submit_ordinary_explanation,
    get_ordinary_explanation_details,
    submit_seasonal_explanation,
    get_seasonal_explanation_details,
    get_pending_case_explanations,
    get_pending_seasonal_explanations,
    get_explanation_dashboard_statistics
)

from api.services.action_item_service import create_action_item


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

# --- Action Item Model (Shared across all explanation types) ---

class ActionItemCreate(BaseModel):
    """Action item to be created with explanation"""
    action_title: str = Field(..., min_length=3, max_length=200, description="Action item title")
    action_description: Optional[str] = Field(None, max_length=1000, description="Detailed description")
    due_date: Optional[date] = Field(None, description="Due date for completion (YYYY-MM-DD)")


# --- Red Flag / Never Event Models ---

class CauseStaff(BaseModel):
    """Staff-related causes"""
    training: bool = Field(default=False)
    incentives: bool = Field(default=False)
    competency: bool = Field(default=False)
    understaffed: bool = Field(default=False)
    non_compliance: bool = Field(default=False)
    no_coordination: bool = Field(default=False)
    other: bool = Field(default=False)
    other_text: Optional[str] = Field(default=None, max_length=1000)


class CauseProcess(BaseModel):
    """Process-related causes"""
    not_comprehensive: bool = Field(default=False)
    unclear: bool = Field(default=False)
    missing_protocol: bool = Field(default=False)
    other: bool = Field(default=False)
    other_text: Optional[str] = Field(default=None, max_length=1000)


class CauseEquipment(BaseModel):
    """Equipment-related causes"""
    not_available: bool = Field(default=False)
    system_incomplete: bool = Field(default=False)
    hard_to_apply: bool = Field(default=False)
    other: bool = Field(default=False)
    other_text: Optional[str] = Field(default=None, max_length=1000)


class CauseEnvironment(BaseModel):
    """Environment-related causes"""
    place_nature: bool = Field(default=False)
    surroundings: bool = Field(default=False)
    work_conditions: bool = Field(default=False)
    other: bool = Field(default=False)
    other_text: Optional[str] = Field(default=None, max_length=1000)


class PreventiveActions(BaseModel):
    """Preventive actions"""
    monthly_meetings: bool = Field(default=False)
    training_programs: bool = Field(default=False)
    increase_staff: bool = Field(default=False)
    mm_committee_actions: bool = Field(default=False)
    other: bool = Field(default=False)
    other_text: Optional[str] = Field(default=None, max_length=1000)


class RedFlagExplanationRequest(BaseModel):
    """Comprehensive feedback for Red Flag/Never Event"""
    explanation_text: str = Field(..., min_length=50, max_length=5000, description="Detailed explanation")
    causes_staff: CauseStaff = Field(default_factory=CauseStaff)
    causes_process: CauseProcess = Field(default_factory=CauseProcess)
    causes_equipment: CauseEquipment = Field(default_factory=CauseEquipment)
    causes_environment: CauseEnvironment = Field(default_factory=CauseEnvironment)
    preventive_actions: PreventiveActions = Field(default_factory=PreventiveActions)
    action_items: List[ActionItemCreate] = Field(default=[], description="Action items to create")
    user_id: int = Field(..., description="ID of user submitting feedback")


# --- Ordinary Case Models ---

class OrdinaryExplanationRequest(BaseModel):
    """Simple explanation for ordinary cases"""
    explanation_text: str = Field(..., min_length=20, max_length=2000, description="Explanation text")
    action_items: List[ActionItemCreate] = Field(default=[], description="Action items to create")
    user_id: int = Field(..., description="ID of user submitting explanation")


# --- Seasonal Report Models ---

class SeasonalExplanationRequest(BaseModel):
    """Explanation for seasonal report"""
    action_items: List[ActionItemCreate] = Field(default=[], description="Action items to create")
    explanation_text: str = Field(..., min_length=50, max_length=5000, description="Explanation for report")
    user_id: int = Field(..., description="ID of user submitting explanation")


# ============================================================
# ROUTER
# ============================================================
router = APIRouter(prefix="/api/explanations", tags=["Explanations"])


# ============================================================
# RED FLAG / NEVER EVENT ENDPOINTS
# ============================================================

@router.post("/red-flag/{case_id}", response_model=Dict[str, Any], status_code=200)
def submit_red_flag_explanation_endpoint(
    case_id: int = Path(..., description="Incident Request Case ID"),
    request: RedFlagExplanationRequest = Body(...)
):
    """
    Submit comprehensive feedback for Red Flag or Never Event case.
    
    **Creates NEW record in APP_IncidentCaseFeedback table**
    
    **FSM Transition:** S0 → S1 (Open + Waiting → In Progress + Responded)
    
    **Required for:**
    - ClinicalRiskTypeID = 2 (Red Flag)
    - ClinicalRiskTypeID = 3 (Never Event)
    
    **Request Body:**
    - `explanation_text`: Detailed explanation (min 50 chars)
    - `causes_staff`: Staff-related causes (training, competency, etc.)
    - `causes_process`: Process-related causes (protocols, procedures)
    - `causes_equipment`: Equipment-related causes (availability, completeness)
    - `causes_environment`: Environment-related causes (workplace conditions)
    - `preventive_actions`: Preventive measures (training programs, meetings, etc.)
    - `user_id`: ID of user submitting feedback
    
    **Returns:**
    - `success`: Boolean
    - `message`: Result message
    - `feedback_created`: True if new record created
    - `fsm_transition`: State transition description
    """
    causes = {
        "staff": request.causes_staff.dict(),
        "process": request.causes_process.dict(),
        "equipment": request.causes_equipment.dict(),
        "environment": request.causes_environment.dict()
    }
    
    preventive_actions = request.preventive_actions.dict()
    
    result = submit_red_flag_explanation(
        incident_request_case_id=case_id,
        explanation_text=request.explanation_text,
        causes=causes,
        preventive_actions=preventive_actions,
        user_id=request.user_id
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    
    # Create action items if provided
    action_items_created = []
    if request.action_items:
        for action_item in request.action_items:
            try:
                action_item_id = create_action_item(
                    action_title=action_item.action_title,
                    action_description=action_item.action_description,
                    due_date=action_item.due_date,
                    created_by_user_id=request.user_id,
                    incident_case_id=case_id,
                    seasonal_report_id=None,
                    season_case_id=None
                )
                action_items_created.append({
                    "action_item_id": action_item_id,
                    "title": action_item.action_title
                })
            except Exception as e:
                # Log error but don't fail the whole request
                print(f"[ERROR] Failed to create action item: {str(e)}")
    
    result["action_items_created"] = action_items_created
    result["action_items_count"] = len(action_items_created)
    
    return result


@router.get("/red-flag/{case_id}", response_model=Dict[str, Any], status_code=200)
def get_red_flag_explanation_endpoint(
    case_id: int = Path(..., description="Incident Request Case ID")
):
    """
    Retrieve existing feedback for Red Flag/Never Event case.
    
    **Returns feedback from APP_IncidentCaseFeedback table**
    """
    result = get_red_flag_explanation_details(case_id)
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    
    return result


# ============================================================
# ORDINARY CASE ENDPOINTS
# ============================================================

@router.post("/ordinary/{case_id}", response_model=Dict[str, Any], status_code=200)
def submit_ordinary_explanation_endpoint(
    case_id: int = Path(..., description="Incident Request Case ID"),
    request: OrdinaryExplanationRequest = Body(...)
):
    """
    Submit simple explanation for Ordinary case.
    
    **Updates TakenAction field in APP_IncidentCase table**
    
    **FSM Transition:** S0 → S1 (Open + Waiting → In Progress + Responded)
    
    **Required for:**
    - ClinicalRiskTypeID = 1 (Ordinary)
    - RequiresExplanation = 1 (True)
    
    **Request Body:**
    - `explanation_text`: Explanation text (min 20 chars)
    - `user_id`: ID of user submitting explanation
    
    **Returns:**
    - `success`: Boolean
    - `message`: Result message
    - `updated_field`: "TakenAction"
    - `fsm_transition`: State transition description
    """
    result = submit_ordinary_explanation(
        incident_request_case_id=case_id,
        explanation_text=request.explanation_text,
        user_id=request.user_id
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    
    # Create action items if provided
    action_items_created = []
    if request.action_items:
        for action_item in request.action_items:
            try:
                action_item_id = create_action_item(
                    action_title=action_item.action_title,
                    action_description=action_item.action_description,
                    due_date=action_item.due_date,
                    created_by_user_id=request.user_id,
                    incident_case_id=case_id,
                    seasonal_report_id=None,
                    season_case_id=None
                )
                action_items_created.append({
                    "action_item_id": action_item_id,
                    "title": action_item.action_title
                })
            except Exception as e:
                print(f"[ERROR] Failed to create action item: {str(e)}")
    
    result["action_items_created"] = action_items_created
    result["action_items_count"] = len(action_items_created)
    
    return result


@router.get("/ordinary/{case_id}", response_model=Dict[str, Any], status_code=200)
def get_ordinary_explanation_endpoint(
    case_id: int = Path(..., description="Incident Request Case ID")
):
    """
    Retrieve TakenAction field for Ordinary case.
    
    **Returns data from APP_IncidentCase table**
    """
    result = get_ordinary_explanation_details(case_id)
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    
    return result


# ============================================================
# SEASONAL REPORT ENDPOINTS
# ============================================================

@router.post("/seasonal/{report_id}", response_model=Dict[str, Any], status_code=200)
def submit_seasonal_explanation_endpoint(
    report_id: int = Path(..., description="Seasonal Report ID"),
    request: SeasonalExplanationRequest = Body(...)
):
    """
    Submit explanation for seasonal report.
    
    **Updates ExplanationText field in APP_SeasonalOrgUnitReport table**
    
    **Request Body:**
    - `explanation_text`: Explanation for report (min 50 chars)
    - `user_id`: ID of user submitting explanation
    
    **Returns:**
    - `success`: Boolean
    - `message`: Result message
    - `seasonal_report_id`: Report ID
    - `updated_field`: "ExplanationText"
    """
    result = submit_seasonal_explanation(
        seasonal_report_id=report_id,
        explanation_text=request.explanation_text,
        user_id=request.user_id
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    
    # Create action items if provided
    action_items_created = []
    if request.action_items:
        for action_item in request.action_items:
            try:
                action_item_id = create_action_item(
                    action_title=action_item.action_title,
                    action_description=action_item.action_description,
                    due_date=action_item.due_date,
                    created_by_user_id=request.user_id,
                    incident_case_id=None,
                    seasonal_report_id=report_id,
                    season_case_id=None
                )
                action_items_created.append({
                    "action_item_id": action_item_id,
                    "title": action_item.action_title
                })
            except Exception as e:
                print(f"[ERROR] Failed to create action item: {str(e)}")
    
    result["action_items_created"] = action_items_created
    result["action_items_count"] = len(action_items_created)
    
    return result


@router.get("/seasonal/{report_id}", response_model=Dict[str, Any], status_code=200)
def get_seasonal_explanation_endpoint(
    report_id: int = Path(..., description="Seasonal Report ID")
):
    """
    Retrieve explanation for seasonal report.
    
    **Returns data from APP_SeasonalOrgUnitReport table**
    """
    result = get_seasonal_explanation_details(report_id)
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    
    return result


# ============================================================
# UNIFIED LISTING ENDPOINTS (For Dashboard)
# ============================================================

@router.get("/pending/cases", response_model=Dict[str, Any], status_code=200)
def get_pending_cases_endpoint(
    dept_id: Optional[int] = Query(None, description="Filter by department ID"),
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    case_type: Optional[str] = Query(None, description="Filter by case type"),
    include_red_flags_only: bool = Query(False, description="Only Red Flag/Never Event cases")
):
    """
    Get all cases (Red Flag, Never Event, Ordinary) needing explanation.
    
    **Returns unified list from APP_IncidentCase with type indicators**
    
    **Query Parameters:**
    - `dept_id`: Filter by target department
    - `start_date`: Filter cases from this date onward (YYYY-MM-DD)
    - `end_date`: Filter cases up to this date (YYYY-MM-DD)
    - `case_type`: Filter by case type
    - `include_red_flags_only`: If true, only return Red Flag/Never Event
    
    **Returns:**
    ```json
    {
        "success": true,
        "data": [
            {
                "incident_request_case_id": 123,
                "clinical_risk_type_id": 2,
                "explanation_type": "red_flag",
                "explanation_endpoint": "/api/explanations/red-flag/123",
                ...
            }
        ],
        "statistics": {
            "total_count": 10,
            "red_flag_count": 3,
            "ordinary_count": 7
        }
    }
    ```
    
    **Explanation Types:**
    - `red_flag`: Use POST /api/explanations/red-flag/{case_id}
    - `never_event`: Use POST /api/explanations/red-flag/{case_id}
    - `ordinary`: Use POST /api/explanations/ordinary/{case_id}
    """
    result = get_pending_case_explanations(
        dept_id=dept_id,
        start_date=start_date,
        end_date=end_date,
        case_type=case_type,
        include_red_flags_only=include_red_flags_only
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result)
    
    return result


@router.get("/pending/seasonal", response_model=Dict[str, Any], status_code=200)
def get_pending_seasonal_endpoint(
    org_unit_id: Optional[int] = Query(None, description="Filter by organization unit"),
    season_id: Optional[int] = Query(None, description="Filter by season"),
    non_compliant_only: bool = Query(False, description="Only non-compliant reports")
):
    """
    Get all seasonal reports needing explanation.
    
    **Returns data from APP_SeasonalOrgUnitReport**
    
    **Query Parameters:**
    - `org_unit_id`: Filter by organization unit
    - `season_id`: Filter by season
    - `non_compliant_only`: If true, only return non-compliant reports
    
    **Returns:**
    ```json
    {
        "success": true,
        "data": [
            {
                "seasonal_report_id": 456,
                "season_name": "Q1 2026",
                "org_unit_id": 10,
                "is_compliant": false,
                "violated_rules": "Rule A, Rule B",
                ...
            }
        ],
        "statistics": {
            "total_count": 5,
            "non_compliant_count": 2
        }
    }
    ```
    """
    result = get_pending_seasonal_explanations(
        org_unit_id=org_unit_id,
        season_id=season_id,
        non_compliant_only=non_compliant_only
    )
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result)
    
    return result


# ============================================================
# STATISTICS ENDPOINT
# ============================================================

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
    
    if not result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=result.get('error', 'Failed to retrieve statistics')
        )
    
    return result
