"""
Service Layer: Three-Type Explanation System
=============================================
Refactored explanation service to handle three distinct types:
1. Red Flag/Never Event → Create APP_IncidentCaseFeedback record
2. Ordinary → Update TakenAction in APP_IncidentCase  
3. Seasonal → Update ExplanationText in APP_SeasonalOrgUnitReport

Each type has different database operations and UI requirements.
"""

from typing import Dict, Any, List, Optional
from api.db_layer.explanation_red_flag_db import (
    create_red_flag_feedback,
    get_red_flag_feedback_details
)
from api.db_layer.explanation_ordinary_db import (
    update_ordinary_explanation,
    get_ordinary_explanation
)
from api.db_layer.explanation_seasonal_db import (
    update_seasonal_explanation,
    get_seasonal_explanation,
    get_seasonal_reports_needing_explanation
)
from api.db_layer.explanation_db import (
    get_cases_needing_explanation,
    get_red_flag_never_event_cases_needing_explanation,
    count_cases_by_explanation_status,
    get_overdue_explanations
)


# ============================================================
# RED FLAG / NEVER EVENT SERVICES
# ============================================================

def submit_red_flag_explanation(
    incident_request_case_id: int,
    explanation_text: str,
    causes: Dict[str, Any],
    preventive_actions: Dict[str, Any],
    user_id: int
) -> Dict[str, Any]:
    """
    Submit comprehensive feedback for Red Flag/Never Event case.
    
    Creates new record in APP_IncidentCaseFeedback with:
    - Root cause analysis (Staff, Process, Equipment, Environment)
    - Preventive action plans
    - Department explanation text
    
    FSM: S0 → S1 (Open + Waiting → In Progress + Responded)
    """
    result = create_red_flag_feedback(
        incident_request_case_id=incident_request_case_id,
        explanation_text=explanation_text,
        causes=causes,
        preventive_actions=preventive_actions,
        user_id=user_id
    )
    
    return result


def get_red_flag_explanation_details(incident_request_case_id: int) -> Dict[str, Any]:
    """
    Retrieve existing feedback for Red Flag/Never Event case.
    """
    feedback = get_red_flag_feedback_details(incident_request_case_id)
    
    if not feedback:
        return {
            "success": False,
            "error": "FEEDBACK_NOT_FOUND",
            "message": "No feedback found for this case"
        }
    
    return {
        "success": True,
        "feedback": feedback
    }


# ============================================================
# ORDINARY CASE SERVICES
# ============================================================

def submit_ordinary_explanation(
    incident_request_case_id: int,
    explanation_text: str,
    user_id: int
) -> Dict[str, Any]:
    """
    Submit simple explanation for Ordinary case.
    
    Appends explanation text to TakenAction field in APP_IncidentCase.
    
    FSM: S0 → S1 (Open + Waiting → In Progress + Responded)
    """
    result = update_ordinary_explanation(
        incident_request_case_id=incident_request_case_id,
        explanation_text=explanation_text,
        user_id=user_id
    )
    
    return result


def get_ordinary_explanation_details(incident_request_case_id: int) -> Dict[str, Any]:
    """
    Retrieve TakenAction field for Ordinary case.
    """
    result = get_ordinary_explanation(incident_request_case_id)
    return result


# ============================================================
# SEASONAL REPORT SERVICES
# ============================================================

def submit_seasonal_explanation(
    seasonal_report_id: int,
    explanation_text: str,
    user_id: int
) -> Dict[str, Any]:
    """
    Submit explanation for seasonal report.
    
    Updates ExplanationText field in APP_SeasonalOrgUnitReport.
    """
    result = update_seasonal_explanation(
        seasonal_report_id=seasonal_report_id,
        explanation_text=explanation_text,
        user_id=user_id
    )
    
    return result


def get_seasonal_explanation_details(seasonal_report_id: int) -> Dict[str, Any]:
    """
    Retrieve explanation for seasonal report.
    """
    result = get_seasonal_explanation(seasonal_report_id)
    return result


def get_pending_seasonal_explanations(
    org_unit_id: Optional[int] = None,
    season_id: Optional[int] = None,
    non_compliant_only: bool = False
) -> Dict[str, Any]:
    """
    Get all seasonal reports needing explanation.
    """
    result = get_seasonal_reports_needing_explanation(
        org_unit_id=org_unit_id,
        season_id=season_id,
        non_compliant_only=non_compliant_only
    )
    
    return result


# ============================================================
# UNIFIED CASE LISTING (For Dashboard)
# ============================================================

def get_pending_case_explanations(
    dept_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    case_type: Optional[str] = None,
    include_red_flags_only: bool = False
) -> Dict[str, Any]:
    """
    Get all cases (Red Flag, Never Event, Ordinary) needing explanation.
    
    Returns unified list for dashboard with case type indicators.
    """
    try:
        # DB functions return raw lists, not dicts
        if include_red_flags_only:
            # Only Red Flag/Never Event cases
            cases = get_red_flag_never_event_cases_needing_explanation(
                department_id=dept_id,
                start_date=start_date,
                end_date=end_date
            )
        else:
            # All cases needing explanation
            cases = get_cases_needing_explanation(
                department_id=dept_id,
                start_date=start_date,
                end_date=end_date,
                case_type=case_type
            )
        
        # Add type indicators for UI
        for case in cases:
            clinical_risk_type_id = case.get("ClinicalRiskTypeID")
            
            if clinical_risk_type_id == 2:
                case["explanation_type"] = "red_flag"
                case["explanation_endpoint"] = f"/api/explanations/red-flag/{case['IncidentRequestCaseID']}"
            elif clinical_risk_type_id == 3:
                case["explanation_type"] = "never_event"
                case["explanation_endpoint"] = f"/api/explanations/red-flag/{case['IncidentRequestCaseID']}"
            elif clinical_risk_type_id == 1 and case.get("RequiresExplanation"):
                case["explanation_type"] = "ordinary"
                case["explanation_endpoint"] = f"/api/explanations/ordinary/{case['IncidentRequestCaseID']}"
            else:
                case["explanation_type"] = "none"
                case["explanation_endpoint"] = None
        
        # Calculate statistics
        total_count = len(cases)
        red_flag_count = sum(1 for c in cases if c.get("ClinicalRiskTypeID") in [2, 3])
        ordinary_count = total_count - red_flag_count
        
        return {
            "success": True,
            "data": cases,
            "statistics": {
                "total_count": total_count,
                "red_flag_count": red_flag_count,
                "ordinary_count": ordinary_count
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": []
        }


# ============================================================
# DASHBOARD STATISTICS
# ============================================================

def get_explanation_dashboard_statistics() -> Dict[str, Any]:
    """
    Get dashboard statistics for explanation workflow.
    
    Returns:
        Dictionary with counts, overdue cases, and trends
    """
    try:
        # Get counts by status
        status_counts = count_cases_by_explanation_status()
        
        # Get overdue cases (>7 days)
        overdue_7days = get_overdue_explanations(days_threshold=7)
        overdue_30days = get_overdue_explanations(days_threshold=30)
        
        # Format response
        status_dict = {
            item['StatusName']: item['CaseCount'] 
            for item in status_counts
        }
        
        return {
            "success": True,
            "statistics": {
                "by_status": status_dict,
                "overdue": {
                    "over_7_days": len(overdue_7days),
                    "over_30_days": len(overdue_30days),
                    "most_overdue": overdue_7days[:5] if overdue_7days else []
                },
                "totals": {
                    "awaiting_explanation": status_dict.get('Waiting', 0),
                    "responded": status_dict.get('Responded', 0),
                    "forcibly_closed": status_dict.get('Forcibly Closed', 0),
                    "no_explanation_needed": status_dict.get('No Explanation Needed', 0)
                }
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
