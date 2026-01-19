"""
Explanation Service Layer
=========================
Business logic layer for explanation workflow.
Coordinates between DB layer and API endpoints.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# Import DB layer functions
from backend.api.db_layer.explanation_db import (
    get_case_by_id,
    get_cases_needing_explanation,
    get_red_flag_never_event_cases_needing_explanation,
    count_cases_by_explanation_status,
    get_overdue_explanations,
    check_case_has_explanation,
    update_case_explanation,
    update_case_requires_explanation,
    force_close_case,
    close_case_after_action_items,
    get_explanation_status_id,
    get_case_status_id
)

# Import action item service
from backend.api.db_layer.action_items import (
    create_action_item,
    list_action_items_for_incident,
    mark_action_item_done
)
from backend.core.database import get_connection


# -----------------------------
# QUERY/RETRIEVAL SERVICES
# -----------------------------

def get_pending_explanations(
    department_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    case_type: Optional[str] = None,
    include_red_flags_only: bool = False
) -> Dict[str, Any]:
    """
    Get all cases pending explanation with optional filters.
    
    Business logic:
    - Validates date formats
    - Applies department filtering
    - Separates red flag cases if requested
    - Returns formatted response
    
    Args:
        department_id: Filter by department
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        case_type: Filter by clinical risk type
        include_red_flags_only: If True, only return Red Flag/Never Event cases
    
    Returns:
        Dictionary with cases list and metadata
    """
    try:
        # Parse dates if provided
        parsed_start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        parsed_end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        
        # Get cases based on filter
        if include_red_flags_only:
            cases = get_red_flag_never_event_cases_needing_explanation(
                department_id=department_id,
                start_date=parsed_start,
                end_date=parsed_end
            )
        else:
            cases = get_cases_needing_explanation(
                department_id=department_id,
                start_date=parsed_start,
                end_date=parsed_end,
                case_type=case_type
            )
        
        # Calculate statistics
        total_count = len(cases)
        red_flag_count = sum(1 for c in cases if c.get('ClinicalRiskType') in ['Red Flag', 'Never Event'])
        ordinary_count = total_count - red_flag_count
        
        return {
            "success": True,
            "total_count": total_count,
            "red_flag_count": red_flag_count,
            "ordinary_count": ordinary_count,
            "cases": cases,
            "filters": {
                "department_id": department_id,
                "start_date": start_date,
                "end_date": end_date,
                "case_type": case_type,
                "red_flags_only": include_red_flags_only
            }
        }
        
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid date format: {str(e)}",
            "cases": []
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "cases": []
        }


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


def get_case_explanation_details(case_id: int) -> Dict[str, Any]:
    """
    Get detailed information about a case for explanation submission.
    
    Args:
        case_id: Incident case ID
    
    Returns:
        Dictionary with case details and validation status
    """
    try:
        case = get_case_by_id(case_id)
        
        if not case:
            return {
                "success": False,
                "error": f"Case {case_id} not found"
            }
        
        # Check if case can receive explanation
        # Red Flag (2) or Never Event (3) ALWAYS need explanation, regardless of RequiresExplanation flag
        is_red_flag_or_never_event = case.get('ClinicalRiskTypeID') in (2, 3)
        requires_explanation = (
            is_red_flag_or_never_event or 
            case.get('RequiresExplanation') == 1
        )
        
        can_submit = (
            requires_explanation and
            case.get('CaseStatusName') != 'Closed' and
            case.get('ExplanationStatusName') == 'Waiting'
        )
        
        has_explanation = check_case_has_explanation(case_id)
        
        return {
            "success": True,
            "case": case,
            "validation": {
                "can_submit_explanation": can_submit,
                "has_existing_explanation": has_explanation,
                "requires_explanation": requires_explanation,
                "is_red_flag_or_never_event": is_red_flag_or_never_event,
                "is_closed": case.get('CaseStatusName') == 'Closed',
                "current_status": case.get('ExplanationStatusName')
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# -----------------------------
# WRITE/UPDATE SERVICES
# -----------------------------

def submit_explanation(
    case_id: int,
    explanation_text: str,
    action_items: Optional[List[Dict[str, Any]]] = None,
    user_id: int = 1
) -> Dict[str, Any]:
    """
    Submit explanation for a case with optional action items.
    
    Business logic:
    - Validates explanation text
    - Validates case can receive explanation
    - Submits explanation (triggers FSM transition)
    - Creates action items if provided
    
    Args:
        case_id: Incident case ID
        explanation_text: Explanation text content
        action_items: Optional list of action items to create
        user_id: ID of user submitting
    
    Returns:
        Dictionary with success status and created resources
    """
    try:
        # Validate explanation text
        if not explanation_text or len(explanation_text.strip()) < 10:
            return {
                "success": False,
                "error": "Explanation text must be at least 10 characters"
            }
        
        # Validate case state
        validation = get_case_explanation_details(case_id)
        if not validation['success']:
            return validation
        
        if not validation['validation']['can_submit_explanation']:
            reasons = []
            if not validation['validation']['requires_explanation']:
                reasons.append("case does not require explanation")
            if validation['validation']['is_closed']:
                reasons.append("case is closed")
            if validation['validation']['current_status'] != 'Waiting':
                reasons.append(f"explanation status is '{validation['validation']['current_status']}'")
            
            return {
                "success": False,
                "error": f"Cannot submit explanation: {', '.join(reasons)}"
            }
        
        # Submit explanation (triggers FSM transition)
        result = update_case_explanation(case_id, explanation_text, user_id)
        
        # Create action items if provided
        created_action_items = []
        if action_items:
            for item in action_items:
                try:
                    action_id = create_action_item(
                        action_title=item.get('title'),
                        action_description=item.get('description', ''),
                        due_date=item.get('due_date'),
                        created_by_user_id=user_id,
                        incident_case_id=case_id
                    )
                    created_action_items.append({
                        "action_item_id": action_id,
                        "title": item.get('title')
                    })
                except Exception as e:
                    # Log error but don't fail entire submission
                    created_action_items.append({
                        "error": f"Failed to create action item: {str(e)}",
                        "title": item.get('title')
                    })
        
        return {
            "success": True,
            "message": "Explanation submitted successfully",
            "case_id": case_id,
            "case": result.get('case'),
            "action_items_created": len(created_action_items),
            "action_items": created_action_items
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def toggle_requires_explanation(
    case_id: int,
    requires_explanation: bool,
    user_id: int,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Toggle the RequiresExplanation flag for a case (admin function).
    
    Args:
        case_id: Incident case ID
        requires_explanation: New value for flag
        user_id: ID of admin user
        reason: Optional reason for change
    
    Returns:
        Dictionary with success status
    """
    try:
        result = update_case_requires_explanation(
            case_id,
            requires_explanation,
            user_id
        )
        
        if reason:
            result['reason'] = reason
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def admin_force_close_case(
    case_id: int,
    user_id: int,
    reason: str
) -> Dict[str, Any]:
    """
    Force close a case (admin override).
    
    Business logic:
    - Validates user has admin permissions (placeholder)
    - Requires a reason
    - Forces case closure regardless of state
    
    Args:
        case_id: Incident case ID
        user_id: ID of admin user
        reason: Reason for force closure (required)
    
    Returns:
        Dictionary with success status
    """
    try:
        # Validate reason provided
        if not reason or len(reason.strip()) < 5:
            return {
                "success": False,
                "error": "Reason for force closure must be provided (min 5 characters)"
            }
        
        # TODO: Add permission check here
        # if not user_has_admin_permission(user_id):
        #     return {"success": False, "error": "Insufficient permissions"}
        
        result = force_close_case(case_id, user_id, reason)
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def complete_case_with_action_items(
    case_id: int,
    user_id: int
) -> Dict[str, Any]:
    """
    Close a case after all action items are completed.
    
    Business logic:
    - Validates all action items are done
    - Validates case is in correct state
    - Closes case with Responded status
    
    Args:
        case_id: Incident case ID
        user_id: ID of user closing case
    
    Returns:
        Dictionary with success status
    """
    try:
        result = close_case_after_action_items(case_id, user_id)
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# -----------------------------
# VALIDATION SERVICES
# -----------------------------

def validate_explanation_submission(
    case_id: int,
    explanation_text: str,
    action_items: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Validate explanation submission without actually submitting.
    
    Args:
        case_id: Incident case ID
        explanation_text: Explanation text to validate
        action_items: Optional action items to validate
    
    Returns:
        Dictionary with validation results and errors
    """
    errors = []
    warnings = []
    
    try:
        # Validate explanation text
        if not explanation_text or len(explanation_text.strip()) < 10:
            errors.append("Explanation text must be at least 10 characters")
        
        if explanation_text and len(explanation_text) > 5000:
            warnings.append("Explanation text is very long (>5000 characters)")
        
        # Validate case state
        validation = get_case_explanation_details(case_id)
        if not validation['success']:
            errors.append(validation.get('error', 'Case not found'))
        elif not validation['validation']['can_submit_explanation']:
            if not validation['validation']['requires_explanation']:
                errors.append("Case does not require explanation")
            if validation['validation']['is_closed']:
                errors.append("Case is closed")
            if validation['validation']['has_existing_explanation']:
                warnings.append("Case already has an explanation")
        
        # Validate action items
        if action_items:
            if not isinstance(action_items, list):
                errors.append("Action items must be a list")
            else:
                for i, item in enumerate(action_items):
                    if not item.get('title'):
                        errors.append(f"Action item {i+1}: Title is required")
                    if item.get('due_date'):
                        try:
                            datetime.strptime(item['due_date'], "%Y-%m-%d")
                        except ValueError:
                            errors.append(f"Action item {i+1}: Invalid due date format (use YYYY-MM-DD)")
        
        is_valid = len(errors) == 0
        
        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "can_submit": is_valid
        }
        
    except Exception as e:
        return {
            "valid": False,
            "errors": [str(e)],
            "warnings": [],
            "can_submit": False
        }


# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------

def get_explanation_history(case_id: int) -> Dict[str, Any]:
    """
    Get explanation history for a case (placeholder for future audit trail).
    
    Args:
        case_id: Incident case ID
    
    Returns:
        Dictionary with history records
    """
    # TODO: Implement audit trail query when history table is available
    try:
        case = get_case_by_id(case_id)
        
        if not case:
            return {
                "success": False,
                "error": "Case not found"
            }
        
        # For now, return current state only
        return {
            "success": True,
            "case_id": case_id,
            "current_explanation": case.get('TakenAction'),
            "current_status": case.get('ExplanationStatusName'),
            "history": []  # Will be populated when audit table exists
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# -----------------------------
# ACTION ITEM INTEGRATION
# -----------------------------

def check_and_close_case_if_complete(case_id: int, user_id: int) -> Dict[str, Any]:
    """
    Check if all action items for a case are complete, and close the case if so.
    
    Business rules:
    - Only applies to cases with ExplanationStatus = 'Responded' (2)
    - Case must have at least one action item
    - All action items must be marked as done (IsDone = 1)
    - Triggers FSM transition: (In Progress + Responded) -> (Closed + Responded)
    
    Args:
        case_id: Incident case ID
        user_id: User performing the check
    
    Returns:
        Dictionary with success status and case state
    """
    try:
        # Get case details
        case = get_case_by_id(case_id)
        if not case:
            return {
                "success": False,
                "error": "Case not found"
            }
        
        # Check if case has responded explanation status
        if case.get('ExplanationStatusID') != 2:  # Not Responded
            return {
                "success": False,
                "error": f"Case explanation status is not 'Responded' (current: {case.get('ExplanationStatusName')})",
                "can_close": False
            }
        
        # Get all action items for this case
        action_items = list_action_items_for_incident(case_id)
        
        if not action_items:
            return {
                "success": False,
                "error": "Case has no action items",
                "can_close": False
            }
        
        # Check if all are complete
        incomplete_items = [item for item in action_items if not item.get('IsDone')]
        
        if incomplete_items:
            return {
                "success": True,
                "can_close": False,
                "message": f"{len(incomplete_items)} action item(s) still incomplete",
                "total_items": len(action_items),
                "complete_items": len(action_items) - len(incomplete_items),
                "incomplete_items": len(incomplete_items)
            }
        
        # All action items complete - close the case
        result = close_case_after_action_items(case_id, user_id)
        
        return {
            "success": True,
            "can_close": True,
            "case_closed": True,
            "message": "All action items complete - case closed",
            "total_items": len(action_items),
            "case": result.get('case')
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def mark_action_item_complete_and_check_case(
    action_item_id: int,
    case_id: int,
    user_id: int
) -> Dict[str, Any]:
    """
    Mark an action item as complete and automatically check if case can be closed.
    
    This is a convenience function that combines:
    1. Marking the action item as done
    2. Checking if all action items are complete
    3. Closing the case if appropriate
    
    Args:
        action_item_id: Action item to mark complete
        case_id: Associated case ID
        user_id: User marking the item complete
    
    Returns:
        Dictionary with action item and case status
    """
    try:
        # Mark action item as done
        mark_action_item_done(action_item_id)
        
        # Check if case can be closed
        close_result = check_and_close_case_if_complete(case_id, user_id)
        
        return {
            "success": True,
            "action_item_id": action_item_id,
            "marked_complete": True,
            "case_status": close_result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_case_completion_status(case_id: int) -> Dict[str, Any]:
    """
    Get the completion status of a case including action items progress.
    
    Args:
        case_id: Incident case ID
    
    Returns:
        Dictionary with case and action items status
    """
    try:
        case = get_case_by_id(case_id)
        if not case:
            return {
                "success": False,
                "error": "Case not found"
            }
        
        action_items = list_action_items_for_incident(case_id)
        
        if not action_items:
            return {
                "success": True,
                "case_id": case_id,
                "has_action_items": False,
                "explanation_status": case.get('ExplanationStatusName'),
                "case_status": case.get('CaseStatusName')
            }
        
        complete_count = sum(1 for item in action_items if item.get('IsDone'))
        total_count = len(action_items)
        
        return {
            "success": True,
            "case_id": case_id,
            "has_action_items": True,
            "total_action_items": total_count,
            "complete_action_items": complete_count,
            "incomplete_action_items": total_count - complete_count,
            "all_complete": complete_count == total_count,
            "completion_percentage": (complete_count / total_count * 100) if total_count > 0 else 0,
            "explanation_status": case.get('ExplanationStatusName'),
            "case_status": case.get('CaseStatusName'),
            "can_close": (
                case.get('ExplanationStatusID') == 2 and  # Responded
                complete_count == total_count and
                total_count > 0
            )
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
