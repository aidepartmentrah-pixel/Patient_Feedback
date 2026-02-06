"""
Workflow Router (API v2)
Unified workflow management endpoints for Phase 3.5.

This router will provide:
- Inbox endpoints (STEP 3.5.2)
- Follow-up action endpoints (to be added in STEP 3.5.3)
- Case action endpoints (to be added in STEP 3.5.4)

Security: All endpoints protected by role guards and scope enforcement.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from backend.api.dependencies.user_context import get_current_user
from backend.api.schemas.auth_models import CurrentUser
from backend.api_v2.services import inbox_service, follow_up_service, case_response_service
from backend.api_v2.guards.high_level_guards import (
    require_section_admin_on_subcase,
    require_dept_admin_on_subcase,
    require_admin_on_subcase
)


# ============================================================
# ROUTER DEFINITION
# ============================================================
router = APIRouter(prefix="/api/v2/workflow", tags=["Workflow v2"])


# ============================================================
# INBOX ENDPOINTS (STEP 3.5.2)
# ============================================================

@router.get("/inbox")
def get_inbox(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get inbox for current user (role-aware).
    
    Returns subcases that require action from the current user based on:
    - User's role (Section/Department/Administration Administrator)
    - User's organizational scope (Phase 2.5 scope engine)
    - Current subcase status in the workflow
    
    Each inbox item contains:
    - subcase_id: Unique identifier
    - case_type: 'INCIDENT' or 'SEASONAL_REPORT'
    - incident_id: Source incident ID (if applicable)
    - seasonal_report_id: Source seasonal report ID (if applicable)
    - target_org_unit_id: Target organizational unit
    - status: Current workflow status
    - created_at: Creation timestamp
    - allowed_actions: List of actions user can perform
    
    Security: Requires authentication. Role and scope filtering applied by service.
    """
    items = inbox_service.get_inbox(current_user)
    return {"items": items}


# ============================================================
# FOLLOW-UP ENDPOINTS (STEP 3.5.3)
# ============================================================

@router.get("/follow-up")
def get_follow_up_items(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get follow-up action items for current user.
    
    Returns action items that:
    - Are assigned to the user, OR
    - User has privileged role to view
    - Are within user's organizational scope (Phase 2.5)
    
    Each action item contains:
    - action_item_id: Unique identifier
    - subcase_id: Parent subcase
    - title: Action item title
    - description: Action item description
    - assigned_to_user_id: Assigned user
    - due_date: Due date
    - status: Current status
    - created_at: Creation timestamp
    
    Security: Requires authentication. Scope filtering applied by service.
    """
    items = follow_up_service.get_action_items_for_user(current_user)
    return {"items": items}


@router.post("/follow-up/{action_item_id}/start")
def start_action_item(
    action_item_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, bool]:
    """
    Start working on an action item.
    
    Marks the action item as "in progress" if the user has permission.
    
    Security:
    - Requires authentication
    - User must be assigned to the action item OR have privileged role
    - Action item must be within user's organizational scope
    - Returns 403 if unauthorized (from service layer)
    """
    success = follow_up_service.start_action_item(action_item_id, current_user)
    return {"success": success}


@router.post("/follow-up/{action_item_id}/complete")
def complete_action_item(
    action_item_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, bool]:
    """
    Mark an action item as completed.
    
    Marks the action item as "completed" if the user has permission.
    
    Security:
    - Requires authentication
    - User must be assigned to the action item OR have privileged role
    - Action item must be within user's organizational scope
    - Returns 403 if unauthorized (from service layer)
    """
    success = follow_up_service.complete_action_item(action_item_id, current_user)
    return {"success": success}


@router.post("/follow-up/{action_item_id}/delay")
def delay_action_item(
    action_item_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, bool]:
    """
    Delay an action item.
    
    Marks the action item as "delayed" if the user has permission.
    
    Security:
    - Requires authentication
    - User must be assigned to the action item OR have privileged role
    - Action item must be within user's organizational scope
    - Returns 403 if unauthorized (from service layer)
    """
    success = follow_up_service.delay_action_item(action_item_id, current_user)
    return {"success": success}


# ============================================================
# CASE ACTION ENDPOINTS (STEP 3.5.4)
# ============================================================

@router.post("/case/{subcase_id}/act")
def act_on_case(
    subcase_id: int,
    body: Dict[str, Any],
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, bool]:
    """
    Unified workflow action endpoint for subcase state transitions.
    
    Accepts a payload-driven action and dispatches to the appropriate
    service function based on the action type.
    
    Supported actions:
    - SUBMIT_RESPONSE: Section admin submits explanation + action items
    - REJECT: Reject responsibility/response at any level
    - APPROVE: Approve response at department or administration level
    - OVERRIDE: Override response at department or administration level
    - FORCE_CLOSE: Force close subcase (administration only)
    
    Request body:
    {
        "action": "SUBMIT_RESPONSE" | "REJECT" | "APPROVE" | "OVERRIDE" | "FORCE_CLOSE",
        "payload": {
            "explanation_text": "string (optional)",
            "rejection_text": "string (optional)",
            "action_items": [...] (optional),
            "reason": "string (optional)"
        }
    }
    
    Security:
    - Authentication required
    - Role and scope validation performed by service layer
    - Returns 403 if unauthorized
    - Returns 400 if invalid action or payload
    
    Note: Guards are NOT applied here because different actions require
    different permission levels. Service layer enforces all authorization.
    """
    action = body.get("action")
    payload = body.get("payload", {})
    
    # Support both nested payload and flat structure for backward compatibility
    # If payload is empty but body has other keys, use body as payload
    if not payload and len(body) > 1:
        payload = {k: v for k, v in body.items() if k != "action"}
    
    if action == "SUBMIT_RESPONSE":
        # Section administrator submits response
        explanation_text = payload.get("explanation_text", "")
        action_items = payload.get("action_items", [])
        case_response_service.submit_section_response(
            subcase_id=subcase_id,
            explanation_text=explanation_text,
            action_items=action_items,
            current_user=current_user
        )
        return {"success": True}
    
    elif action == "REJECT":
        # Reject at section, department, or administration level
        rejection_text = payload.get("rejection_text", "")
        
        # Determine level based on current status - try each level and let proper exceptions bubble up
        errors = []
        
        # Try section level first (most common)
        try:
            case_response_service.reject_responsibility(
                subcase_id=subcase_id,
                rejection_text=rejection_text,
                current_user=current_user
            )
            return {"success": True}
        except Exception as e:
            errors.append(f"Section: {str(e)}")
        
        # Try department level
        try:
            case_response_service.reject_department(
                subcase_id=subcase_id,
                rejection_text=rejection_text,
                current_user=current_user
            )
            return {"success": True}
        except Exception as e:
            errors.append(f"Department: {str(e)}")
        
        # Try administration level
        try:
            case_response_service.reject_administration(
                subcase_id=subcase_id,
                rejection_text=rejection_text,
                current_user=current_user
            )
            return {"success": True}
        except Exception as e:
            errors.append(f"Administration: {str(e)}")
        
        # If all failed, report errors
        raise HTTPException(status_code=400, detail=f"Reject failed at all levels: {'; '.join(errors)}")
    
    elif action == "APPROVE":
        # Approve at department or administration level
        errors = []
        
        # Try department first
        try:
            case_response_service.approve_department(
                subcase_id=subcase_id,
                current_user=current_user
            )
            return {"success": True}
        except Exception as e:
            errors.append(f"Department: {str(e)}")
        
        # Try administration
        try:
            case_response_service.approve_administration(
                subcase_id=subcase_id,
                current_user=current_user
            )
            return {"success": True}
        except Exception as e:
            errors.append(f"Administration: {str(e)}")
        
        # If both failed, report errors
        raise HTTPException(status_code=400, detail=f"Approve failed at all levels: {'; '.join(errors)}")
    
    elif action == "OVERRIDE":
        # Override at department or administration level
        explanation_text = payload.get("explanation_text", "")
        action_items = payload.get("action_items", [])
        
        errors = []
        
        # Try department override first
        try:
            case_response_service.override_department(
                subcase_id=subcase_id,
                explanation_text=explanation_text,
                action_items=action_items,
                current_user=current_user
            )
            return {"success": True}
        except Exception as e:
            errors.append(f"Department: {str(e)}")
        
        # Try administration override
        try:
            case_response_service.override_administration(
                subcase_id=subcase_id,
                explanation_text=explanation_text,
                action_items=action_items,
                current_user=current_user
            )
            return {"success": True}
        except Exception as e:
            errors.append(f"Administration: {str(e)}")
        
        # If both failed, report errors
        raise HTTPException(status_code=400, detail=f"Override failed at all levels: {'; '.join(errors)}")
    
    elif action == "FORCE_CLOSE":
        # Force close (administration only)
        reason = payload.get("reason", "")
        case_response_service.force_close_subcase(
            subcase_id=subcase_id,
            reason_text=reason,
            current_user=current_user
        )
        return {"success": True}
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

