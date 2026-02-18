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


@router.get("/inbox/archive")
def get_inbox_archive(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get archive (completed/processed items) for current user.
    
    Returns subcases that the user has previously processed and that have
    moved past their workflow stage. These are READ-ONLY - no workflow
    actions can be performed on archived items.
    
    Use cases:
    - Section admin: View cases they approved/rejected that moved to department
    - Department admin: View cases they processed that moved to administration
    - Administration admin: View cases they finalized (approved/force-closed)
    - Worker/Supervisor: View cases they reopened
    
    Each archive item contains:
    - subcase_id: Unique identifier
    - case_type: 'INCIDENT' or 'SEASONAL_REPORT'
    - incident_id: Source incident ID (if applicable)
    - seasonal_report_id: Source seasonal report ID (if applicable)
    - target_org_unit_id: Target organizational unit
    - status: Current workflow status (showing where it ended up)
    - created_at: Creation timestamp
    - updated_at: When it was last processed
    - allowed_actions: Always ["view"] (archive is read-only)
    
    Security: Requires authentication. Role and scope filtering applied by service.
    """
    items = inbox_service.get_archive(current_user)
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
    try:
        success = follow_up_service.start_action_item(action_item_id, current_user)
        return {"success": success}
    except follow_up_service.NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except follow_up_service.Unauthorized as e:
        raise HTTPException(status_code=401, detail=str(e))
    except follow_up_service.Forbidden as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    try:
        success = follow_up_service.complete_action_item(action_item_id, current_user)
        return {"success": success}
    except follow_up_service.NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except follow_up_service.Unauthorized as e:
        raise HTTPException(status_code=401, detail=str(e))
    except follow_up_service.Forbidden as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/follow-up/{action_item_id}/delay")
def delay_action_item(
    action_item_id: int,
    body: Dict[str, Any] = {},
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Delay an action item by extending its due date.
    
    Request body:
    {
        "delay_days": 7   // Number of days to extend (1-90, default: 7)
    }
    
    Response:
    {
        "success": true,
        "action_item_id": 323,
        "previous_due_date": "2026-02-26",
        "new_due_date": "2026-03-05",
        "delay_days": 7
    }
    
    Security:
    - Requires authentication
    - User must be assigned to the action item OR have privileged role
    - Action item must be within user's organizational scope
    - Returns 403 if unauthorized (from service layer)
    """
    try:
        delay_days = body.get("delay_days", 7) if body else 7
        
        # Validate
        if not isinstance(delay_days, int) or delay_days < 1 or delay_days > 90:
            raise HTTPException(
                status_code=400,
                detail="delay_days must be an integer between 1 and 90"
            )
        
        result = follow_up_service.delay_action_item(action_item_id, delay_days, current_user)
        return result
    except HTTPException:
        raise
    except follow_up_service.NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except follow_up_service.Unauthorized as e:
        raise HTTPException(status_code=401, detail=str(e))
    except follow_up_service.Forbidden as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# SEASONAL REPORT VIEWER (read-only)
# ============================================================

@router.get("/seasonal-report/{seasonal_report_id}")
def get_seasonal_report_detail(
    seasonal_report_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Return the full seasonal report data for an inbox "view" action.

    Fetches report header (severity/domain counts, compliance),
    classification breakdowns, and policy snapshot.

    Authorization: user must have the report's OrgUnitID in their scope.

    Response (200):
    {
        "header": { "seasonal_report_id", "period", "orgunit_name",
                     "total_cases", severity counts, domain counts,
                     "is_compliant", "violated_rules", ... },
        "classification_stats": [{ "classification_name", counts, ... }],
        "policy_snapshot": { limits, rule flags } | null
    }

    Errors:
        403 — user not in scope for this report's org unit
        404 — seasonal report not found
    """
    from backend.api.db_layer.seasonal_report import (
        get_seasonal_report_keys_by_id,
        get_full_seasonal_report,
    )
    from backend.api.services.scope_resolver import resolve_user_scope

    # Step 1: Resolve keys from the report ID
    keys = get_seasonal_report_keys_by_id(seasonal_report_id)
    if keys is None:
        raise HTTPException(status_code=404, detail=f"Seasonal report {seasonal_report_id} not found")

    # Step 2: Scope check — user must have access to this org unit
    # resolve_user_scope returns Set[int] of allowed org unit IDs
    allowed_unit_ids = resolve_user_scope(current_user)
    if keys['orgunit_id'] not in allowed_unit_ids:
        raise HTTPException(
            status_code=403,
            detail=f"You do not have access to org unit {keys['orgunit_id']}"
        )

    # Step 3: Fetch the full report
    report = get_full_seasonal_report(
        season_id=keys['season_id'],
        orgunit_id=keys['orgunit_id'],
        orgunit_type=keys['orgunit_type']
    )
    if report is None:
        raise HTTPException(status_code=404, detail="Seasonal report data not found")

    return report


# ============================================================
# CASE RESPONSE VIEWER (read-only)
# ============================================================

@router.get("/case/{subcase_id}/response")
def get_subcase_response(
    subcase_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Return the latest submitted response (explanation + action items)
    for a subcase so reviewers can inspect before accepting/rejecting.

    Authorization: same role-scope rules as inbox — enforced by
    case_response_service.get_subcase_response.

    Response (200):
    {
        "explanation_text": "...",
        "action_items": [...],
        "submitted_by": "username",
        "submitted_at": "2026-02-10T14:30:00"
    }

    Errors:
        403 — user not authorised for this subcase
        404 — subcase not found or no response submitted yet
    """
    return case_response_service.get_subcase_response(subcase_id, current_user)


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
    
    Accepts a payloaad-driven action and dispatches to the appropriate
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
    # Check if subcase is force-closed BEFORE any action
    from backend.api_v2.db_layer import administrative_subcase_db
    subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
    
    if not subcase:
        raise HTTPException(status_code=404, detail=f"Subcase {subcase_id} not found")
    
    # Block all actions on force-closed subcases
    if subcase.get('status') == 'FORCE_CLOSED':
        raise HTTPException(
            status_code=400,
            detail="Cannot perform actions on force-closed cases. This case was administratively closed."
        )
    
    # Check if parent incident is force-closed
    incident_id = subcase.get('incident_request_case_id')
    if incident_id:
        from backend.api.db_layer import incident_case
        incident = incident_case.get_incident_case_by_id(incident_id)
        if incident and incident.get('ForceClosedAt'):
            raise HTTPException(
                status_code=400,
                detail="Cannot perform actions on force-closed cases. The parent incident was administratively closed."
            )
    
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
    
    elif action == "REOPEN":
        # Complaint Supervisor reopens a SECTION_DENIED case
        rejection_text = payload.get("rejection_text", "")
        case_response_service.reopen_denied_case(
            subcase_id=subcase_id,
            rejection_text=rejection_text,
            current_user=current_user
        )
        return {"success": True}
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")


# ============================================================
# FORCE CLOSE ENDPOINT (Administrative)
# ============================================================

@router.post("/case/{incident_id}/force-close")
def force_close_case_and_subcases(
    incident_id: int,
    body: Dict[str, Any],
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Force close an incident and ALL its subcases (Administrative).
    
    This endpoint allows authorized roles to administratively close cases that are:
    - Stuck in workflow
    - Duplicates
    - Need immediate closure
    
    Authorization:
    - SOFTWARE_ADMIN: Full administrative access
    - WORKER: Administrative access for case management
    - COMPLAINT_SUPERVISOR: Supervisory access for case oversight
    
    All other roles will receive 403 Forbidden.
    
    Request Body:
    {
        "reason": "Reason for force closing (min 10 characters)"
    }
    
    Response:
    {
        "success": true,
        "incident_id": 123,
        "incident_status": "FORCE_CLOSED",
        "subcases_closed": [456, 457, 458],
        "total_subcases_closed": 3,
        "closed_at": "2026-02-10T15:30:00Z",
        "closed_by": "admin_user",
        "reason": "Duplicate case - merged with incident #12345"
    }
    
    Error Responses:
    - 403: User does not have permission (not SOFTWARE_ADMIN, WORKER, or COMPLAINT_SUPERVISOR)
    - 404: Incident not found
    - 400: Invalid request (reason too short, etc.)
    """
    # Authorization check: Only SOFTWARE_ADMIN, WORKER, COMPLAINT_SUPERVISOR
    if not current_user or not current_user.scopes:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions. Only SOFTWARE_ADMIN, WORKER, or COMPLAINT_SUPERVISOR can force close cases."
        )
    
    primary_role = current_user.scopes[0].role_code
    allowed_roles = ['SOFTWARE_ADMIN', 'WORKER', 'COMPLAINT_SUPERVISOR']
    
    if primary_role not in allowed_roles:
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient permissions. Only SOFTWARE_ADMIN, WORKER, or COMPLAINT_SUPERVISOR can force close cases. Your role: {primary_role}"
        )
    
    # Extract reason from body
    reason = body.get("reason", "")
    
    # Validate reason
    if not reason or len(reason) < 10:
        raise HTTPException(
            status_code=400,
            detail="Reason is required and must be at least 10 characters."
        )
    
    # Verify incident exists
    from backend.api.db_layer import incident_case
    incident = incident_case.get_incident_case_by_id(incident_id)
    if not incident:
        raise HTTPException(
            status_code=404,
            detail=f"Incident ID {incident_id} not found."
        )
    
    # Force close incident and all subcases
    try:
        result = case_response_service.force_close_incident(
            incident_id=incident_id,
            reason_text=reason,
            current_user=current_user
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to force close incident: {str(e)}"
        )


