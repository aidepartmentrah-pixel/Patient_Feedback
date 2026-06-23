"""
Workflow Router (API v2)
Unified workflow management endpoints for Phase 3.5.

This router will provide:
- Inbox endpoints (STEP 3.5.2)
- Follow-up action endpoints (to be added in STEP 3.5.3)
- Case action endpoints (to be added in STEP 3.5.4)
- Incident detail endpoint (read-only for workflow inbox)

Security: All endpoints protected by role guards and scope enforcement.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from backend.api.dependencies.user_context import get_current_user
from backend.api.schemas.auth_models import CurrentUser
from backend.api_v2.services import inbox_service, follow_up_service, case_response_service, workflow_incident_service
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


@router.get("/inbox/accountability")
def get_inbox_accountability(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Return force-close accountability data for SECTION_ADMIN and DEPARTMENT_ADMIN.

    Response:
        {
          "red":  [ {...subcase fields...} ],   # still escalated, view-only
          "gray": [ {...subcase fields...} ],   # restored + progressed, dismissible
        }

    Other roles receive empty lists.
    """
    return inbox_service.get_inbox_accountability(current_user)


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
# INCIDENT DETAIL VIEWER (read-only for workflow inbox)
# ============================================================

@router.get("/incident/{incident_id}")
def get_workflow_incident_detail(
    incident_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Return read-only incident details for the workflow inbox "View" action.
    
    Allows section admins, department admins, and administration admins to 
    view incident details without needing full edit permissions (WORKER-level).
    
    Authorization:
    - User must have an active subcase assigned to their org unit for this incident, OR
    - User must be COMPLAINT_SUPERVISOR or higher (SOFTWARE_ADMIN)
    
    Response (200):
    {
        "incident_id": 503,
        "complaint_text": "...",
        "domain_name": "...",
        "category_name": "...",
        "subcategory_name": "...",
        "classification_name": "...",
        "severity_name": "...",
        "issuing_department_name": "...",
        "target_departments": [{"id": 1, "name": "..."}],
        "building": "BIC",
        "in_out": "IN",
        "feedback_received_date": "2026-02-10",
        "created_at": "2026-02-10T...",
        "patient_name": "...",
        "immediate_action": "...",
        "taken_action": "...",
        "doctors": [{"doctor_id": 1, "name": "..."}],
        "employees": [{"employee_id": 1, "full_name": "..."}]
    }
    
    Errors:
        403 — user not authorized to view this incident
        404 — incident not found
    """
    # Step 1: Authorization check
    if not workflow_incident_service.can_view_incident(incident_id, current_user):
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to view this incident. "
                   "You must have an active subcase for this incident in your org unit scope."
        )
    
    # Step 2: Fetch incident details
    incident = workflow_incident_service.get_incident_detail(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    
    return incident


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
# INVESTIGATION HISTORY VIEWER (Stage 6 – read-only)
# ============================================================

@router.get("/case/{subcase_id}/history")
def get_subcase_history(
    subcase_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Return the full investigation history for a subcase:
    Section / Department / Administration explanation texts and
    Patient Services decision, each with available metadata.
    Also returns action items attached to the subcase.

    Authorization: user must have the subcase in their allowed scope
    (same rule as inbox).

    Response (200):
    {
        "subcase_id": 123,
        "org_unit_name": "Cardiology",
        "section":        { "has_content": bool, "text": str|null, "entered_by": str|null, "entered_at": str|null },
        "department":     { ... },
        "administration": { ... },
        "patient_services": { ... },
        "action_items": [{ "title": str, "description": str|null, "due_date": str|null, "status": str }]
    }

    Errors:
        403 — subcase not in user scope
        404 — subcase not found
    """
    from backend.api_v2.db_layer import administrative_subcase_db, action_item_subcase_db

    history = administrative_subcase_db.get_subcase_history(subcase_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"Subcase {subcase_id} not found")

    # Scope check — subcase target unit must be in user's allowed units
    subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
    allowed_unit_ids = set(current_user.allowed_unit_ids) if current_user.allowed_unit_ids else set()
    is_privileged = any(
        s.role_code in ("SOFTWARE_ADMIN", "COMPLAINT_SUPERVISOR", "ADMINISTRATION_ADMIN")
        for s in (current_user.scopes or [])
    )
    if not is_privileged and subcase.get("target_org_unit_id") not in allowed_unit_ids:
        raise HTTPException(status_code=403, detail="Not authorised to view this subcase")

    # Attach action items
    raw_items = action_item_subcase_db.get_action_items_by_subcase(subcase_id)
    history["action_items"] = [
        {
            "title": i.get("title"),
            "description": i.get("description") or None,
            "due_date": i.get("due_date").strftime("%Y-%m-%d") if i.get("due_date") else None,
            "status": i.get("status"),
        }
        for i in raw_items
    ]

    return history


# ============================================================
# CASE ACTION ENDPOINTS (STEP 3.5.4)
# ============================================================

@router.post("/case/{subcase_id}/act")
def act_on_case(
    subcase_id: int,
    body: Dict[str, Any],
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
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
    - REOPEN: Complaint supervisor reopens a SECTION_DENIED case

    Request body:
    {
        "action": "SUBMIT_RESPONSE" | "REJECT" | "APPROVE" | "OVERRIDE" | "FORCE_CLOSE" | "REOPEN",
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
    
    # Block all workflow actions on force-closed subcases (any variant)
    _fc_statuses = ('FORCE_CLOSED', 'FORCE_CLOSED_DRAFT', 'FORCE_CLOSED_COMPLETE')
    if subcase.get('status') in _fc_statuses:
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
        # Section administrator submits response with mandatory RCA
        explanation_text = payload.get("explanation_text", "")
        action_items = payload.get("action_items", [])
        rca_feedback = payload.get("rca_feedback", None)
        case_response_service.submit_section_response(
            subcase_id=subcase_id,
            explanation_text=explanation_text,
            action_items=action_items,
            current_user=current_user,
            rca_feedback=rca_feedback
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
        # Force close — COMPLAINT_SUPERVISOR and WORKER only (enforced in service)
        reason = payload.get("reason", "")
        new_status = case_response_service.force_close_subcase(
            subcase_id=subcase_id,
            reason_text=reason,
            current_user=current_user
        )
        return {"success": True, "new_status": new_status}
    
    elif action == "REOPEN":
        # Complaint Supervisor reopens a SECTION_DENIED case
        rejection_text = payload.get("rejection_text", "")
        case_response_service.reopen_denied_case(
            subcase_id=subcase_id,
            rejection_text=rejection_text,
            current_user=current_user
        )
        return {"success": True}

    elif action == "ACCEPT_COMPLAINT":
        # Section admin accepts complaint with constant text, no action items, no RCA
        case_response_service.accept_section_complaint(
            subcase_id=subcase_id,
            current_user=current_user
        )
        return {"success": True}

    elif action == "SAVE_PATIENT_SERVICES_DECISION":
        # Complaint Supervisor records قرار خدمات المرضى بحسب المراجع العلميّة
        decision_text = payload.get("decision_text", "")
        case_response_service.save_patient_services_decision(
            subcase_id=subcase_id,
            decision_text=decision_text,
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
    Force close an incident and ALL its subcases.

    This endpoint allows authorized roles to administratively close cases that are:
    - Stuck in workflow
    - Duplicates
    - Need immediate closure

    Authorization:
    - COMPLAINT_SUPERVISOR: Supervisory access for case oversight
    - WORKER: Administrative access for case management

    All other roles will receive 403 Forbidden.

    Request Body:
    {
        "reason": "Reason for force closing (min 10 characters)"
    }

    Response:
    {
        "success": true,
        "incident_id": 123,
        "subcases_closed": [{"subcase_id": 456, "new_status": "FORCE_CLOSED_DRAFT"}, ...],
        "total_subcases_closed": 3,
        "closed_at": "2026-02-10T15:30:00Z",
        "closed_by": "admin_user",
        "reason": "Duplicate case - merged with incident #12345"
    }

    Error Responses:
    - 403: User does not have permission (not COMPLAINT_SUPERVISOR or WORKER)
    - 404: Incident not found
    - 400: Invalid request (reason too short, etc.)
    """
    # Authorization check: COMPLAINT_SUPERVISOR and WORKER only
    if not current_user or not current_user.scopes:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions. Only COMPLAINT_SUPERVISOR or WORKER can force close cases."
        )

    primary_role = current_user.scopes[0].role_code
    allowed_roles = ['COMPLAINT_SUPERVISOR', 'WORKER']

    if primary_role not in allowed_roles:
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient permissions. Only COMPLAINT_SUPERVISOR or WORKER can force close cases. Your role: {primary_role}"
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


# ============================================================
# AUTOMATIC FORCE CLOSE - MANUAL SCAN ENDPOINT (Session 4)
# ============================================================

@router.post("/automatic-force-close/run-scan")
def run_automatic_force_close_scan(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Manually trigger one pass of the Automatic Force Close scan.

    Runs the exact same automatic_force_close_service logic used by the
    hourly scheduled job (see backend/main.py
    startup_automatic_force_close_scheduler). Intended for
    testing/administration only - this endpoint is NOT used by the frontend.

    No-ops (returns skipped_disabled_policy=True) if
    automatic_force_close_enabled is not 'true'.

    Authorization:
    - SOFTWARE_ADMIN
    - COMPLAINT_SUPERVISOR

    All other roles receive 403 Forbidden.

    Response: summary dict from
    automatic_force_close_service.run_automatic_force_close_scan() -
    success, enabled, scanned_count, section_force_closed_count,
    department_force_closed_count, administration_force_closed_count,
    skipped_notice_count, skipped_morbidity_count, skipped_disabled_policy.
    """
    if not current_user or not current_user.scopes:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions. Only SOFTWARE_ADMIN or COMPLAINT_SUPERVISOR can run the automatic force-close scan."
        )

    primary_role = current_user.scopes[0].role_code
    allowed_roles = ['SOFTWARE_ADMIN', 'COMPLAINT_SUPERVISOR']

    if primary_role not in allowed_roles:
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient permissions. Only SOFTWARE_ADMIN or COMPLAINT_SUPERVISOR can run the automatic force-close scan. Your role: {primary_role}"
        )

    from backend.api_v2.services import automatic_force_close_service
    return automatic_force_close_service.run_automatic_force_close_scan(
        triggered_by_user_id=current_user.user_id
    )


# ============================================================
# INCIDENT RESPONSE SUMMARY (all subcases + responses for one incident)
# ============================================================

@router.get("/incident/{incident_id}/responses")
def get_incident_responses(
    incident_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Return all subcases for an incident, each with their submitted
    explanation texts and action items. Used by Table View's "View Responses" feature.

    Authorization: scope check — user must have at least one subcase's org unit in scope,
    OR be COMPLAINT_SUPERVISOR / SOFTWARE_ADMIN.

    Response:
    {
        "incident_id": 123,
        "subcases": [
            {
                "subcase_id": 1,
                "target_org_unit_id": 10,
                "target_org_unit_name": "...",
                "status": "ADMIN_APPROVED",
                "section_explanation": "..." | null,
                "department_explanation": "..." | null,
                "administration_explanation": "..." | null,
                "action_items": [{"title", "description", "due_date", "status"}]
            }
        ]
    }
    """
    from backend.api_v2.db_layer import administrative_subcase_db, action_item_subcase_db
    from backend.core.database import get_connection

    role_code = (
        current_user.scopes[0].role_code
        if current_user and current_user.scopes
        else None
    )
    privileged = role_code in ('COMPLAINT_SUPERVISOR', 'SOFTWARE_ADMIN', 'WORKER')
    allowed_unit_ids = getattr(current_user, 'allowed_unit_ids', None) or set()

    subcases = administrative_subcase_db.get_subcases_by_incident(incident_id)
    if not subcases:
        return {"incident_id": incident_id, "subcases": []}

    # Resolve org unit names in one query
    unit_ids = list({sc.get('target_org_unit_id') for sc in subcases if sc.get('target_org_unit_id')})
    unit_names: dict = {}
    if unit_ids:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(unit_ids))
            cursor.execute(
                f"SELECT UniqueID, Name FROM dbo.AdminsrationUnit WHERE UniqueID IN ({placeholders})",
                unit_ids
            )
            unit_names = {row.UniqueID: row.Name for row in cursor.fetchall()}
        finally:
            conn.close()

    result = []
    for sc in subcases:
        sc_unit = sc.get('target_org_unit_id')
        if not privileged and sc_unit not in allowed_unit_ids:
            continue

        raw_items = action_item_subcase_db.get_action_items_by_subcase(sc['subcase_id'])
        action_items = [
            {
                "title": item.get('title'),
                "description": item.get('description'),
                "due_date": item['due_date'].isoformat() if item.get('due_date') else None,
                "status": item.get('status'),
            }
            for item in raw_items
        ]
        result.append({
            "subcase_id": sc['subcase_id'],
            "target_org_unit_id": sc_unit,
            "target_org_unit_name": unit_names.get(sc_unit) or f"Unit {sc_unit}",
            "status": sc.get('status'),
            "section_explanation": sc.get('section_explanation_text') or None,
            "department_explanation": sc.get('department_explanation_text') or None,
            "administration_explanation": sc.get('administration_explanation_text') or None,
            "action_items": action_items,
        })

    return {"incident_id": incident_id, "subcases": result}


# ============================================================
# MANUAL INTERVENTION: FILL STATE (read)
# ============================================================

@router.get("/subcase/{subcase_id}/fill-state")
def get_subcase_fill_state(
    subcase_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Return the current manual-fill state for a subcase.

    Includes explanation texts for all three levels plus full ownership
    metadata (who entered each level, on behalf of which role, when).

    Authorization: COMPLAINT_SUPERVISOR and WORKER only.

    Response (200):
    {
        "subcase_id": 123,
        "status": "FORCE_CLOSED_DRAFT",
        "incident_id": 456,
        "force_close_reason": "...",
        "force_closed_at": "2026-05-12T10:00:00",
        "force_closed_by": "ali",
        "section": {
            "explanation_text": "...",
            "entered_by": "sara",
            "entered_for_role": "SECTION_ADMIN",
            "entry_mode": "FORCE_CLOSE_INTERVENTION",
            "entry_timestamp": "2026-05-12T11:00:00"
        },
        "department": { ... },
        "administration": { ... }
    }

    Errors:
        403 — caller is not COMPLAINT_SUPERVISOR or WORKER
        404 — subcase not found
    """
    if not current_user or not current_user.scopes:
        raise HTTPException(status_code=403, detail="Insufficient permissions.")
    primary_role = current_user.scopes[0].role_code
    if primary_role not in ('COMPLAINT_SUPERVISOR', 'WORKER'):
        raise HTTPException(
            status_code=403,
            detail="Only COMPLAINT_SUPERVISOR or WORKER can access the manual fill state."
        )
    from backend.api_v2.db_layer import administrative_subcase_db
    state = administrative_subcase_db.get_subcase_fill_state(subcase_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Subcase {subcase_id} not found.")
    return state


# ============================================================
# MANUAL INTERVENTION: FILL ON BEHALF ENDPOINTS
# ============================================================

@router.post("/subcase/{subcase_id}/fill/section")
def fill_section_on_behalf(
    subcase_id: int,
    body: Dict[str, Any],
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Fill the section explanation on behalf of SECTION_ADMIN.

    Authorization: COMPLAINT_SUPERVISOR and WORKER only.
    Works on active subcases and FORCE_CLOSED_DRAFT subcases.
    Overwrite of existing data is permitted.

    Sequential lock: none (section is always the first level).

    Request body:
    {
        "explanation_text": "...",
        "action_items": [{"title": "...", "description": "...", "due_date": "YYYY-MM-DD"}]  // optional
    }

    Response:
    { "success": true, "subcase_id": 123, "level": "section",
      "entry_mode": "ON_BEHALF" | "FORCE_CLOSE_INTERVENTION" }
    """
    explanation_text = body.get("explanation_text", "")
    if not explanation_text:
        raise HTTPException(status_code=400, detail="explanation_text is required")
    action_items = body.get("action_items", []) or []
    try:
        entry_mode = case_response_service.fill_explanation_on_behalf(
            subcase_id=subcase_id,
            level="section",
            explanation_text=explanation_text,
            current_user=current_user,
            action_items=action_items
        )
        return {"success": True, "subcase_id": subcase_id, "level": "section", "entry_mode": entry_mode}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/subcase/{subcase_id}/fill/department")
def fill_department_on_behalf(
    subcase_id: int,
    body: Dict[str, Any],
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Fill the department explanation on behalf of DEPARTMENT_ADMIN.

    Authorization: COMPLAINT_SUPERVISOR and WORKER only.
    Works on active subcases and FORCE_CLOSED_DRAFT subcases.
    Overwrite of existing data is permitted.

    Sequential lock: section explanation must already be present.

    Request body:
    { "explanation_text": "..." }

    Response:
    { "success": true, "subcase_id": 123, "level": "department",
      "entry_mode": "ON_BEHALF" | "FORCE_CLOSE_INTERVENTION" }
    """
    explanation_text = body.get("explanation_text", "")
    if not explanation_text:
        raise HTTPException(status_code=400, detail="explanation_text is required")
    try:
        entry_mode = case_response_service.fill_explanation_on_behalf(
            subcase_id=subcase_id,
            level="department",
            explanation_text=explanation_text,
            current_user=current_user
        )
        return {"success": True, "subcase_id": subcase_id, "level": "department", "entry_mode": entry_mode}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/subcase/{subcase_id}/fill/administration")
def fill_administration_on_behalf(
    subcase_id: int,
    body: Dict[str, Any],
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Fill the administration explanation on behalf of ADMINISTRATION_ADMIN.

    Authorization: COMPLAINT_SUPERVISOR and WORKER only.
    Works on active subcases and FORCE_CLOSED_DRAFT subcases.
    Overwrite of existing data is permitted.

    Sequential lock: department explanation must already be present.

    Request body:
    { "explanation_text": "..." }

    Response:
    { "success": true, "subcase_id": 123, "level": "administration",
      "entry_mode": "ON_BEHALF" | "FORCE_CLOSE_INTERVENTION" }
    """
    explanation_text = body.get("explanation_text", "")
    if not explanation_text:
        raise HTTPException(status_code=400, detail="explanation_text is required")
    try:
        entry_mode = case_response_service.fill_explanation_on_behalf(
            subcase_id=subcase_id,
            level="administration",
            explanation_text=explanation_text,
            current_user=current_user
        )
        return {"success": True, "subcase_id": subcase_id, "level": "administration", "entry_mode": entry_mode}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# COMPLETE FORCE CLOSED DRAFT ENDPOINT
# ============================================================

@router.post("/subcase/{subcase_id}/complete-force-close")
def complete_force_closed_draft(
    subcase_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Transition a FORCE_CLOSED_DRAFT subcase to FORCE_CLOSED_COMPLETE.

    Called after all required data (section, department, administration
    explanations) has been filled in via the manual intervention API.

    Authorization: COMPLAINT_SUPERVISOR and WORKER only.

    Error Responses:
    - 403: User does not have permission
    - 404: Subcase not found
    - 400: Subcase is not in FORCE_CLOSED_DRAFT, or data is still incomplete
    """
    try:
        case_response_service.complete_force_closed_draft(
            subcase_id=subcase_id,
            current_user=current_user
        )
        return {"success": True, "subcase_id": subcase_id, "new_status": "FORCE_CLOSED_COMPLETE"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ============================================================
# GIVE MORE TIME WORKFLOW (HCAT Automatic Force Close Policy - Session 5)
# ============================================================

@router.post("/subcase/{subcase_id}/give-section-more-time")
def give_section_more_time(
    subcase_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Give the Section a fresh deadline after it was automatically
    force-closed (Status = FORCE_CLOSED_AT_SECTION).

    Restarts SectionDeadlineAt from now using the current
    section_deadline_days setting, records SectionExtraTimeGrantedAt/By,
    and returns Status to SUBMITTED_TO_SECTION so the Section is responsible
    and editable again. SectionForceClosedAt and SectionLateReply are
    preserved permanently (history is never erased).

    Authorization: DEPARTMENT_ADMIN, COMPLAINT_SUPERVISOR, SOFTWARE_ADMIN
    (subcase must be within the caller's organizational scope).

    Error Responses:
    - 403: Unauthorized role or subcase outside caller's scope
    - 400: Subcase not found, or not currently FORCE_CLOSED_AT_SECTION
        (e.g. not force-closed, force-closed at a different level, or
        already restored)
    """
    try:
        state = case_response_service.give_section_more_time(
            subcase_id=subcase_id,
            current_user=current_user
        )
        return {"success": True, "subcase_id": subcase_id, "workflow_state": state}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/subcase/{subcase_id}/give-department-more-time")
def give_department_more_time(
    subcase_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Give the Department a fresh deadline after it was automatically
    force-closed (Status = FORCE_CLOSED_AT_DEPARTMENT).

    Restarts DepartmentDeadlineAt from now using the current
    department_deadline_days setting, records
    DepartmentExtraTimeGrantedAt/By, and returns Status to
    SECTION_ACCEPTED_PENDING_DEPT so the Department is responsible and
    editable again. DepartmentForceClosedAt and DepartmentLateReply are
    preserved permanently (history is never erased).

    Authorization: ADMINISTRATION_ADMIN, COMPLAINT_SUPERVISOR, SOFTWARE_ADMIN
    (subcase must be within the caller's organizational scope).

    Error Responses:
    - 403: Unauthorized role or subcase outside caller's scope
    - 400: Subcase not found, or not currently FORCE_CLOSED_AT_DEPARTMENT
        (e.g. not force-closed, force-closed at a different level, or
        already restored)
    """
    try:
        state = case_response_service.give_department_more_time(
            subcase_id=subcase_id,
            current_user=current_user
        )
        return {"success": True, "subcase_id": subcase_id, "workflow_state": state}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/subcase/{subcase_id}/give-administration-more-time")
def give_administration_more_time(
    subcase_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Give Administration a fresh deadline after it was automatically
    force-closed (Status = FORCE_CLOSED_AT_ADMINISTRATION).

    Restarts AdministrationDeadlineAt from now using the current
    administration_deadline_days setting, records
    AdministrationExtraTimeGrantedAt/By, and returns Status to
    DEPT_ACCEPTED_PENDING_ADMIN so Administration is responsible and editable
    again. AdministrationForceClosedAt and AdministrationLateReply are
    preserved permanently (history is never erased).

    Authorization: COMPLAINT_SUPERVISOR, SOFTWARE_ADMIN only. Administration
    is the top operational level - no ADMINISTRATION-level role sits above it
    to grant the extension.

    Error Responses:
    - 403: Unauthorized role or subcase outside caller's scope
    - 400: Subcase not found, or not currently FORCE_CLOSED_AT_ADMINISTRATION
        (e.g. not force-closed, force-closed at a different level, or
        already restored)
    """
    try:
        state = case_response_service.give_administration_more_time(
            subcase_id=subcase_id,
            current_user=current_user
        )
        return {"success": True, "subcase_id": subcase_id, "workflow_state": state}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
