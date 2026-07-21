"""
Supervisor Action Item Router (API v2)
Action Item Coordination (Iteration 4, administrative scope).

Allows the Complaint Supervisor to create Action Items that target any
organizational unit or person, independent of the case's own escalation
chain. No acceptance/proposal workflow - administrative assignment only.

Security: create/cancel restricted to Complaint Supervisor/Software Admin.
List/complete open to any authenticated user, scoped by the service layer.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List

from api.dependencies.user_context import get_current_user
from api.schemas.auth_models import CurrentUser
from api.utils.guards import require_software_admin
from api_v2.services import supervisor_action_item_service as service


router = APIRouter(prefix="/api/v2/supervisor-action-items", tags=["Supervisor Action Items"])


@router.post("")
def create_action_item(
    body: Dict[str, Any],
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a new supervisor action item targeting any org unit/person.

    Authorization: Complaint Supervisor / Software Admin only.

    Request body:
    {
        "incident_request_case_id": 123,
        "target_org_unit_id": 45,
        "description": "...",
        "subcase_id": 678,            // optional
        "target_user_id": 9,          // optional
        "due_date": "YYYY-MM-DD"      // optional
    }
    """
    require_software_admin(current_user)

    incident_request_case_id = body.get("incident_request_case_id")
    target_org_unit_id = body.get("target_org_unit_id")
    description = body.get("description")

    if incident_request_case_id is None or target_org_unit_id is None or not description:
        raise HTTPException(
            status_code=400,
            detail="incident_request_case_id, target_org_unit_id, and description are required."
        )

    try:
        item = service.create_action_item(
            current_user=current_user,
            incident_request_case_id=incident_request_case_id,
            target_org_unit_id=target_org_unit_id,
            description=description,
            subcase_id=body.get("subcase_id"),
            target_user_id=body.get("target_user_id"),
            due_date=body.get("due_date"),
        )
        return {"item": item}
    except service.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except service.ForbiddenError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except service.SupervisorActionItemError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("")
def list_action_items(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    List supervisor action items visible to the current user.

    Scoped by the caller's resolved organizational scope - hospital-wide for
    Complaint Supervisor/Software Admin/Worker, a single subtree otherwise.
    """
    items = service.list_action_items_for_user(current_user)
    return {"items": items}


@router.get("/unacknowledged")
def list_unacknowledged_action_items(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    List unacknowledged supervisor action item assignments for the current
    user's inbox — items targeted directly at them, or at an org unit in
    their scope when no specific target user was set.
    """
    items = service.list_unacknowledged_for_user(current_user)
    return {"items": items}


@router.post("/{action_item_id}/complete")
def complete_action_item(
    action_item_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Mark a supervisor action item as completed.

    Authorization: the target unit's scope, or the specifically named target user.
    """
    try:
        item = service.complete_action_item(current_user, action_item_id)
        return {"item": item}
    except service.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except service.ForbiddenError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except service.SupervisorActionItemError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{action_item_id}/cancel")
def cancel_action_item(
    action_item_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Cancel a supervisor action item.

    Authorization: the creator only (Complaint Supervisor who created it).
    """
    try:
        item = service.cancel_action_item(current_user, action_item_id)
        return {"item": item}
    except service.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except service.ForbiddenError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except service.SupervisorActionItemError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{action_item_id}/acknowledge")
def acknowledge_action_item(
    action_item_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Acknowledge receipt of a supervisor action item assignment (inbox notification).

    Pure notification side-channel — does not change Status. Calling this
    on an already-acknowledged item is a safe no-op (idempotent).

    Authorization: the target unit's scope, or the specifically named target user.
    """
    try:
        item = service.acknowledge_action_item(current_user, action_item_id)
        return {"item": item}
    except service.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except service.ForbiddenError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except service.SupervisorActionItemError as e:
        raise HTTPException(status_code=400, detail=str(e))
