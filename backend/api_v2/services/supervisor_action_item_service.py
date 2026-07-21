"""
Supervisor Action Item Service (API V2)
Business logic for Action Item Coordination (Iteration 4, administrative scope).

Validates and orchestrates create/complete/cancel for APP_SupervisorActionItem.
Deliberately independent of action_item_subcase_db.py / follow_up_service.py —
this is a separate, simpler administrative assignment mechanism, not the
case-escalation-gated workflow those use.
"""

from typing import Dict, Any, List, Optional
from datetime import date

from api.services import org_tree_service
from api.db_layer import incident_case
from api_v2.db_layer import administrative_subcase_db
from api_v2.db_layer import supervisor_action_item_db as db


class SupervisorActionItemError(Exception):
    """Base for validation/business-rule failures. Router maps this to HTTP 400."""


class NotFoundError(SupervisorActionItemError):
    """Case/subcase/org-unit/action-item does not exist. Router maps this to HTTP 404."""


class ForbiddenError(SupervisorActionItemError):
    """Current user is not authorized for this action. Router maps this to HTTP 403."""


def create_action_item(
    current_user,
    incident_request_case_id: int,
    target_org_unit_id: int,
    description: str,
    subcase_id: Optional[int] = None,
    target_user_id: Optional[int] = None,
    due_date: Optional[date] = None,
) -> Dict[str, Any]:
    """Create a new supervisor action item, after validating the case/subcase/target."""
    if not description or not description.strip():
        raise SupervisorActionItemError("Description is required.")

    if incident_case.get_incident_case_by_id(incident_request_case_id) is None:
        raise NotFoundError(f"Case {incident_request_case_id} does not exist.")

    if subcase_id is not None:
        subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
        if subcase is None:
            raise NotFoundError(f"Subcase {subcase_id} does not exist.")
        if subcase["incident_request_case_id"] != incident_request_case_id:
            raise SupervisorActionItemError(
                f"Subcase {subcase_id} does not belong to case {incident_request_case_id}."
            )

    frozen = db.get_org_unit_frozen_status(target_org_unit_id)
    if frozen is None:
        raise NotFoundError(f"Target org unit {target_org_unit_id} does not exist.")
    if frozen:
        raise SupervisorActionItemError(
            f"Target org unit {target_org_unit_id} is frozen/inactive and cannot receive new action items."
        )

    if target_user_id is not None:
        valid_unit_ids = org_tree_service.get_descendants(target_org_unit_id)
        user_unit_ids = set(db.get_org_unit_ids_for_user(target_user_id))
        if not user_unit_ids & valid_unit_ids:
            raise SupervisorActionItemError(
                f"Target user {target_user_id} does not belong to org unit {target_org_unit_id} "
                "or any of its sub-units."
            )

    primary_role = current_user.roles[0] if getattr(current_user, "roles", None) else "UNKNOWN"

    action_item_id = db.create_supervisor_action_item(
        incident_request_case_id=incident_request_case_id,
        target_org_unit_id=target_org_unit_id,
        created_by_user_id=current_user.user_id,
        created_by_role_code=primary_role,
        description=description.strip(),
        subcase_id=subcase_id,
        target_user_id=target_user_id,
        due_date=due_date,
    )
    if action_item_id is None:
        raise SupervisorActionItemError("Failed to create action item.")

    db.create_audit_log_entry(action_item_id, "CREATED", current_user.user_id)

    return db.get_supervisor_action_item_by_id(action_item_id)


def list_action_items_for_user(current_user) -> List[Dict[str, Any]]:
    """
    List supervisor action items visible to the current user, scoped by their
    resolved allowed_unit_ids — already the full hospital-wide list for
    Complaint Supervisor/Software Admin/Worker, or a single subtree otherwise.
    """
    allowed_unit_ids = list(getattr(current_user, "allowed_unit_ids", None) or [])
    return db.get_supervisor_action_items_by_org_units(allowed_unit_ids)


def list_unacknowledged_for_user(current_user) -> List[Dict[str, Any]]:
    """
    List unacknowledged supervisor action item assignments for the current
    user's inbox notifications — items targeted directly at them, or at an
    org unit in their scope when no specific target user was set.
    """
    allowed_unit_ids = list(getattr(current_user, "allowed_unit_ids", None) or [])
    return db.get_unacknowledged_supervisor_action_items_for_user(current_user.user_id, allowed_unit_ids)


def complete_action_item(current_user, action_item_id: int) -> Dict[str, Any]:
    item = _get_or_raise(action_item_id)
    _require_status(item, "PENDING", "completed")
    _require_target_access(current_user, item)

    if not db.set_supervisor_action_item_completed(action_item_id, current_user.user_id):
        raise SupervisorActionItemError(f"Action item {action_item_id} not found.")

    db.create_audit_log_entry(action_item_id, "COMPLETED", current_user.user_id)
    return db.get_supervisor_action_item_by_id(action_item_id)


def cancel_action_item(current_user, action_item_id: int) -> Dict[str, Any]:
    item = _get_or_raise(action_item_id)
    _require_status(item, "PENDING", "cancelled")

    if item["created_by_user_id"] != current_user.user_id:
        raise ForbiddenError("Only the creator can cancel this action item.")

    if not db.set_supervisor_action_item_cancelled(action_item_id, current_user.user_id):
        raise SupervisorActionItemError(f"Action item {action_item_id} not found.")

    db.create_audit_log_entry(action_item_id, "CANCELLED", current_user.user_id)
    return db.get_supervisor_action_item_by_id(action_item_id)


def acknowledge_action_item(current_user, action_item_id: int) -> Dict[str, Any]:
    """
    Acknowledge receipt of a supervisor action item assignment.

    Pure notification side-channel — independent of Status (Pending/
    Completed/Cancelled). Authorization is the same as completion: the
    target unit's scope, or the specifically named target user.
    """
    item = _get_or_raise(action_item_id)
    _require_target_access(current_user, item)

    if not db.acknowledge_supervisor_action_item(action_item_id, current_user.user_id):
        if item["acknowledged_at"] is not None:
            return db.get_supervisor_action_item_by_id(action_item_id)
        raise SupervisorActionItemError(f"Action item {action_item_id} not found.")

    return db.get_supervisor_action_item_by_id(action_item_id)


def _get_or_raise(action_item_id: int) -> Dict[str, Any]:
    item = db.get_supervisor_action_item_by_id(action_item_id)
    if item is None:
        raise NotFoundError(f"Action item {action_item_id} not found.")
    return item


def _require_status(item: Dict[str, Any], required: str, attempted_action: str) -> None:
    if item["status"] != required:
        raise SupervisorActionItemError(
            f"Cannot mark action item as {attempted_action}: status is "
            f"'{item['status']}' but must be '{required}'."
        )


def _require_target_access(current_user, item: Dict[str, Any]) -> None:
    """Completion is allowed for the target unit's scope, or the specifically named target user."""
    allowed_unit_ids = set(getattr(current_user, "allowed_unit_ids", None) or [])
    if item["target_org_unit_id"] in allowed_unit_ids:
        return
    if item["target_user_id"] is not None and item["target_user_id"] == current_user.user_id:
        return
    raise ForbiddenError("You are not authorized to act on this action item.")
