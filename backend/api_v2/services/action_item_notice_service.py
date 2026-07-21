"""
Action Item Change Notice Service (API V2)
Business logic for acknowledging APP_SubcaseActionItemChangeNotice rows.

Part of Action Item Coordination — AIC-S7 Notification Foundation.
The notice itself is created by case_response_service._upsert_action_items
whenever a higher-hierarchy edit changes a lower level's suggested action
item. This service only handles the recipient's acknowledgment.
"""

from typing import Dict, Any, List

from api_v2.db_layer import action_item_change_notice_db as db


class ActionItemNoticeError(Exception):
    """Base for validation/business-rule failures. Router maps this to HTTP 400."""


class NotFoundError(ActionItemNoticeError):
    """Notice does not exist. Router maps this to HTTP 404."""


class ForbiddenError(ActionItemNoticeError):
    """Current user is not the notice's recipient. Router maps this to HTTP 403."""


def list_unacknowledged_for_user(current_user) -> List[Dict[str, Any]]:
    """List pending action-item change notices addressed to the current user's inbox."""
    return db.get_unacknowledged_change_notices_for_user(current_user.user_id)


def acknowledge_change_notice(current_user, notice_id: int) -> Dict[str, Any]:
    """
    Acknowledge a pending action-item change notice.

    Authorization: the recipient only (the original creator of the action
    item whose values were changed). Calling this on an already-acknowledged
    notice is a safe no-op (idempotent).
    """
    notice = db.get_change_notice_by_id(notice_id)
    if notice is None:
        raise NotFoundError(f"Notice {notice_id} not found.")

    if notice["recipient_user_id"] != current_user.user_id:
        raise ForbiddenError("You are not authorized to acknowledge this notice.")

    if not db.acknowledge_change_notice(notice_id, current_user.user_id):
        if notice["acknowledged_at"] is not None:
            return notice
        raise ActionItemNoticeError(f"Notice {notice_id} not found.")

    return db.get_change_notice_by_id(notice_id)
