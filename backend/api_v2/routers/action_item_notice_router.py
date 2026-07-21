"""
Action Item Change Notice Router (API v2)
Action Item Coordination — AIC-S7 Notification Foundation.

Lets the recipient of a "your suggested action item was changed" inbox
notification acknowledge it. Notices themselves are created automatically
by case_response_service when a higher-hierarchy edit changes a lower
level's action item — there is no creation endpoint here.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List

from api.dependencies.user_context import get_current_user
from api.schemas.auth_models import CurrentUser
from api_v2.services import action_item_notice_service as service


router = APIRouter(prefix="/api/v2/action-item-notices", tags=["Action Item Change Notices"])


@router.get("/unacknowledged")
def list_unacknowledged_notices(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, List[Dict[str, Any]]]:
    """List pending action-item change notices addressed to the current user's inbox."""
    notices = service.list_unacknowledged_for_user(current_user)
    return {"notices": notices}


@router.post("/{notice_id}/acknowledge")
def acknowledge_change_notice(
    notice_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Acknowledge a pending action-item change notice.

    Pure notification side-channel — does not change the underlying action
    item. Authorization: the recipient only.
    """
    try:
        notice = service.acknowledge_change_notice(current_user, notice_id)
        return {"notice": notice}
    except service.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except service.ForbiddenError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except service.ActionItemNoticeError as e:
        raise HTTPException(status_code=400, detail=str(e))
