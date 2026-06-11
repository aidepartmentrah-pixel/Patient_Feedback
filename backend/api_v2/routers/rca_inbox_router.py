"""
RCA Inbox Router
Endpoints for the section inbox to read and save RCA suggestion selections.
"""

from fastapi import APIRouter, Path, HTTPException, Depends
from typing import List
from pydantic import BaseModel

from backend.api.dependencies.user_context import get_current_user
from backend.api.schemas.auth_models import CurrentUser
from backend.api.utils.guards import require_logged_in
from backend.api_v2.db_layer import rca_db
from backend.api_v2.db_layer import administrative_subcase_db


router = APIRouter(
    prefix="/api/v2/administrative-subcases",
    tags=["RCA Inbox"]
)


# ==================== REQUEST MODELS ====================

class RcaSelectionsRequest(BaseModel):
    selected_suggestion_ids: List[int] = []


# ==================== HELPERS ====================

def _load_subcase_or_404(subcase_id: int) -> dict:
    subcase = administrative_subcase_db.get_subcase_by_id(subcase_id)
    if subcase is None:
        raise HTTPException(status_code=404, detail={
            "error": "not_found",
            "message": f"Subcase {subcase_id} not found",
            "message_ar": "الحالة الفرعية غير موجودة"
        })
    return subcase


def _check_rca_write_permission(subcase: dict, current_user: CurrentUser) -> None:
    """
    Verify that current_user has write access to this subcase's target org unit.
    COMPLAINT_SUPERVISOR and SOFTWARE_ADMIN bypass scope check.
    """
    primary_role = (
        current_user.scopes[0].role_code
        if current_user.scopes else None
    )

    bypass_roles = {"COMPLAINT_SUPERVISOR", "SOFTWARE_ADMIN"}
    if primary_role in bypass_roles:
        return

    target_org_unit_id = subcase.get("target_org_unit_id")
    allowed_unit_ids = getattr(current_user, "allowed_unit_ids", None)

    if not allowed_unit_ids or target_org_unit_id not in allowed_unit_ids:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "forbidden",
                "message": "You do not have scope access to this subcase.",
                "message_ar": "ليس لديك صلاحية الوصول إلى هذه الحالة الفرعية"
            }
        )


# ==================== ENDPOINTS ====================

@router.get("/{subcase_id}/rca-suggestions")
async def get_rca_suggestions_for_subcase(
    subcase_id: int = Path(..., gt=0),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Return active RCA suggestions grouped by category.
    Each suggestion carries an is_selected flag for the given subcase's
    current selections so the frontend can pre-check them.
    """
    require_logged_in(current_user)
    _load_subcase_or_404(subcase_id)
    try:
        categories = rca_db.get_suggestions_grouped_by_category(subcase_id)
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "fetch_failed",
            "message": str(e),
            "message_ar": f"فشل جلب اقتراحات RCA: {str(e)}"
        })


@router.put("/{subcase_id}/rca-selections")
async def save_rca_selections(
    subcase_id: int = Path(..., gt=0),
    body: RcaSelectionsRequest = ...,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Save (replace) RCA suggestion selections for a subcase.
    Passing an empty list clears all existing selections.
    Idempotent: safe to call multiple times with the same IDs.
    """
    require_logged_in(current_user)
    subcase = _load_subcase_or_404(subcase_id)
    _check_rca_write_permission(subcase, current_user)

    # Deduplicate IDs (defensive)
    unique_ids = list(set(body.selected_suggestion_ids))

    try:
        rca_db.save_selections_for_subcase(
            subcase_id=subcase_id,
            suggestion_ids=unique_ids,
            selected_by_user_id=current_user.user_id
        )
        return {"success": True, "saved_count": len(unique_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "save_failed",
            "message": str(e),
            "message_ar": f"فشل حفظ اختيارات RCA: {str(e)}"
        })
