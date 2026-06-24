"""
RCA Settings Router
Endpoints for managing RCA factor categories and suggestions.
Only SOFTWARE_ADMIN / COMPLAINT_SUPERVISOR may write.
"""

from fastapi import APIRouter, Query, HTTPException, Depends, Path
from typing import Optional, List
from pydantic import BaseModel

from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_logged_in, require_software_admin
from backend.api_v2.db_layer import rca_db


router = APIRouter(prefix="/api/settings/rca", tags=["RCA Settings"])


# ==================== REQUEST MODELS ====================

class CategoryCreateRequest(BaseModel):
    category_code: str
    category_name_en: str
    category_name_ar: Optional[str] = None
    sort_order: int = 0


class CategoryUpdateRequest(BaseModel):
    category_name_en: str
    category_name_ar: Optional[str] = None
    sort_order: int = 0


class ActiveToggleRequest(BaseModel):
    is_active: bool


class SuggestionCreateRequest(BaseModel):
    category_id: int
    suggestion_type: str
    suggestion_text_en: Optional[str] = None
    suggestion_text_ar: str
    sort_order: int = 0


class SuggestionUpdateRequest(BaseModel):
    category_id: int
    suggestion_type: str
    suggestion_text_en: Optional[str] = None
    suggestion_text_ar: str
    sort_order: int = 0


class PairCreateRequest(BaseModel):
    category_id: int
    cause_text_ar: str
    cause_text_en: Optional[str] = None
    cause_description_ar: Optional[str] = None
    action_text_ar: str
    action_text_en: Optional[str] = None


class PairUpdateRequest(BaseModel):
    cause_text_ar: str
    cause_text_en: Optional[str] = None
    cause_description_ar: Optional[str] = None
    action_text_ar: str
    action_text_en: Optional[str] = None


# ==================== CATEGORIES ====================

@router.get("/categories")
async def list_rca_categories(
    active_only: bool = Query(True, description="Filter by active status"),
    current_user: CurrentUser = Depends(get_current_user)
):
    require_logged_in(current_user)
    require_software_admin(current_user)
    try:
        return rca_db.get_all_categories(active_only=active_only)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "fetch_failed",
            "message": str(e),
            "message_ar": f"فشل جلب فئات RCA: {str(e)}"
        })


@router.post("/categories", status_code=201)
async def create_rca_category(
    body: CategoryCreateRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    require_logged_in(current_user)
    require_software_admin(current_user)
    try:
        new_id = rca_db.create_category(
            code=body.category_code,
            name_en=body.category_name_en,
            name_ar=body.category_name_ar,
            sort_order=body.sort_order,
            created_by=current_user.user_id
        )
        return {"category_id": new_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "create_failed",
            "message": str(e),
            "message_ar": f"فشل إنشاء الفئة: {str(e)}"
        })


@router.put("/categories/{category_id}")
async def update_rca_category(
    category_id: int = Path(..., gt=0),
    body: CategoryUpdateRequest = ...,
    current_user: CurrentUser = Depends(get_current_user)
):
    require_logged_in(current_user)
    require_software_admin(current_user)
    try:
        ok = rca_db.update_category(
            category_id=category_id,
            name_en=body.category_name_en,
            name_ar=body.category_name_ar,
            sort_order=body.sort_order,
            updated_by=current_user.user_id
        )
        if not ok:
            raise HTTPException(status_code=404, detail={
                "error": "not_found",
                "message": f"Category {category_id} not found",
                "message_ar": "الفئة غير موجودة"
            })
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "update_failed",
            "message": str(e),
            "message_ar": f"فشل تحديث الفئة: {str(e)}"
        })


@router.patch("/categories/{category_id}/active")
async def toggle_rca_category_active(
    category_id: int = Path(..., gt=0),
    body: ActiveToggleRequest = ...,
    current_user: CurrentUser = Depends(get_current_user)
):
    require_logged_in(current_user)
    require_software_admin(current_user)
    try:
        ok = rca_db.set_category_active(
            category_id=category_id,
            is_active=body.is_active,
            updated_by=current_user.user_id
        )
        if not ok:
            raise HTTPException(status_code=404, detail={
                "error": "not_found",
                "message": f"Category {category_id} not found",
                "message_ar": "الفئة غير موجودة"
            })
        return {"success": True, "is_active": body.is_active}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "toggle_failed",
            "message": str(e),
            "message_ar": f"فشل تغيير حالة الفئة: {str(e)}"
        })


# ==================== SUGGESTIONS ====================

@router.get("/suggestions")
async def list_rca_suggestions(
    active_only: bool = Query(True, description="Filter by active status"),
    suggestion_type: Optional[str] = Query(None, description="CAUSE or ACTION_ITEM"),
    category_id: Optional[int] = Query(None, description="Filter by category"),
    current_user: CurrentUser = Depends(get_current_user)
):
    require_logged_in(current_user)
    require_software_admin(current_user)
    if suggestion_type and suggestion_type not in ("CAUSE", "ACTION_ITEM"):
        raise HTTPException(status_code=400, detail={
            "error": "invalid_type",
            "message": "suggestion_type must be CAUSE or ACTION_ITEM",
            "message_ar": "نوع الاقتراح يجب أن يكون CAUSE أو ACTION_ITEM"
        })
    try:
        return rca_db.get_all_suggestions(
            active_only=active_only,
            suggestion_type=suggestion_type,
            category_id=category_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "fetch_failed",
            "message": str(e),
            "message_ar": f"فشل جلب الاقتراحات: {str(e)}"
        })


@router.post("/suggestions", status_code=201)
async def create_rca_suggestion(
    body: SuggestionCreateRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    require_logged_in(current_user)
    require_software_admin(current_user)
    if body.suggestion_type not in ("CAUSE", "ACTION_ITEM"):
        raise HTTPException(status_code=400, detail={
            "error": "invalid_type",
            "message": "suggestion_type must be CAUSE or ACTION_ITEM",
            "message_ar": "نوع الاقتراح يجب أن يكون CAUSE أو ACTION_ITEM"
        })
    try:
        new_id = rca_db.create_suggestion(
            category_id=body.category_id,
            suggestion_type=body.suggestion_type,
            text_en=body.suggestion_text_en,
            text_ar=body.suggestion_text_ar,
            sort_order=body.sort_order,
            created_by=current_user.user_id
        )
        return {"suggestion_id": new_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "create_failed",
            "message": str(e),
            "message_ar": f"فشل إنشاء الاقتراح: {str(e)}"
        })


@router.put("/suggestions/{suggestion_id}")
async def update_rca_suggestion(
    suggestion_id: int = Path(..., gt=0),
    body: SuggestionUpdateRequest = ...,
    current_user: CurrentUser = Depends(get_current_user)
):
    require_logged_in(current_user)
    require_software_admin(current_user)
    if body.suggestion_type not in ("CAUSE", "ACTION_ITEM"):
        raise HTTPException(status_code=400, detail={
            "error": "invalid_type",
            "message": "suggestion_type must be CAUSE or ACTION_ITEM",
            "message_ar": "نوع الاقتراح يجب أن يكون CAUSE أو ACTION_ITEM"
        })
    try:
        ok = rca_db.update_suggestion(
            suggestion_id=suggestion_id,
            category_id=body.category_id,
            suggestion_type=body.suggestion_type,
            text_en=body.suggestion_text_en,
            text_ar=body.suggestion_text_ar,
            sort_order=body.sort_order,
            updated_by=current_user.user_id
        )
        if not ok:
            raise HTTPException(status_code=404, detail={
                "error": "not_found",
                "message": f"Suggestion {suggestion_id} not found",
                "message_ar": "الاقتراح غير موجود"
            })
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "update_failed",
            "message": str(e),
            "message_ar": f"فشل تحديث الاقتراح: {str(e)}"
        })


@router.patch("/suggestions/{suggestion_id}/active")
async def toggle_rca_suggestion_active(
    suggestion_id: int = Path(..., gt=0),
    body: ActiveToggleRequest = ...,
    current_user: CurrentUser = Depends(get_current_user)
):
    require_logged_in(current_user)
    require_software_admin(current_user)
    try:
        ok = rca_db.set_suggestion_active(
            suggestion_id=suggestion_id,
            is_active=body.is_active,
            updated_by=current_user.user_id
        )
        if not ok:
            raise HTTPException(status_code=404, detail={
                "error": "not_found",
                "message": f"Suggestion {suggestion_id} not found",
                "message_ar": "الاقتراح غير موجود"
            })
        return {"success": True, "is_active": body.is_active}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "toggle_failed",
            "message": str(e),
            "message_ar": f"فشل تغيير حالة الاقتراح: {str(e)}"
        })


# ==================== PAIRS (Cause <-> Corrective Action) ====================

@router.get("/pairs")
async def list_rca_pairs(
    category_id: Optional[int] = Query(None, description="Filter by category"),
    active_only: bool = Query(False, description="Filter by active status"),
    current_user: CurrentUser = Depends(get_current_user)
):
    require_logged_in(current_user)
    require_software_admin(current_user)
    try:
        return rca_db.get_pairs(category_id=category_id, active_only=active_only)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "fetch_failed",
            "message": str(e),
            "message_ar": f"فشل جلب الأزواج: {str(e)}"
        })


@router.post("/pairs", status_code=201)
async def create_rca_pair(
    body: PairCreateRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    require_logged_in(current_user)
    require_software_admin(current_user)
    try:
        result = rca_db.create_pair(
            category_id=body.category_id,
            cause_text_ar=body.cause_text_ar,
            cause_text_en=body.cause_text_en,
            action_text_ar=body.action_text_ar,
            action_text_en=body.action_text_en,
            created_by=current_user.user_id,
            cause_description_ar=body.cause_description_ar
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "create_failed",
            "message": str(e),
            "message_ar": f"فشل إنشاء الزوج: {str(e)}"
        })


@router.put("/pairs/{pair_id}")
async def update_rca_pair(
    pair_id: int = Path(..., gt=0),
    body: PairUpdateRequest = ...,
    current_user: CurrentUser = Depends(get_current_user)
):
    require_logged_in(current_user)
    require_software_admin(current_user)
    try:
        ok = rca_db.update_pair(
            pair_id=pair_id,
            cause_text_ar=body.cause_text_ar,
            cause_text_en=body.cause_text_en,
            action_text_ar=body.action_text_ar,
            action_text_en=body.action_text_en,
            updated_by=current_user.user_id,
            cause_description_ar=body.cause_description_ar
        )
        if not ok:
            raise HTTPException(status_code=404, detail={
                "error": "not_found",
                "message": f"Pair {pair_id} not found",
                "message_ar": "الزوج غير موجود"
            })
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "update_failed",
            "message": str(e),
            "message_ar": f"فشل تحديث الزوج: {str(e)}"
        })


@router.patch("/pairs/{pair_id}/active")
async def toggle_rca_pair_active(
    pair_id: int = Path(..., gt=0),
    body: ActiveToggleRequest = ...,
    current_user: CurrentUser = Depends(get_current_user)
):
    require_logged_in(current_user)
    require_software_admin(current_user)
    try:
        ok = rca_db.set_pair_active(
            pair_id=pair_id,
            is_active=body.is_active,
            updated_by=current_user.user_id
        )
        if not ok:
            raise HTTPException(status_code=404, detail={
                "error": "not_found",
                "message": f"Pair {pair_id} not found",
                "message_ar": "الزوج غير موجود"
            })
        return {"success": True, "is_active": body.is_active}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "toggle_failed",
            "message": str(e),
            "message_ar": f"فشل تغيير حالة الزوج: {str(e)}"
        })
