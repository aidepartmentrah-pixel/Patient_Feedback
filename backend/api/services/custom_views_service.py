from datetime import datetime
from typing import Any, Dict, List, Optional

from ..db_layer.custom_views import (
    create_custom_view as db_create,
    get_custom_view as db_get,
    list_custom_views as db_list,
    update_custom_view as db_update,
    deactivate_custom_view as db_deactivate,
)


SHOW_FIELDS = [
    "ShowIncidentRequestCaseID",
    "ShowComplaintText",
    "ShowImmediateAction",
    "ShowTakenAction",
    "ShowFeedbackRecievedDate",
    "ShowPatientName",
    "ShowIssuingOrgUnitID",
    "ShowCreatedAt",
    "ShowCreatedByUserID",
    "ShowIsInPatient",
    "ShowClinicalRiskTypeID",
    "ShowFeedbackIntentTypeID",
    "ShowBuildingID",
    "ShowDomainID",
    "ShowCategoryID",
    "ShowSubCategoryID",
    "ShowClassificationID",
    "ShowSeverityID",
    "ShowStageID",
    "ShowHarmLevelID",
    "ShowCaseStatusID",
    "ShowSourceID",
    "ShowExplanationStatusID",
]


def _normalize_flags(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(payload)
    for f in SHOW_FIELDS:
        data[f] = bool(data.get(f, False))
    return data


def create_view(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload.get("ViewName"):
        return {"success": False, "error": "VALIDATION_ERROR", "message": "ViewName is required"}

    data = _normalize_flags(payload)

    if not any(data.get(f, False) for f in SHOW_FIELDS):
        return {"success": False, "error": "VALIDATION_ERROR", "message": "At least one column must be selected"}

    view_id = db_create({
        **data,
        "CreatedAt": datetime.now(),
        "CreatedByUserID": payload.get("CreatedByUserID", 1),
        "IsActive": payload.get("IsActive", True),
    })
    view = db_get(view_id)
    return {"success": True, "id": view_id, "view": view}


def get_view(view_id: int) -> Dict[str, Any]:
    view = db_get(view_id)
    if not view:
        return {"success": False, "error": "NOT_FOUND", "message": f"View {view_id} not found"}
    return {"success": True, "view": view}


def list_views(active_only: bool = True) -> Dict[str, Any]:
    items = db_list(active_only=active_only)
    return {"success": True, "total": len(items), "views": items}


def update_view(view_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not db_get(view_id):
        return {"success": False, "error": "NOT_FOUND", "message": f"View {view_id} not found"}

    updates = _normalize_flags(payload)
    if "ViewName" in payload and not payload["ViewName"]:
        return {"success": False, "error": "VALIDATION_ERROR", "message": "ViewName cannot be empty"}

    # If flags provided, ensure at least one is true
    if any(k in payload for k in SHOW_FIELDS) and not any(updates.get(f, False) for f in SHOW_FIELDS):
        return {"success": False, "error": "VALIDATION_ERROR", "message": "At least one column must be selected"}

    db_update(view_id, updates)
    return get_view(view_id)


def delete_view(view_id: int, hard: bool = False) -> Dict[str, Any]:
    if not db_get(view_id):
        return {"success": False, "error": "NOT_FOUND", "message": f"View {view_id} not found"}

    if hard:
        # Prefer soft delete; hard delete can be implemented if needed
        from ..db_layer.custom_views import delete_custom_view as db_delete
        db_delete(view_id)
        return {"success": True, "deleted": True, "hard": True}

    db_deactivate(view_id)
    return {"success": True, "deleted": True, "hard": False}
