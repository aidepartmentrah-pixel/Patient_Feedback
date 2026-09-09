"""
Service layer for Source Management (Settings).
Validates inputs and calls the DB layer. Mirrors
classification_mgmt_service.py -- Source is a flat lookup with both
SourceName (EN) and SourceNameAr (AR) required (NOT NULL on the table),
unlike Category/Subcategory where English already existed and Arabic was
the new optional-until-filled-in column.
"""
from api.db_layer.source_mgmt_db import (
    get_sources_for_management,
    get_source_by_id,
    duplicate_exists,
    insert_source,
    update_source_names,
    set_source_active,
)


def list_sources() -> dict:
    try:
        rows = get_sources_for_management()
        sources = [
            {
                "id": r["SourceID"],
                "name_ar": r["SourceNameAr"],
                "name_en": r["SourceName"],
                "display_order": r["DisplayOrder"],
                "is_active": bool(r["IsActive"]),
            }
            for r in rows
        ]
        return {"sources": sources}
    except Exception as e:
        return {"sources": [], "error": str(e)}


def add_source(name_ar: str, name_en: str) -> dict:
    name_ar = (name_ar or "").strip()
    name_en = (name_en or "").strip()

    if not name_ar:
        return {"success": False, "error": "VALIDATION_ERROR", "message": "Arabic name is required."}
    if not name_en:
        return {"success": False, "error": "VALIDATION_ERROR", "message": "English name is required."}

    if duplicate_exists(name_ar, name_en):
        return {"success": False, "error": "DUPLICATE_NAME",
                "message": "A source with this name already exists."}

    new_id = insert_source(name_ar, name_en)
    return {"success": True, "source_id": new_id, "message": "Source added successfully."}


def update_source(source_id: int, name_ar: str, name_en: str) -> dict:
    name_ar = (name_ar or "").strip()
    name_en = (name_en or "").strip()

    if not name_ar:
        return {"success": False, "error": "VALIDATION_ERROR", "message": "Arabic name is required."}
    if not name_en:
        return {"success": False, "error": "VALIDATION_ERROR", "message": "English name is required."}

    existing = get_source_by_id(source_id)
    if not existing:
        return {"success": False, "error": "NOT_FOUND", "message": "Source not found."}

    if duplicate_exists(name_ar, name_en, exclude_id=source_id):
        return {"success": False, "error": "DUPLICATE_NAME",
                "message": "A source with this name already exists."}

    update_source_names(source_id, name_ar, name_en)
    return {"success": True, "message": "Source updated successfully."}


def freeze_source(source_id: int) -> dict:
    existing = get_source_by_id(source_id)
    if not existing:
        return {"success": False, "error": "NOT_FOUND", "message": "Source not found."}
    set_source_active(source_id, False)
    return {"success": True, "message": "Source frozen. It will not appear in new case forms."}


def unfreeze_source(source_id: int) -> dict:
    existing = get_source_by_id(source_id)
    if not existing:
        return {"success": False, "error": "NOT_FOUND", "message": "Source not found."}
    set_source_active(source_id, True)
    return {"success": True, "message": "Source unfrozen. It will appear in case forms again."}
