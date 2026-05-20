"""
Service layer for Classification Management (Settings).
Validates inputs and calls the DB layer.
"""
from api.db_layer.classification_mgmt_db import (
    get_classifications_for_management,
    get_classification_by_id,
    get_subcategory_by_id,
    duplicate_exists,
    insert_classification,
    update_classification_names,
    set_classification_active,
)


def _group_by_subcategory(rows: list[dict]) -> list[dict]:
    """Group flat classification rows into subcategory buckets."""
    subcats: dict[int, dict] = {}
    for row in rows:
        sid = row["SubCategoryID"]
        if sid not in subcats:
            subcats[sid] = {
                "subcategory_id": sid,
                "subcategory_name": row["SubCategoryName"],
                "category_id": row["CategoryID"],
                "category_name": row["CategoryName"],
                "domain_id": row["DomainID"],
                "domain_name": row["DomainName"],
                "classifications": [],
            }
        subcats[sid]["classifications"].append({
            "id": row["ClassificationID"],
            "name_ar": row["Classification_AR"],
            "name_en": row["Classification_EN"],
            "is_active": bool(row["IsActive"]),
        })
    return list(subcats.values())


def list_classifications_grouped() -> dict:
    try:
        rows = get_classifications_for_management()
        return {"subcategories": _group_by_subcategory(rows)}
    except Exception as e:
        return {"subcategories": [], "error": str(e)}


def add_classification(subcategory_id: int, name_ar: str, name_en: str | None) -> dict:
    name_ar = (name_ar or "").strip()
    name_en = (name_en or "").strip() or None

    if not name_ar:
        return {"success": False, "error": "VALIDATION_ERROR", "message": "Arabic name is required."}

    subcat = get_subcategory_by_id(subcategory_id)
    if not subcat:
        return {"success": False, "error": "INVALID_SUBCATEGORY", "message": "Subcategory not found."}

    if duplicate_exists(subcategory_id, name_ar, name_en):
        return {"success": False, "error": "DUPLICATE_NAME",
                "message": "A classification with this name already exists under the same subcategory."}

    new_id = insert_classification(subcategory_id, name_ar, name_en)
    return {"success": True, "classification_id": new_id,
            "message": "Classification added successfully."}


def update_classification(classification_id: int, name_ar: str, name_en: str | None) -> dict:
    name_ar = (name_ar or "").strip()
    name_en = (name_en or "").strip() or None

    if not name_ar:
        return {"success": False, "error": "VALIDATION_ERROR", "message": "Arabic name is required."}

    existing = get_classification_by_id(classification_id)
    if not existing:
        return {"success": False, "error": "NOT_FOUND", "message": "Classification not found."}

    if duplicate_exists(existing["SubCategoryID"], name_ar, name_en, exclude_id=classification_id):
        return {"success": False, "error": "DUPLICATE_NAME",
                "message": "A classification with this name already exists under the same subcategory."}

    update_classification_names(classification_id, name_ar, name_en)
    return {"success": True, "message": "Classification updated successfully."}


def freeze_classification(classification_id: int) -> dict:
    existing = get_classification_by_id(classification_id)
    if not existing:
        return {"success": False, "error": "NOT_FOUND", "message": "Classification not found."}
    set_classification_active(classification_id, False)
    return {"success": True, "message": "Classification frozen. It will not appear in new case forms."}


def unfreeze_classification(classification_id: int) -> dict:
    existing = get_classification_by_id(classification_id)
    if not existing:
        return {"success": False, "error": "NOT_FOUND", "message": "Classification not found."}
    set_classification_active(classification_id, True)
    return {"success": True, "message": "Classification unfrozen. It will appear in case forms again."}
