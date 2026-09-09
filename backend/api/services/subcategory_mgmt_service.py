"""
Service layer for Sub-Category Management (Settings).
Validates inputs and calls the DB layer. Mirrors
classification_mgmt_service.py one level up the hierarchy.
"""
from api.db_layer.subcategory_mgmt_db import (
    get_subcategories_for_management,
    get_subcategory_by_id,
    get_category_by_id,
    duplicate_exists,
    insert_subcategory,
    update_subcategory_names,
    set_subcategory_active,
)


def _group_by_category(rows: list[dict]) -> list[dict]:
    """Group flat subcategory rows into category buckets."""
    cats: dict[int, dict] = {}
    for row in rows:
        cid = row["CategoryID"]
        if cid not in cats:
            cats[cid] = {
                "category_id": cid,
                "category_name": row["CategoryName"],
                "category_name_ar": row["CategoryNameAr"],
                "domain_id": row["DomainID"],
                "domain_name": row["DomainName"],
                "subcategories": [],
            }
        cats[cid]["subcategories"].append({
            "id": row["SubCategoryID"],
            "name_ar": row["SubCategoryNameAr"],
            "name_en": row["SubCategoryName"],
            "is_active": bool(row["IsActive"]),
        })
    return list(cats.values())


def list_subcategories_grouped() -> dict:
    try:
        rows = get_subcategories_for_management()
        return {"categories": _group_by_category(rows)}
    except Exception as e:
        return {"categories": [], "error": str(e)}


def add_subcategory(category_id: int, name_ar: str, name_en: str | None) -> dict:
    name_ar = (name_ar or "").strip()
    name_en = (name_en or "").strip() or None

    if not name_ar:
        return {"success": False, "error": "VALIDATION_ERROR", "message": "Arabic name is required."}

    category = get_category_by_id(category_id)
    if not category:
        return {"success": False, "error": "INVALID_CATEGORY", "message": "Category not found."}

    if duplicate_exists(category_id, name_ar, name_en):
        return {"success": False, "error": "DUPLICATE_NAME",
                "message": "A subcategory with this name already exists under the same category."}

    new_id = insert_subcategory(category_id, name_ar, name_en)
    return {"success": True, "subcategory_id": new_id,
            "message": "Subcategory added successfully."}


def update_subcategory(subcategory_id: int, name_ar: str, name_en: str | None) -> dict:
    name_ar = (name_ar or "").strip()
    name_en = (name_en or "").strip() or None

    if not name_ar:
        return {"success": False, "error": "VALIDATION_ERROR", "message": "Arabic name is required."}

    existing = get_subcategory_by_id(subcategory_id)
    if not existing:
        return {"success": False, "error": "NOT_FOUND", "message": "Subcategory not found."}

    if duplicate_exists(existing["CategoryID"], name_ar, name_en, exclude_id=subcategory_id):
        return {"success": False, "error": "DUPLICATE_NAME",
                "message": "A subcategory with this name already exists under the same category."}

    update_subcategory_names(subcategory_id, name_ar, name_en)
    return {"success": True, "message": "Subcategory updated successfully."}


def freeze_subcategory(subcategory_id: int) -> dict:
    existing = get_subcategory_by_id(subcategory_id)
    if not existing:
        return {"success": False, "error": "NOT_FOUND", "message": "Subcategory not found."}
    set_subcategory_active(subcategory_id, False)
    return {"success": True, "message": "Subcategory frozen. It will not appear in new case forms."}


def unfreeze_subcategory(subcategory_id: int) -> dict:
    existing = get_subcategory_by_id(subcategory_id)
    if not existing:
        return {"success": False, "error": "NOT_FOUND", "message": "Subcategory not found."}
    set_subcategory_active(subcategory_id, True)
    return {"success": True, "message": "Subcategory unfrozen. It will appear in case forms again."}
