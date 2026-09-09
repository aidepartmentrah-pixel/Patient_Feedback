"""
Service layer for Category Management (Settings).
Validates inputs and calls the DB layer. Mirrors
classification_mgmt_service.py one level up the hierarchy.
"""
from api.db_layer.category_mgmt_db import (
    get_categories_for_management,
    get_category_by_id,
    get_domain_by_id,
    duplicate_exists,
    insert_category,
    update_category_names,
    set_category_active,
)


def _group_by_domain(rows: list[dict]) -> list[dict]:
    """Group flat category rows into domain buckets."""
    domains: dict[int, dict] = {}
    for row in rows:
        did = row["DomainID"]
        if did not in domains:
            domains[did] = {
                "domain_id": did,
                "domain_name": row["DomainName"],
                "categories": [],
            }
        domains[did]["categories"].append({
            "id": row["CategoryID"],
            "name_ar": row["CategoryNameAr"],
            "name_en": row["CategoryName"],
            "is_active": bool(row["IsActive"]),
        })
    return list(domains.values())


def list_categories_grouped() -> dict:
    try:
        rows = get_categories_for_management()
        return {"domains": _group_by_domain(rows)}
    except Exception as e:
        return {"domains": [], "error": str(e)}


def add_category(domain_id: int, name_ar: str, name_en: str | None) -> dict:
    name_ar = (name_ar or "").strip()
    name_en = (name_en or "").strip() or None

    if not name_ar:
        return {"success": False, "error": "VALIDATION_ERROR", "message": "Arabic name is required."}

    domain = get_domain_by_id(domain_id)
    if not domain:
        return {"success": False, "error": "INVALID_DOMAIN", "message": "Domain not found."}

    if duplicate_exists(domain_id, name_ar, name_en):
        return {"success": False, "error": "DUPLICATE_NAME",
                "message": "A category with this name already exists under the same domain."}

    new_id = insert_category(domain_id, name_ar, name_en)
    return {"success": True, "category_id": new_id,
            "message": "Category added successfully."}


def update_category(category_id: int, name_ar: str, name_en: str | None) -> dict:
    name_ar = (name_ar or "").strip()
    name_en = (name_en or "").strip() or None

    if not name_ar:
        return {"success": False, "error": "VALIDATION_ERROR", "message": "Arabic name is required."}

    existing = get_category_by_id(category_id)
    if not existing:
        return {"success": False, "error": "NOT_FOUND", "message": "Category not found."}

    if duplicate_exists(existing["DomainID"], name_ar, name_en, exclude_id=category_id):
        return {"success": False, "error": "DUPLICATE_NAME",
                "message": "A category with this name already exists under the same domain."}

    update_category_names(category_id, name_ar, name_en)
    return {"success": True, "message": "Category updated successfully."}


def freeze_category(category_id: int) -> dict:
    existing = get_category_by_id(category_id)
    if not existing:
        return {"success": False, "error": "NOT_FOUND", "message": "Category not found."}
    set_category_active(category_id, False)
    return {"success": True, "message": "Category frozen. It will not appear in new case forms."}


def unfreeze_category(category_id: int) -> dict:
    existing = get_category_by_id(category_id)
    if not existing:
        return {"success": False, "error": "NOT_FOUND", "message": "Category not found."}
    set_category_active(category_id, True)
    return {"success": True, "message": "Category unfrozen. It will appear in case forms again."}
