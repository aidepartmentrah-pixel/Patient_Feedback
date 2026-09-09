"""
DB layer for Sub-Category Management (Settings).
Handles CRUD on APP_LOOKUP_SUBCATEGORY with IsActive support.
Mirrors classification_mgmt_db.py one level up the hierarchy
(Sub-Category, grouped under Category, instead of Classification
under Sub-Category).
"""
from core.database import get_connection


def _fetch_all(query: str, params: tuple = ()) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    conn.close()
    return [dict(zip(columns, row)) for row in rows]


def _execute(query: str, params: tuple = ()) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    conn.close()


def get_category_by_id(category_id: int) -> dict | None:
    rows = _fetch_all(
        "SELECT CategoryID, DomainID, CategoryName, CategoryNameAr "
        "FROM dbo.APP_LOOKUP_CATEGORY "
        "WHERE CategoryID = ?",
        (category_id,),
    )
    return rows[0] if rows else None


def get_subcategories_for_management() -> list[dict]:
    """Return all subcategories with their full hierarchy path, all active states."""
    return _fetch_all(
        """
        SELECT
            sc.SubCategoryID,
            sc.CategoryID,
            sc.SubCategoryName,
            sc.SubCategoryNameAr,
            sc.IsActive,
            cat.CategoryName,
            cat.CategoryNameAr,
            cat.DomainID,
            d.DomainName
        FROM dbo.APP_LOOKUP_SUBCATEGORY sc
        JOIN dbo.APP_LOOKUP_CATEGORY cat ON cat.CategoryID = sc.CategoryID
        JOIN dbo.APP_LOOKUP_DOMAIN d ON d.DomainID = cat.DomainID
        ORDER BY d.DomainName, cat.CategoryName, sc.SubCategoryName
        """
    )


def get_subcategory_by_id(subcategory_id: int) -> dict | None:
    rows = _fetch_all(
        "SELECT SubCategoryID, CategoryID, SubCategoryName, SubCategoryNameAr, IsActive "
        "FROM dbo.APP_LOOKUP_SUBCATEGORY "
        "WHERE SubCategoryID = ?",
        (subcategory_id,),
    )
    return rows[0] if rows else None


def duplicate_exists(category_id: int, name_ar: str, name_en: str | None, exclude_id: int | None = None) -> bool:
    """Check if a subcategory with the same AR or EN name already exists under the same category."""
    if name_en:
        rows = _fetch_all(
            "SELECT SubCategoryID FROM dbo.APP_LOOKUP_SUBCATEGORY "
            "WHERE CategoryID = ? "
            "AND (LOWER(SubCategoryNameAr) = LOWER(?) OR LOWER(SubCategoryName) = LOWER(?)) "
            "AND SubCategoryID != COALESCE(?, -1)",
            (category_id, name_ar, name_en, exclude_id),
        )
    else:
        rows = _fetch_all(
            "SELECT SubCategoryID FROM dbo.APP_LOOKUP_SUBCATEGORY "
            "WHERE CategoryID = ? "
            "AND LOWER(SubCategoryNameAr) = LOWER(?) "
            "AND SubCategoryID != COALESCE(?, -1)",
            (category_id, name_ar, exclude_id),
        )
    return len(rows) > 0


def insert_subcategory(category_id: int, name_ar: str, name_en: str | None) -> int:
    """Insert a new subcategory and return its new SubCategoryID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO dbo.APP_LOOKUP_SUBCATEGORY "
        "(CategoryID, SubCategoryName, SubCategoryNameAr, IsActive) "
        "OUTPUT INSERTED.SubCategoryID "
        "VALUES (?, ?, ?, 1)",
        (category_id, (name_en.strip() if name_en else name_ar.strip()), name_ar.strip()),
    )
    row = cursor.fetchone()
    conn.commit()
    conn.close()
    return row[0]


def update_subcategory_names(subcategory_id: int, name_ar: str, name_en: str | None) -> None:
    _execute(
        "UPDATE dbo.APP_LOOKUP_SUBCATEGORY "
        "SET SubCategoryNameAr = ?, SubCategoryName = COALESCE(?, SubCategoryName) "
        "WHERE SubCategoryID = ?",
        (name_ar.strip(), name_en.strip() if name_en else None, subcategory_id),
    )


def set_subcategory_active(subcategory_id: int, is_active: bool) -> None:
    _execute(
        "UPDATE dbo.APP_LOOKUP_SUBCATEGORY "
        "SET IsActive = ? "
        "WHERE SubCategoryID = ?",
        (1 if is_active else 0, subcategory_id),
    )
