"""
DB layer for Category Management (Settings).
Handles CRUD on APP_LOOKUP_CATEGORY with IsActive support.
Mirrors classification_mgmt_db.py one level up the hierarchy
(Category, grouped under Domain, instead of Classification under
Sub-Category).
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


def get_domain_by_id(domain_id: int) -> dict | None:
    rows = _fetch_all(
        "SELECT DomainID, DomainName "
        "FROM dbo.APP_LOOKUP_DOMAIN "
        "WHERE DomainID = ?",
        (domain_id,),
    )
    return rows[0] if rows else None


def get_categories_for_management() -> list[dict]:
    """Return all categories with their domain, all active states."""
    return _fetch_all(
        """
        SELECT
            c.CategoryID,
            c.DomainID,
            c.CategoryName,
            c.CategoryNameAr,
            c.CategoryOrder,
            c.IsActive,
            d.DomainName
        FROM dbo.APP_LOOKUP_CATEGORY c
        JOIN dbo.APP_LOOKUP_DOMAIN d ON d.DomainID = c.DomainID
        ORDER BY d.DomainName, c.CategoryName
        """
    )


def get_category_by_id(category_id: int) -> dict | None:
    rows = _fetch_all(
        "SELECT CategoryID, DomainID, CategoryName, CategoryNameAr, CategoryOrder, IsActive "
        "FROM dbo.APP_LOOKUP_CATEGORY "
        "WHERE CategoryID = ?",
        (category_id,),
    )
    return rows[0] if rows else None


def duplicate_exists(domain_id: int, name_ar: str, name_en: str | None, exclude_id: int | None = None) -> bool:
    """Check if a category with the same AR or EN name already exists under the same domain."""
    if name_en:
        rows = _fetch_all(
            "SELECT CategoryID FROM dbo.APP_LOOKUP_CATEGORY "
            "WHERE DomainID = ? "
            "AND (LOWER(CategoryNameAr) = LOWER(?) OR LOWER(CategoryName) = LOWER(?)) "
            "AND CategoryID != COALESCE(?, -1)",
            (domain_id, name_ar, name_en, exclude_id),
        )
    else:
        rows = _fetch_all(
            "SELECT CategoryID FROM dbo.APP_LOOKUP_CATEGORY "
            "WHERE DomainID = ? "
            "AND LOWER(CategoryNameAr) = LOWER(?) "
            "AND CategoryID != COALESCE(?, -1)",
            (domain_id, name_ar, exclude_id),
        )
    return len(rows) > 0


def get_next_category_order(domain_id: int) -> int:
    rows = _fetch_all(
        "SELECT ISNULL(MAX(CategoryOrder), 0) + 1 AS next_order "
        "FROM dbo.APP_LOOKUP_CATEGORY WHERE DomainID = ?",
        (domain_id,),
    )
    return rows[0]["next_order"] if rows else 1


def insert_category(domain_id: int, name_ar: str, name_en: str | None) -> int:
    """Insert a new category and return its new CategoryID."""
    order = get_next_category_order(domain_id)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO dbo.APP_LOOKUP_CATEGORY "
        "(DomainID, CategoryName, CategoryNameAr, CategoryOrder, IsActive) "
        "OUTPUT INSERTED.CategoryID "
        "VALUES (?, ?, ?, ?, 1)",
        (domain_id, (name_en.strip() if name_en else name_ar.strip()), name_ar.strip(), order),
    )
    row = cursor.fetchone()
    conn.commit()
    conn.close()
    return row[0]


def update_category_names(category_id: int, name_ar: str, name_en: str | None) -> None:
    _execute(
        "UPDATE dbo.APP_LOOKUP_CATEGORY "
        "SET CategoryNameAr = ?, CategoryName = COALESCE(?, CategoryName) "
        "WHERE CategoryID = ?",
        (name_ar.strip(), name_en.strip() if name_en else None, category_id),
    )


def set_category_active(category_id: int, is_active: bool) -> None:
    _execute(
        "UPDATE dbo.APP_LOOKUP_CATEGORY "
        "SET IsActive = ? "
        "WHERE CategoryID = ?",
        (1 if is_active else 0, category_id),
    )
