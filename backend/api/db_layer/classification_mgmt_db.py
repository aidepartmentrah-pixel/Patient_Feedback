"""
DB layer for Classification Management (Settings).
Handles CRUD on APP_LOOKUP_CLASSIFICATION with IsActive support.
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


def get_subcategory_by_id(subcategory_id: int) -> dict | None:
    rows = _fetch_all(
        "SELECT SubCategoryID, CategoryID, SubCategoryName "
        "FROM dbo.APP_LOOKUP_SUBCATEGORY "
        "WHERE SubCategoryID = ?",
        (subcategory_id,),
    )
    return rows[0] if rows else None


def get_classifications_for_management() -> list[dict]:
    """Return all classifications with their full hierarchy path, all active states."""
    return _fetch_all(
        """
        SELECT
            c.ClassificationID,
            c.SubCategoryID,
            c.Classification_AR,
            c.Classification_EN,
            c.IsActive,
            sc.SubCategoryName,
            sc.CategoryID,
            cat.CategoryName,
            cat.DomainID,
            d.DomainName
        FROM dbo.APP_LOOKUP_CLASSIFICATION c
        JOIN dbo.APP_LOOKUP_SUBCATEGORY sc ON sc.SubCategoryID = c.SubCategoryID
        JOIN dbo.APP_LOOKUP_CATEGORY cat ON cat.CategoryID = sc.CategoryID
        JOIN dbo.APP_LOOKUP_DOMAIN d ON d.DomainID = cat.DomainID
        ORDER BY d.DomainName, cat.CategoryName, sc.SubCategoryName, c.Classification_AR
        """
    )


def get_classification_by_id(classification_id: int) -> dict | None:
    rows = _fetch_all(
        "SELECT ClassificationID, SubCategoryID, Classification_AR, Classification_EN, IsActive "
        "FROM dbo.APP_LOOKUP_CLASSIFICATION "
        "WHERE ClassificationID = ?",
        (classification_id,),
    )
    return rows[0] if rows else None


def duplicate_exists(subcategory_id: int, name_ar: str, name_en: str | None, exclude_id: int | None = None) -> bool:
    """Check if a classification with the same AR or EN name already exists under the same subcategory."""
    if name_en:
        rows = _fetch_all(
            "SELECT ClassificationID FROM dbo.APP_LOOKUP_CLASSIFICATION "
            "WHERE SubCategoryID = ? "
            "AND (LOWER(Classification_AR) = LOWER(?) OR LOWER(Classification_EN) = LOWER(?)) "
            "AND ClassificationID != COALESCE(?, -1)",
            (subcategory_id, name_ar, name_en, exclude_id),
        )
    else:
        rows = _fetch_all(
            "SELECT ClassificationID FROM dbo.APP_LOOKUP_CLASSIFICATION "
            "WHERE SubCategoryID = ? "
            "AND LOWER(Classification_AR) = LOWER(?) "
            "AND ClassificationID != COALESCE(?, -1)",
            (subcategory_id, name_ar, exclude_id),
        )
    return len(rows) > 0


def insert_classification(subcategory_id: int, name_ar: str, name_en: str | None) -> int:
    """Insert a new classification and return its new ClassificationID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO dbo.APP_LOOKUP_CLASSIFICATION "
        "(SubCategoryID, Classification_AR, Classification_EN, IsActive) "
        "OUTPUT INSERTED.ClassificationID "
        "VALUES (?, ?, ?, 1)",
        (subcategory_id, name_ar.strip(), name_en.strip() if name_en else None),
    )
    row = cursor.fetchone()
    conn.commit()
    conn.close()
    return row[0]


def update_classification_names(classification_id: int, name_ar: str, name_en: str | None) -> None:
    _execute(
        "UPDATE dbo.APP_LOOKUP_CLASSIFICATION "
        "SET Classification_AR = ?, Classification_EN = ? "
        "WHERE ClassificationID = ?",
        (name_ar.strip(), name_en.strip() if name_en else None, classification_id),
    )


def set_classification_active(classification_id: int, is_active: bool) -> None:
    _execute(
        "UPDATE dbo.APP_LOOKUP_CLASSIFICATION "
        "SET IsActive = ? "
        "WHERE ClassificationID = ?",
        (1 if is_active else 0, classification_id),
    )
