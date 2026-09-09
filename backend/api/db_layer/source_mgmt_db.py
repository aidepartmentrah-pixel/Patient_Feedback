"""
DB layer for Source Management (Settings).
Handles CRUD on APP_LOOKUP_SOURCE with IsActive support.
Mirrors classification_mgmt_db.py -- Source is a flat lookup (no parent
level to group under), so this is the simplest of the four *_mgmt_db
modules.
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


def get_sources_for_management() -> list[dict]:
    """Return all sources, all active states, in display order."""
    return _fetch_all(
        """
        SELECT SourceID, SourceName, SourceNameAr, DisplayOrder, IsActive
        FROM dbo.APP_LOOKUP_SOURCE
        ORDER BY DisplayOrder, SourceNameAr
        """
    )


def get_source_by_id(source_id: int) -> dict | None:
    rows = _fetch_all(
        "SELECT SourceID, SourceName, SourceNameAr, DisplayOrder, IsActive "
        "FROM dbo.APP_LOOKUP_SOURCE "
        "WHERE SourceID = ?",
        (source_id,),
    )
    return rows[0] if rows else None


def duplicate_exists(name_ar: str, name_en: str, exclude_id: int | None = None) -> bool:
    """Check if a source with the same AR or EN name already exists."""
    rows = _fetch_all(
        "SELECT SourceID FROM dbo.APP_LOOKUP_SOURCE "
        "WHERE (LOWER(SourceNameAr) = LOWER(?) OR LOWER(SourceName) = LOWER(?)) "
        "AND SourceID != COALESCE(?, -1)",
        (name_ar, name_en, exclude_id),
    )
    return len(rows) > 0


def get_next_display_order() -> int:
    rows = _fetch_all("SELECT ISNULL(MAX(DisplayOrder), 0) + 1 AS next_order FROM dbo.APP_LOOKUP_SOURCE")
    return rows[0]["next_order"] if rows else 1


def insert_source(name_ar: str, name_en: str) -> int:
    """Insert a new source and return its new SourceID. Both names are
    required -- unlike Category/Subcategory, SourceName and SourceNameAr are
    both NOT NULL on this table already."""
    order = get_next_display_order()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO dbo.APP_LOOKUP_SOURCE "
        "(SourceName, SourceNameAr, DisplayOrder, IsActive, CreatedAt, UpdatedAt) "
        "OUTPUT INSERTED.SourceID "
        "VALUES (?, ?, ?, 1, GETDATE(), GETDATE())",
        (name_en.strip(), name_ar.strip(), order),
    )
    row = cursor.fetchone()
    conn.commit()
    conn.close()
    return row[0]


def update_source_names(source_id: int, name_ar: str, name_en: str) -> None:
    _execute(
        "UPDATE dbo.APP_LOOKUP_SOURCE "
        "SET SourceNameAr = ?, SourceName = ?, UpdatedAt = GETDATE() "
        "WHERE SourceID = ?",
        (name_ar.strip(), name_en.strip(), source_id),
    )


def set_source_active(source_id: int, is_active: bool) -> None:
    _execute(
        "UPDATE dbo.APP_LOOKUP_SOURCE "
        "SET IsActive = ?, UpdatedAt = GETDATE() "
        "WHERE SourceID = ?",
        (1 if is_active else 0, source_id),
    )
