"""
RCA DB Layer
Raw SQL access for APP_RCAFactorCategory, APP_RCASuggestion,
and APP_SubcaseRCASuggestionSelection.
"""

from typing import List, Dict, Any, Optional
from core.database import get_connection


# ============================================================
# CATEGORIES
# ============================================================

def get_all_categories(active_only: bool = True) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    sql = """
        SELECT CategoryID, CategoryCode, CategoryNameEn, CategoryNameAr,
               SortOrder, IsActive, CreatedAt, UpdatedAt
        FROM dbo.APP_RCAFactorCategory
    """
    if active_only:
        sql += " WHERE IsActive = 1"
    sql += " ORDER BY SortOrder, CategoryID"
    cursor.execute(sql)
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    return [dict(zip(columns, row)) for row in rows]


def create_category(
    code: str,
    name_en: str,
    name_ar: Optional[str],
    sort_order: int,
    created_by: Optional[int]
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO dbo.APP_RCAFactorCategory
            (CategoryCode, CategoryNameEn, CategoryNameAr, SortOrder, IsActive, CreatedByUserID)
        OUTPUT INSERTED.CategoryID
        VALUES (?, ?, ?, ?, 1, ?)
        """,
        (code, name_en, name_ar, sort_order, created_by)
    )
    new_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()
    return new_id


def update_category(
    category_id: int,
    name_en: str,
    name_ar: Optional[str],
    sort_order: int,
    updated_by: Optional[int]
) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE dbo.APP_RCAFactorCategory
        SET CategoryNameEn = ?, CategoryNameAr = ?, SortOrder = ?,
            UpdatedAt = SYSUTCDATETIME(), UpdatedByUserID = ?
        WHERE CategoryID = ?
        """,
        (name_en, name_ar, sort_order, updated_by, category_id)
    )
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def set_category_active(
    category_id: int,
    is_active: bool,
    updated_by: Optional[int]
) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE dbo.APP_RCAFactorCategory
        SET IsActive = ?, UpdatedAt = SYSUTCDATETIME(), UpdatedByUserID = ?
        WHERE CategoryID = ?
        """,
        (1 if is_active else 0, updated_by, category_id)
    )
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


# ============================================================
# SUGGESTIONS
# ============================================================

def get_all_suggestions(
    active_only: bool = True,
    suggestion_type: Optional[str] = None,
    category_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    conditions = []
    params = []
    if active_only:
        conditions.append("s.IsActive = 1")
    if suggestion_type:
        conditions.append("s.SuggestionType = ?")
        params.append(suggestion_type)
    if category_id is not None:
        conditions.append("s.CategoryID = ?")
        params.append(category_id)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"""
        SELECT s.SuggestionID, s.CategoryID, c.CategoryNameEn, c.CategoryNameAr,
               s.SuggestionType, s.SuggestionTextEn, s.SuggestionTextAr,
               s.SortOrder, s.IsActive, s.CreatedAt, s.UpdatedAt
        FROM dbo.APP_RCASuggestion s
        JOIN dbo.APP_RCAFactorCategory c ON s.CategoryID = c.CategoryID
        {where_clause}
        ORDER BY c.SortOrder, s.SortOrder, s.SuggestionID
    """
    cursor.execute(sql, params)
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    return [dict(zip(columns, row)) for row in rows]


def get_suggestion_by_id(suggestion_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT SuggestionID, CategoryID, SuggestionType, SuggestionTextEn,
               SuggestionTextAr, SortOrder, IsActive, CreatedAt, UpdatedAt
        FROM dbo.APP_RCASuggestion
        WHERE SuggestionID = ?
        """,
        (suggestion_id,)
    )
    row = cursor.fetchone()
    columns = [col[0] for col in cursor.description] if cursor.description else []
    conn.close()
    return dict(zip(columns, row)) if row else None


def create_suggestion(
    category_id: int,
    suggestion_type: str,
    text_en: Optional[str],
    text_ar: str,
    sort_order: int,
    created_by: Optional[int]
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO dbo.APP_RCASuggestion
            (CategoryID, SuggestionType, SuggestionTextEn, SuggestionTextAr,
             SortOrder, IsActive, CreatedByUserID)
        OUTPUT INSERTED.SuggestionID
        VALUES (?, ?, ?, ?, ?, 1, ?)
        """,
        (category_id, suggestion_type, text_en, text_ar, sort_order, created_by)
    )
    new_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()
    return new_id


def update_suggestion(
    suggestion_id: int,
    category_id: int,
    suggestion_type: str,
    text_en: Optional[str],
    text_ar: str,
    sort_order: int,
    updated_by: Optional[int]
) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE dbo.APP_RCASuggestion
        SET CategoryID = ?, SuggestionType = ?, SuggestionTextEn = ?,
            SuggestionTextAr = ?, SortOrder = ?,
            UpdatedAt = SYSUTCDATETIME(), UpdatedByUserID = ?
        WHERE SuggestionID = ?
        """,
        (category_id, suggestion_type, text_en, text_ar, sort_order, updated_by, suggestion_id)
    )
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def set_suggestion_active(
    suggestion_id: int,
    is_active: bool,
    updated_by: Optional[int]
) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE dbo.APP_RCASuggestion
        SET IsActive = ?, UpdatedAt = SYSUTCDATETIME(), UpdatedByUserID = ?
        WHERE SuggestionID = ?
        """,
        (1 if is_active else 0, updated_by, suggestion_id)
    )
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


# ============================================================
# PAIRS: Cause <-> Action Item (PairedSuggestionID)
# ============================================================

def get_pairs(
    category_id: Optional[int] = None,
    active_only: bool = True
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    conditions = ["c.SuggestionType = 'CAUSE'", "a.SuggestionType = 'ACTION_ITEM'"]
    params: List[Any] = []
    if category_id is not None:
        conditions.append("c.CategoryID = ?")
        params.append(category_id)
    if active_only:
        conditions.append("c.IsActive = 1")
        conditions.append("a.IsActive = 1")

    where_clause = "WHERE " + " AND ".join(conditions)
    sql = f"""
        SELECT c.SuggestionID AS PairID, c.CategoryID,
               cat.CategoryNameEn, cat.CategoryNameAr,
               c.SuggestionTextAr AS CauseTextAr, c.SuggestionTextEn AS CauseTextEn,
               a.SuggestionID AS ActionSuggestionID,
               a.SuggestionTextAr AS ActionTextAr, a.SuggestionTextEn AS ActionTextEn,
               c.SortOrder, c.IsActive
        FROM dbo.APP_RCASuggestion c
        JOIN dbo.APP_RCASuggestion a ON a.SuggestionID = c.PairedSuggestionID
        JOIN dbo.APP_RCAFactorCategory cat ON cat.CategoryID = c.CategoryID
        {where_clause}
        ORDER BY c.CategoryID, c.SortOrder, c.SuggestionID
    """
    cursor.execute(sql, params)
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    return [dict(zip(columns, row)) for row in rows]


def create_pair(
    category_id: int,
    cause_text_ar: str,
    cause_text_en: Optional[str],
    action_text_ar: str,
    action_text_en: Optional[str],
    created_by: Optional[int]
) -> Dict[str, int]:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT ISNULL(MAX(SortOrder), -1) + 1
            FROM dbo.APP_RCASuggestion
            WHERE CategoryID = ? AND SuggestionType = 'CAUSE'
            """,
            (category_id,)
        )
        next_sort = cursor.fetchone()[0]

        cursor.execute(
            """
            INSERT INTO dbo.APP_RCASuggestion
                (CategoryID, SuggestionType, SuggestionTextEn, SuggestionTextAr,
                 SortOrder, IsActive, CreatedByUserID)
            OUTPUT INSERTED.SuggestionID
            VALUES (?, 'CAUSE', ?, ?, ?, 1, ?)
            """,
            (category_id, cause_text_en, cause_text_ar, next_sort, created_by)
        )
        cause_id = cursor.fetchone()[0]

        cursor.execute(
            """
            INSERT INTO dbo.APP_RCASuggestion
                (CategoryID, SuggestionType, SuggestionTextEn, SuggestionTextAr,
                 SortOrder, IsActive, CreatedByUserID)
            OUTPUT INSERTED.SuggestionID
            VALUES (?, 'ACTION_ITEM', ?, ?, ?, 1, ?)
            """,
            (category_id, action_text_en, action_text_ar, next_sort, created_by)
        )
        action_id = cursor.fetchone()[0]

        cursor.execute(
            "UPDATE dbo.APP_RCASuggestion SET PairedSuggestionID = ? WHERE SuggestionID = ?",
            (action_id, cause_id)
        )
        cursor.execute(
            "UPDATE dbo.APP_RCASuggestion SET PairedSuggestionID = ? WHERE SuggestionID = ?",
            (cause_id, action_id)
        )

        conn.commit()
        return {"pair_id": cause_id, "action_suggestion_id": action_id}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_pair(
    pair_id: int,
    cause_text_ar: str,
    cause_text_en: Optional[str],
    action_text_ar: str,
    action_text_en: Optional[str],
    updated_by: Optional[int]
) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT PairedSuggestionID FROM dbo.APP_RCASuggestion
            WHERE SuggestionID = ? AND SuggestionType = 'CAUSE'
            """,
            (pair_id,)
        )
        row = cursor.fetchone()
        if row is None or row[0] is None:
            return False
        action_id = row[0]

        cursor.execute(
            """
            UPDATE dbo.APP_RCASuggestion
            SET SuggestionTextAr = ?, SuggestionTextEn = ?,
                UpdatedAt = SYSUTCDATETIME(), UpdatedByUserID = ?
            WHERE SuggestionID = ?
            """,
            (cause_text_ar, cause_text_en, updated_by, pair_id)
        )
        cursor.execute(
            """
            UPDATE dbo.APP_RCASuggestion
            SET SuggestionTextAr = ?, SuggestionTextEn = ?,
                UpdatedAt = SYSUTCDATETIME(), UpdatedByUserID = ?
            WHERE SuggestionID = ?
            """,
            (action_text_ar, action_text_en, updated_by, action_id)
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def set_pair_active(
    pair_id: int,
    is_active: bool,
    updated_by: Optional[int]
) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT PairedSuggestionID FROM dbo.APP_RCASuggestion
            WHERE SuggestionID = ? AND SuggestionType = 'CAUSE'
            """,
            (pair_id,)
        )
        row = cursor.fetchone()
        if row is None or row[0] is None:
            return False
        action_id = row[0]

        cursor.execute(
            """
            UPDATE dbo.APP_RCASuggestion
            SET IsActive = ?, UpdatedAt = SYSUTCDATETIME(), UpdatedByUserID = ?
            WHERE SuggestionID IN (?, ?)
            """,
            (1 if is_active else 0, updated_by, pair_id, action_id)
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ============================================================
# INBOX: suggestions grouped by category (with is_selected flag)
# ============================================================

def get_suggestions_grouped_by_category(subcase_id: int) -> List[Dict[str, Any]]:
    """
    Returns active suggestions grouped by category.
    Each suggestion carries an is_selected flag based on current
    selections for the given subcase.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            c.CategoryID,
            c.CategoryNameEn,
            c.CategoryNameAr,
            c.SortOrder AS CategorySortOrder,
            s.SuggestionID,
            s.SuggestionType,
            s.SuggestionTextEn,
            s.SuggestionTextAr,
            s.SortOrder AS SuggestionSortOrder,
            CASE WHEN sel.SuggestionID IS NOT NULL THEN 1 ELSE 0 END AS IsSelected
        FROM dbo.APP_RCAFactorCategory c
        JOIN dbo.APP_RCASuggestion s ON s.CategoryID = c.CategoryID
        LEFT JOIN dbo.APP_SubcaseRCASuggestionSelection sel
            ON sel.SuggestionID = s.SuggestionID
            AND sel.SubcaseID = ?
        WHERE c.IsActive = 1
          AND s.IsActive = 1
        ORDER BY c.SortOrder, c.CategoryID, s.SortOrder, s.SuggestionID
        """,
        (subcase_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    categories: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        (cat_id, cat_name_en, cat_name_ar, cat_sort,
         sug_id, sug_type, sug_text_en, sug_text_ar,
         sug_sort, is_selected) = row

        if cat_id not in categories:
            categories[cat_id] = {
                "category_id": cat_id,
                "category_name_en": cat_name_en,
                "category_name_ar": cat_name_ar,
                "sort_order": cat_sort,
                "causes": [],
                "action_items": [],
            }

        item = {
            "suggestion_id": sug_id,
            "text_en": sug_text_en,
            "text_ar": sug_text_ar,
            "sort_order": sug_sort,
            "is_selected": bool(is_selected),
        }
        if sug_type == "CAUSE":
            categories[cat_id]["causes"].append(item)
        else:
            categories[cat_id]["action_items"].append(item)

    return list(categories.values())


def get_pairs_grouped_by_category(subcase_id: int) -> List[Dict[str, Any]]:
    """
    Returns active Cause/Corrective-Action pairs grouped by category.
    Each pair carries an is_selected flag based on current selections
    for the given subcase (true if either half of the pair is selected).
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            cat.CategoryID, cat.CategoryNameEn, cat.CategoryNameAr,
            cat.SortOrder AS CategorySortOrder,
            c.SuggestionID AS PairID, a.SuggestionID AS ActionSuggestionID,
            c.SuggestionTextAr AS CauseTextAr, c.SuggestionTextEn AS CauseTextEn,
            a.SuggestionTextAr AS ActionTextAr, a.SuggestionTextEn AS ActionTextEn,
            c.SortOrder AS PairSortOrder,
            CASE WHEN selc.SuggestionID IS NOT NULL OR sela.SuggestionID IS NOT NULL
                 THEN 1 ELSE 0 END AS IsSelected
        FROM dbo.APP_RCAFactorCategory cat
        JOIN dbo.APP_RCASuggestion c ON c.CategoryID = cat.CategoryID AND c.SuggestionType = 'CAUSE'
        JOIN dbo.APP_RCASuggestion a ON a.SuggestionID = c.PairedSuggestionID AND a.SuggestionType = 'ACTION_ITEM'
        LEFT JOIN dbo.APP_SubcaseRCASuggestionSelection selc
            ON selc.SuggestionID = c.SuggestionID AND selc.SubcaseID = ?
        LEFT JOIN dbo.APP_SubcaseRCASuggestionSelection sela
            ON sela.SuggestionID = a.SuggestionID AND sela.SubcaseID = ?
        WHERE cat.IsActive = 1 AND c.IsActive = 1 AND a.IsActive = 1
        ORDER BY cat.SortOrder, cat.CategoryID, c.SortOrder, c.SuggestionID
        """,
        (subcase_id, subcase_id)
    )
    rows = cursor.fetchall()
    conn.close()

    categories: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        (cat_id, cat_name_en, cat_name_ar, cat_sort,
         pair_id, action_id, cause_ar, cause_en, action_ar, action_en,
         pair_sort, is_selected) = row

        if cat_id not in categories:
            categories[cat_id] = {
                "category_id": cat_id,
                "category_name_en": cat_name_en,
                "category_name_ar": cat_name_ar,
                "sort_order": cat_sort,
                "pairs": [],
            }

        categories[cat_id]["pairs"].append({
            "pair_id": pair_id,
            "action_suggestion_id": action_id,
            "cause_text_ar": cause_ar,
            "cause_text_en": cause_en,
            "action_text_ar": action_ar,
            "action_text_en": action_en,
            "sort_order": pair_sort,
            "is_selected": bool(is_selected),
        })

    return list(categories.values())


# ============================================================
# SELECTIONS: read / save
# ============================================================

def get_selections_for_subcase(subcase_id: int) -> List[int]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT SuggestionID FROM dbo.APP_SubcaseRCASuggestionSelection WHERE SubcaseID = ?",
        (subcase_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


def save_selections_for_subcase(
    subcase_id: int,
    suggestion_ids: List[int],
    selected_by_user_id: Optional[int]
) -> bool:
    """
    Idempotent save: deletes existing selections then inserts new ones.
    Empty suggestion_ids list clears all selections for the subcase.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM dbo.APP_SubcaseRCASuggestionSelection WHERE SubcaseID = ?",
            (subcase_id,)
        )
        for sid in suggestion_ids:
            cursor.execute(
                """
                INSERT INTO dbo.APP_SubcaseRCASuggestionSelection
                    (SubcaseID, SuggestionID, SelectedByUserID)
                VALUES (?, ?, ?)
                """,
                (subcase_id, sid, selected_by_user_id)
            )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
