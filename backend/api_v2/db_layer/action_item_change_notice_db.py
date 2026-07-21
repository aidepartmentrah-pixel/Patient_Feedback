"""
Action Item Change Notice Database Layer (API V2)
Handles SQL operations for APP_SubcaseActionItemChangeNotice.

Part of Action Item Coordination — AIC-S7 Notification Foundation.
NO business logic. NO authorization. ONLY SQL operations.

One pending (unacknowledged) row per ActionItemID at a time, enforced by
UX_ActionItemChangeNotice_PendingPerItem. upsert_change_notice() updates
the existing pending row's "New*" fields (keeping the original "Old*"
baseline) if one already exists, otherwise inserts a new row.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, date
from core.database import get_connection


_SELECT_COLUMNS = """
    cn.NoticeID,
    cn.ActionItemID,
    cn.RecipientUserID,
    cn.OldTitle,
    cn.NewTitle,
    cn.OldDescription,
    cn.NewDescription,
    cn.OldDueDate,
    cn.NewDueDate,
    cn.ChangedByUserID,
    cn.ChangedAt,
    cn.AcknowledgedAt,
    cn.AcknowledgedByUserID,
    ai.Title AS CurrentActionItemTitle,
    ai.SubcaseID,
    cu.DisplayName AS ChangedByDisplayName
"""

_FROM_JOINS = """
    FROM dbo.APP_SubcaseActionItemChangeNotice cn
    LEFT JOIN dbo.APP_SubcaseActionItem ai ON cn.ActionItemID = ai.ActionItemID
    LEFT JOIN dbo.APP_Users cu ON cn.ChangedByUserID = cu.UserID
"""


def _row_to_dict(row) -> Dict[str, Any]:
    return {
        "notice_id": row.NoticeID,
        "action_item_id": row.ActionItemID,
        "recipient_user_id": row.RecipientUserID,
        "old_title": row.OldTitle,
        "new_title": row.NewTitle,
        "old_description": row.OldDescription,
        "new_description": row.NewDescription,
        "old_due_date": row.OldDueDate,
        "new_due_date": row.NewDueDate,
        "changed_by_user_id": row.ChangedByUserID,
        "changed_at": row.ChangedAt,
        "acknowledged_at": row.AcknowledgedAt,
        "acknowledged_by_user_id": row.AcknowledgedByUserID,
        "current_action_item_title": row.CurrentActionItemTitle,
        "subcase_id": row.SubcaseID,
        "changed_by_display_name": row.ChangedByDisplayName,
    }


def upsert_change_notice(
    action_item_id: int,
    recipient_user_id: int,
    old_title: str,
    new_title: str,
    old_description: Optional[str],
    new_description: Optional[str],
    old_due_date: Optional[date],
    new_due_date: Optional[date],
    changed_by_user_id: int
) -> Optional[int]:
    """
    Record a cross-user change to an Action Item as a pending notice.

    If an unacknowledged notice for this ActionItemID already exists,
    update its "New*" fields and ChangedBy/ChangedAt (the original "Old*"
    baseline from the first unacknowledged change is preserved). Otherwise
    insert a fresh row.

    Returns:
        NoticeID of the (possibly pre-existing) pending notice.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        update_query = """
            UPDATE dbo.APP_SubcaseActionItemChangeNotice
            SET NewTitle = ?,
                NewDescription = ?,
                NewDueDate = ?,
                ChangedByUserID = ?,
                ChangedAt = ?
            OUTPUT INSERTED.NoticeID
            WHERE ActionItemID = ? AND AcknowledgedAt IS NULL
        """
        cursor.execute(update_query, (
            new_title, new_description, new_due_date,
            changed_by_user_id, datetime.now(),
            action_item_id
        ))
        row = cursor.fetchone()

        if row is None:
            insert_query = """
                INSERT INTO dbo.APP_SubcaseActionItemChangeNotice (
                    ActionItemID, RecipientUserID,
                    OldTitle, NewTitle,
                    OldDescription, NewDescription,
                    OldDueDate, NewDueDate,
                    ChangedByUserID, ChangedAt
                )
                OUTPUT INSERTED.NoticeID
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(insert_query, (
                action_item_id, recipient_user_id,
                old_title, new_title,
                old_description, new_description,
                old_due_date, new_due_date,
                changed_by_user_id, datetime.now()
            ))
            row = cursor.fetchone()

        conn.commit()
        return row[0] if row else None

    finally:
        cursor.close()
        conn.close()


def acknowledge_change_notice(notice_id: int, acknowledged_by_user_id: int) -> bool:
    """
    Mark a change notice as acknowledged.

    Returns:
        True if this call set the acknowledgment, False if not found
        or already acknowledged.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = """
            UPDATE dbo.APP_SubcaseActionItemChangeNotice
            SET AcknowledgedAt = ?,
                AcknowledgedByUserID = ?
            WHERE NoticeID = ? AND AcknowledgedAt IS NULL
        """
        cursor.execute(query, (datetime.now(), acknowledged_by_user_id, notice_id))
        conn.commit()
        return cursor.rowcount > 0

    finally:
        cursor.close()
        conn.close()


def get_change_notice_by_id(notice_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single change notice by ID, with action item/user context joined in."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = f"""
            SELECT {_SELECT_COLUMNS}
            {_FROM_JOINS}
            WHERE cn.NoticeID = ?
        """
        cursor.execute(query, (notice_id,))
        row = cursor.fetchone()
        return _row_to_dict(row) if row else None

    finally:
        cursor.close()
        conn.close()


def get_unacknowledged_change_notices_for_user(user_id: int) -> List[Dict[str, Any]]:
    """Fetch all unacknowledged change notices addressed to this user."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = f"""
            SELECT {_SELECT_COLUMNS}
            {_FROM_JOINS}
            WHERE cn.RecipientUserID = ? AND cn.AcknowledgedAt IS NULL
            ORDER BY cn.ChangedAt DESC
        """
        cursor.execute(query, (user_id,))
        rows = cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    finally:
        cursor.close()
        conn.close()
