"""
Supervisor Action Item Database Layer (API V2)
Handles SQL operations for APP_SupervisorActionItem / APP_SupervisorActionItemAuditLog.

Part of Action Item Coordination (Iteration 4, administrative scope).
NO business logic. NO authorization. ONLY SQL operations.

Deliberately separate from action_item_subcase_db.py — that table's status
machine is load-bearing for case-escalation gating and must not be touched here.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, date
from core.database import get_connection


# ============================================================
# CREATION / FETCH
# ============================================================

def create_supervisor_action_item(
    incident_request_case_id: int,
    target_org_unit_id: int,
    created_by_user_id: int,
    created_by_role_code: str,
    description: str,
    subcase_id: Optional[int] = None,
    target_user_id: Optional[int] = None,
    due_date: Optional[date] = None,
    initial_status: str = "PENDING"
) -> Optional[int]:
    """
    Create a new supervisor action item.

    Returns:
        ActionItemID if created, None on failure
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = """
            INSERT INTO dbo.APP_SupervisorActionItem (
                IncidentRequestCaseID,
                SubcaseID,
                TargetOrgUnitID,
                TargetUserID,
                CreatedByUserID,
                CreatedByRoleCode,
                Description,
                DueDate,
                Status,
                CreatedAt
            )
            OUTPUT INSERTED.ActionItemID
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        cursor.execute(query, (
            incident_request_case_id,
            subcase_id,
            target_org_unit_id,
            target_user_id,
            created_by_user_id,
            created_by_role_code,
            description,
            due_date,
            initial_status,
            datetime.now()
        ))

        row = cursor.fetchone()
        new_id = row[0] if row else None

        conn.commit()
        return new_id

    finally:
        cursor.close()
        conn.close()


_SELECT_COLUMNS = """
    ai.ActionItemID,
    ai.IncidentRequestCaseID,
    ai.SubcaseID,
    ai.TargetOrgUnitID,
    ai.TargetUserID,
    ai.CreatedByUserID,
    ai.CreatedByRoleCode,
    ai.Description,
    ai.DueDate,
    ai.Status,
    ai.CreatedAt,
    ai.CompletedAt,
    ai.CancelledAt,
    ai.UpdatedAt,
    ai.UpdatedByUserID,
    ai.AcknowledgedAt,
    ai.AcknowledgedByUserID,
    ou.Name AS TargetOrgUnitName,
    tu.DisplayName AS TargetUserDisplayName,
    cu.DisplayName AS CreatedByDisplayName,
    ic.ComplaintText AS CaseDescription,
    ic.PatientName,
    inc.incident_number AS IncidentNumber
"""

_FROM_JOINS = """
    FROM dbo.APP_SupervisorActionItem ai
    LEFT JOIN dbo.AdminsrationUnit ou ON ai.TargetOrgUnitID = ou.UniqueID
    LEFT JOIN dbo.APP_Users tu ON ai.TargetUserID = tu.UserID
    LEFT JOIN dbo.APP_Users cu ON ai.CreatedByUserID = cu.UserID
    LEFT JOIN dbo.APP_IncidentCase ic ON ai.IncidentRequestCaseID = ic.IncidentRequestCaseID
    LEFT JOIN dbo.APP_Incident inc ON ic.incident_id = inc.incident_id
"""


def _row_to_dict(row) -> Dict[str, Any]:
    return {
        "action_item_id": row.ActionItemID,
        "incident_request_case_id": row.IncidentRequestCaseID,
        "subcase_id": row.SubcaseID,
        "target_org_unit_id": row.TargetOrgUnitID,
        "target_user_id": row.TargetUserID,
        "created_by_user_id": row.CreatedByUserID,
        "created_by_role_code": row.CreatedByRoleCode,
        "description": row.Description,
        "due_date": row.DueDate,
        "status": row.Status,
        "created_at": row.CreatedAt,
        "completed_at": row.CompletedAt,
        "cancelled_at": row.CancelledAt,
        "updated_at": row.UpdatedAt,
        "updated_by_user_id": row.UpdatedByUserID,
        "acknowledged_at": row.AcknowledgedAt,
        "acknowledged_by_user_id": row.AcknowledgedByUserID,
        "target_org_unit_name": row.TargetOrgUnitName,
        "target_user_display_name": row.TargetUserDisplayName,
        "created_by_display_name": row.CreatedByDisplayName,
        "case_description": row.CaseDescription,
        "patient_name": row.PatientName,
        "incident_number": row.IncidentNumber,
    }


def get_supervisor_action_item_by_id(action_item_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single supervisor action item by ID, with org unit/user/case context joined in."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = f"""
            SELECT {_SELECT_COLUMNS}
            {_FROM_JOINS}
            WHERE ai.ActionItemID = ?
        """
        cursor.execute(query, (action_item_id,))
        row = cursor.fetchone()
        return _row_to_dict(row) if row else None

    finally:
        cursor.close()
        conn.close()


def get_supervisor_action_items_by_org_units(org_unit_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Fetch supervisor action items targeted at any of the given org units.
    Caller passes current_user.allowed_unit_ids — already resolved to every org
    unit for hospital-wide roles, or a single subtree for scoped roles.
    """
    if not org_unit_ids:
        return []

    conn = get_connection()
    cursor = conn.cursor()

    try:
        placeholders = ','.join(['?'] * len(org_unit_ids))
        query = f"""
            SELECT {_SELECT_COLUMNS}
            {_FROM_JOINS}
            WHERE ai.TargetOrgUnitID IN ({placeholders})
            ORDER BY ai.DueDate ASC, ai.CreatedAt DESC
        """
        cursor.execute(query, list(org_unit_ids))
        rows = cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    finally:
        cursor.close()
        conn.close()


# ============================================================
# VALIDATION LOOKUPS (used by the service layer)
# ============================================================

def get_org_unit_frozen_status(unit_id: int) -> Optional[bool]:
    """
    Return whether a target org unit is frozen (legacy/inactive).
    Returns None if the unit doesn't exist.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT Frozen FROM dbo.AdminsrationUnit WHERE UniqueID = ?",
            (unit_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return bool(row.Frozen)

    finally:
        cursor.close()
        conn.close()


def get_org_unit_ids_for_user(user_id: int) -> List[int]:
    """
    Return the set of OrgUnitIDs a user holds a role-scope in (APP_UserRoleScope).
    A user may have multiple rows (multiple roles/units).
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT DISTINCT OrgUnitID FROM dbo.APP_UserRoleScope WHERE UserID = ?",
            (user_id,)
        )
        rows = cursor.fetchall()
        return [row.OrgUnitID for row in rows]

    finally:
        cursor.close()
        conn.close()


# ============================================================
# LIFECYCLE MUTATION
# ============================================================

def set_supervisor_action_item_completed(action_item_id: int, updated_by_user_id: int) -> bool:
    """
    Mark a supervisor action item as completed (set CompletedAt and Status).
    NO validation. Service layer handles workflow rules.

    Returns:
        True if updated, False if action item not found
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = """
            UPDATE dbo.APP_SupervisorActionItem
            SET Status = 'COMPLETED',
                CompletedAt = ?,
                UpdatedAt = ?,
                UpdatedByUserID = ?
            WHERE ActionItemID = ?
        """
        cursor.execute(query, (
            datetime.now(),
            datetime.now(),
            updated_by_user_id,
            action_item_id
        ))
        conn.commit()
        return cursor.rowcount > 0

    finally:
        cursor.close()
        conn.close()


def set_supervisor_action_item_cancelled(action_item_id: int, updated_by_user_id: int) -> bool:
    """
    Mark a supervisor action item as cancelled (set CancelledAt and Status).
    NO validation. Service layer handles workflow rules.

    Returns:
        True if updated, False if action item not found
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = """
            UPDATE dbo.APP_SupervisorActionItem
            SET Status = 'CANCELLED',
                CancelledAt = ?,
                UpdatedAt = ?,
                UpdatedByUserID = ?
            WHERE ActionItemID = ?
        """
        cursor.execute(query, (
            datetime.now(),
            datetime.now(),
            updated_by_user_id,
            action_item_id
        ))
        conn.commit()
        return cursor.rowcount > 0

    finally:
        cursor.close()
        conn.close()


# ============================================================
# NOTIFICATION / ACKNOWLEDGMENT
# Pure side-channel flag -- orthogonal to Status. NO validation,
# service layer confirms the caller is the actual recipient.
# ============================================================

def acknowledge_supervisor_action_item(action_item_id: int, acknowledged_by_user_id: int) -> bool:
    """
    Mark a supervisor action item as acknowledged by its recipient.

    Returns:
        True if this call set the acknowledgment, False if the item
        was not found or was already acknowledged.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = """
            UPDATE dbo.APP_SupervisorActionItem
            SET AcknowledgedAt = ?,
                AcknowledgedByUserID = ?
            WHERE ActionItemID = ? AND AcknowledgedAt IS NULL
        """
        cursor.execute(query, (datetime.now(), acknowledged_by_user_id, action_item_id))
        conn.commit()
        return cursor.rowcount > 0

    finally:
        cursor.close()
        conn.close()


def get_unacknowledged_supervisor_action_items_for_user(
    user_id: int,
    org_unit_ids: List[int]
) -> List[Dict[str, Any]]:
    """
    Fetch unacknowledged supervisor action items visible to this user:
    targeted directly at them, or targeted at an org unit they belong to
    when no specific TargetUserID was set.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        if org_unit_ids:
            placeholders = ','.join(['?'] * len(org_unit_ids))
            query = f"""
                SELECT {_SELECT_COLUMNS}
                {_FROM_JOINS}
                WHERE ai.AcknowledgedAt IS NULL
                AND (
                    ai.TargetUserID = ?
                    OR (ai.TargetUserID IS NULL AND ai.TargetOrgUnitID IN ({placeholders}))
                )
                ORDER BY ai.CreatedAt DESC
            """
            cursor.execute(query, [user_id] + list(org_unit_ids))
        else:
            query = f"""
                SELECT {_SELECT_COLUMNS}
                {_FROM_JOINS}
                WHERE ai.AcknowledgedAt IS NULL AND ai.TargetUserID = ?
                ORDER BY ai.CreatedAt DESC
            """
            cursor.execute(query, (user_id,))

        rows = cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    finally:
        cursor.close()
        conn.close()


# ============================================================
# AUDIT LOG
# ============================================================

def create_audit_log_entry(
    action_item_id: int,
    action: str,
    performed_by_user_id: int,
    note: Optional[str] = None
) -> Optional[int]:
    """
    Append an entry to APP_SupervisorActionItemAuditLog.
    NO validation. Service layer decides when to call this.

    Returns:
        AuditLogID if created, None on failure
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = """
            INSERT INTO dbo.APP_SupervisorActionItemAuditLog (
                ActionItemID,
                Action,
                PerformedByUserID,
                PerformedAt,
                Note
            )
            OUTPUT INSERTED.AuditLogID
            VALUES (?, ?, ?, ?, ?)
        """
        cursor.execute(query, (
            action_item_id,
            action,
            performed_by_user_id,
            datetime.now(),
            note
        ))
        row = cursor.fetchone()
        new_id = row[0] if row else None

        conn.commit()
        return new_id

    finally:
        cursor.close()
        conn.close()


def get_audit_log_for_action_item(action_item_id: int) -> List[Dict[str, Any]]:
    """Fetch the audit trail for a single supervisor action item, oldest first."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT AuditLogID, ActionItemID, Action, PerformedByUserID, PerformedAt, Note
            FROM dbo.APP_SupervisorActionItemAuditLog
            WHERE ActionItemID = ?
            ORDER BY PerformedAt ASC, AuditLogID ASC
            """,
            (action_item_id,)
        )
        rows = cursor.fetchall()
        return [
            {
                "audit_log_id": row.AuditLogID,
                "action_item_id": row.ActionItemID,
                "action": row.Action,
                "performed_by_user_id": row.PerformedByUserID,
                "performed_at": row.PerformedAt,
                "note": row.Note,
            }
            for row in rows
        ]

    finally:
        cursor.close()
        conn.close()
