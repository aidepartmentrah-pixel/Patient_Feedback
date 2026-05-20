"""
Action Item Subcase Database Layer (API V2)
Handles SQL operations for APP_SubcaseActionItem table.

This is part of Phase 3 parallel workflow system.
NO business logic. NO authorization. ONLY SQL operations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, date
import pyodbc
from core.database import get_connection


# ============================================================
# CREATION / FETCH
# ============================================================

def create_action_item(
    subcase_id: int,
    title: str,
    description: Optional[str],
    due_date: Optional[date],
    created_by_user_id: int,
    initial_status: str = "DRAFT",
    assigned_to_user_id: Optional[int] = None
) -> Optional[int]:
    """
    Create a new action item for a subcase.
    
    Args:
        subcase_id: FK to APP_AdministrativeSubcase
        title: Action item title (max 300 chars)
        description: Optional detailed description
        due_date: Optional due date
        created_by_user_id: Who created this action item
        initial_status: Initial status (default: DRAFT)
        assigned_to_user_id: Optional user assignment
    
    Returns:
        ActionItemID if created, None on failure
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            INSERT INTO dbo.APP_SubcaseActionItem (
                SubcaseID,
                Status,
                Title,
                Description,
                DueDate,
                AssignedToUserID,
                CreatedAt,
                CreatedByUserID
            )
            OUTPUT INSERTED.ActionItemID
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor.execute(query, (
            subcase_id,
            initial_status,
            title,
            description,
            due_date,
            assigned_to_user_id,
            datetime.now(),
            created_by_user_id
        ))
        
        row = cursor.fetchone()
        new_id = row[0] if row else None
        
        conn.commit()
        return new_id
    
    finally:
        cursor.close()
        conn.close()


def get_action_item_by_id(action_item_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch a single action item by ID.
    
    Returns:
        Action item dict or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT 
                ActionItemID,
                SubcaseID,
                Status,
                Title,
                Description,
                DueDate,
                AssignedToUserID,
                StartedAt,
                CompletedAt,
                VerifiedAt,
                CreatedAt,
                CreatedByUserID,
                UpdatedAt,
                UpdatedByUserID
            FROM dbo.APP_SubcaseActionItem
            WHERE ActionItemID = ?
        """
        
        cursor.execute(query, (action_item_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return {
            "action_item_id": row.ActionItemID,
            "subcase_id": row.SubcaseID,
            "status": row.Status,
            "title": row.Title,
            "description": row.Description,
            "due_date": row.DueDate,
            "assigned_to_user_id": row.AssignedToUserID,
            "started_at": row.StartedAt,
            "completed_at": row.CompletedAt,
            "verified_at": row.VerifiedAt,
            "created_at": row.CreatedAt,
            "created_by_user_id": row.CreatedByUserID,
            "updated_at": row.UpdatedAt,
            "updated_by_user_id": row.UpdatedByUserID
        }
    
    finally:
        cursor.close()
        conn.close()


def get_action_items_by_subcase(subcase_id: int) -> List[Dict[str, Any]]:
    """
    Fetch all action items for a specific subcase.
    
    Returns:
        List of action item dicts
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT 
                ActionItemID,
                SubcaseID,
                Status,
                Title,
                Description,
                DueDate,
                AssignedToUserID,
                StartedAt,
                CompletedAt,
                VerifiedAt,
                CreatedAt,
                CreatedByUserID,
                UpdatedAt,
                UpdatedByUserID
            FROM dbo.APP_SubcaseActionItem
            WHERE SubcaseID = ?
            ORDER BY CreatedAt ASC
        """
        
        cursor.execute(query, (subcase_id,))
        rows = cursor.fetchall()
        
        return [
            {
                "action_item_id": row.ActionItemID,
                "subcase_id": row.SubcaseID,
                "status": row.Status,
                "title": row.Title,
                "description": row.Description,
                "due_date": row.DueDate,
                "assigned_to_user_id": row.AssignedToUserID,
                "started_at": row.StartedAt,
                "completed_at": row.CompletedAt,
                "verified_at": row.VerifiedAt,
                "created_at": row.CreatedAt,
                "created_by_user_id": row.CreatedByUserID,
                "updated_at": row.UpdatedAt,
                "updated_by_user_id": row.UpdatedByUserID
            }
            for row in rows
        ]
    
    finally:
        cursor.close()
        conn.close()


def get_action_items_by_status(status_code: str) -> List[Dict[str, Any]]:
    """
    Fetch all action items with a specific status.
    
    Returns:
        List of action item dicts
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT 
                ActionItemID,
                SubcaseID,
                Status,
                Title,
                Description,
                DueDate,
                AssignedToUserID,
                StartedAt,
                CompletedAt,
                VerifiedAt,
                CreatedAt,
                CreatedByUserID,
                UpdatedAt,
                UpdatedByUserID
            FROM dbo.APP_SubcaseActionItem
            WHERE Status = ?
            ORDER BY CreatedAt DESC
        """
        
        cursor.execute(query, (status_code,))
        rows = cursor.fetchall()
        
        return [
            {
                "action_item_id": row.ActionItemID,
                "subcase_id": row.SubcaseID,
                "status": row.Status,
                "title": row.Title,
                "description": row.Description,
                "due_date": row.DueDate,
                "assigned_to_user_id": row.AssignedToUserID,
                "started_at": row.StartedAt,
                "completed_at": row.CompletedAt,
                "verified_at": row.VerifiedAt,
                "created_at": row.CreatedAt,
                "created_by_user_id": row.CreatedByUserID,
                "updated_at": row.UpdatedAt,
                "updated_by_user_id": row.UpdatedByUserID
            }
            for row in rows
        ]
    
    finally:
        cursor.close()
        conn.close()


# ============================================================
# ASSIGNMENT / TRACKING
# ============================================================

def get_action_items_by_assigned_user(user_id: int) -> List[Dict[str, Any]]:
    """
    Fetch all action items assigned to a specific user.
    Joins subcase + incident tables to include case context fields.

    Returns:
        List of action item dicts
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = """
            SELECT
                ai.ActionItemID,
                ai.SubcaseID,
                ai.Status,
                ai.Title,
                ai.Description,
                ai.DueDate,
                ai.AssignedToUserID,
                ai.StartedAt,
                ai.CompletedAt,
                ai.VerifiedAt,
                ai.CreatedAt,
                ai.CreatedByUserID,
                ai.UpdatedAt,
                ai.UpdatedByUserID,
                s.TargetOrgUnitID,
                s.CaseType,
                s.IncidentRequestCaseID,
                s.SeasonalReportID,
                ou.Name AS OrgUnitName,
                ic.ComplaintText AS CaseDescription,
                ic.PatientName,
                inc.incident_number AS IncidentNumber,
                sev.SeverityName,
                cat.CategoryName
            FROM dbo.APP_SubcaseActionItem ai
            INNER JOIN dbo.APP_AdministrativeSubcase s ON ai.SubcaseID = s.SubcaseID
            LEFT JOIN dbo.AdminsrationUnit ou ON s.TargetOrgUnitID = ou.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic ON s.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_Incident inc ON ic.incident_id = inc.incident_id
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat ON ic.CategoryID = cat.CategoryID
            WHERE ai.AssignedToUserID = ?
            ORDER BY ai.DueDate ASC, ai.CreatedAt DESC
        """

        cursor.execute(query, (user_id,))
        rows = cursor.fetchall()

        return [
            {
                "action_item_id": row.ActionItemID,
                "subcase_id": row.SubcaseID,
                "status": row.Status,
                "title": row.Title,
                "description": row.Description,
                "due_date": row.DueDate,
                "assigned_to_user_id": row.AssignedToUserID,
                "started_at": row.StartedAt,
                "completed_at": row.CompletedAt,
                "verified_at": row.VerifiedAt,
                "created_at": row.CreatedAt,
                "created_by_user_id": row.CreatedByUserID,
                "updated_at": row.UpdatedAt,
                "updated_by_user_id": row.UpdatedByUserID,
                "target_org_unit_id": row.TargetOrgUnitID,
                "case_type": row.CaseType,
                "incident_request_case_id": row.IncidentRequestCaseID,
                "seasonal_report_id": row.SeasonalReportID,
                "org_unit_name": row.OrgUnitName,
                "case_description": row.CaseDescription,
                "patient_name": row.PatientName,
                "incident_number": row.IncidentNumber,
                "severity_name": row.SeverityName,
                "category_name": row.CategoryName,
            }
            for row in rows
        ]

    finally:
        cursor.close()
        conn.close()


def get_action_items_by_scope(allowed_unit_ids: List[int], actionable_only: bool = True) -> List[Dict[str, Any]]:
    """
    Fetch action items within a set of org unit scopes.
    
    Joins APP_SubcaseActionItem with APP_AdministrativeSubcase so we can
    filter by the subcase's TargetOrgUnitID matching the user's allowed_unit_ids.
    
    Args:
        allowed_unit_ids: List of OrgUnitIDs the user is allowed to see
        actionable_only: If True, only return items in actionable statuses
                         (ADMIN_APPROVED, IN_PROGRESS). If False, return all.
    
    Returns:
        List of action item dicts (with extra subcase context fields)
    """
    if not allowed_unit_ids:
        return []
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        placeholders = ','.join(['?'] * len(allowed_unit_ids))
        
        status_filter = ""
        if actionable_only:
            status_filter = "AND ai.Status IN ('ADMIN_APPROVED', 'IN_PROGRESS')"
        
        query = f"""
            SELECT
                ai.ActionItemID,
                ai.SubcaseID,
                ai.Status,
                ai.Title,
                ai.Description,
                ai.DueDate,
                ai.AssignedToUserID,
                ai.StartedAt,
                ai.CompletedAt,
                ai.VerifiedAt,
                ai.CreatedAt,
                ai.CreatedByUserID,
                ai.UpdatedAt,
                ai.UpdatedByUserID,
                s.TargetOrgUnitID,
                s.CaseType,
                s.IncidentRequestCaseID,
                s.SeasonalReportID,
                ou.Name AS OrgUnitName,
                ic.ComplaintText AS CaseDescription,
                ic.PatientName,
                inc.incident_number AS IncidentNumber,
                sev.SeverityName,
                cat.CategoryName
            FROM dbo.APP_SubcaseActionItem ai
            INNER JOIN dbo.APP_AdministrativeSubcase s ON ai.SubcaseID = s.SubcaseID
            LEFT JOIN dbo.AdminsrationUnit ou ON s.TargetOrgUnitID = ou.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic ON s.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_Incident inc ON ic.incident_id = inc.incident_id
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat ON ic.CategoryID = cat.CategoryID
            WHERE s.TargetOrgUnitID IN ({placeholders})
            {status_filter}
            ORDER BY ai.DueDate ASC, ai.CreatedAt DESC
        """

        cursor.execute(query, list(allowed_unit_ids))
        rows = cursor.fetchall()

        return [
            {
                "action_item_id": row.ActionItemID,
                "subcase_id": row.SubcaseID,
                "status": row.Status,
                "title": row.Title,
                "description": row.Description,
                "due_date": row.DueDate,
                "assigned_to_user_id": row.AssignedToUserID,
                "started_at": row.StartedAt,
                "completed_at": row.CompletedAt,
                "verified_at": row.VerifiedAt,
                "created_at": row.CreatedAt,
                "created_by_user_id": row.CreatedByUserID,
                "updated_at": row.UpdatedAt,
                "updated_by_user_id": row.UpdatedByUserID,
                "target_org_unit_id": row.TargetOrgUnitID,
                "case_type": row.CaseType,
                "incident_request_case_id": row.IncidentRequestCaseID,
                "seasonal_report_id": row.SeasonalReportID,
                "org_unit_name": row.OrgUnitName,
                "case_description": row.CaseDescription,
                "patient_name": row.PatientName,
                "incident_number": row.IncidentNumber,
                "severity_name": row.SeverityName,
                "category_name": row.CategoryName,
            }
            for row in rows
        ]
    
    finally:
        cursor.close()
        conn.close()


def get_overdue_action_items() -> List[Dict[str, Any]]:
    """
    Fetch all action items that are overdue (past due date and not in final status).
    
    Returns:
        List of overdue action item dicts
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT 
                ActionItemID,
                SubcaseID,
                Status,
                Title,
                Description,
                DueDate,
                AssignedToUserID,
                StartedAt,
                CompletedAt,
                VerifiedAt,
                CreatedAt,
                CreatedByUserID,
                UpdatedAt,
                UpdatedByUserID
            FROM dbo.APP_SubcaseActionItem
            WHERE DueDate IS NOT NULL
              AND DueDate < CAST(GETDATE() AS DATE)
              AND Status NOT IN ('VERIFIED', 'CANCELLED')
            ORDER BY DueDate ASC
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        return [
            {
                "action_item_id": row.ActionItemID,
                "subcase_id": row.SubcaseID,
                "status": row.Status,
                "title": row.Title,
                "description": row.Description,
                "due_date": row.DueDate,
                "assigned_to_user_id": row.AssignedToUserID,
                "started_at": row.StartedAt,
                "completed_at": row.CompletedAt,
                "verified_at": row.VerifiedAt,
                "created_at": row.CreatedAt,
                "created_by_user_id": row.CreatedByUserID,
                "updated_at": row.UpdatedAt,
                "updated_by_user_id": row.UpdatedByUserID
            }
            for row in rows
        ]
    
    finally:
        cursor.close()
        conn.close()


# ============================================================
# WORKFLOW MUTATION
# ============================================================

def update_action_item_status(
    action_item_id: int,
    new_status: str,
    updated_by_user_id: int
) -> bool:
    """
    Update action item status.
    NO validation. Service layer handles workflow rules.
    
    Returns:
        True if updated, False if action item not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_SubcaseActionItem
            SET Status = ?,
                UpdatedAt = ?,
                UpdatedByUserID = ?
            WHERE ActionItemID = ?
        """
        
        cursor.execute(query, (
            new_status,
            datetime.now(),
            updated_by_user_id,
            action_item_id
        ))
        
        conn.commit()
        return cursor.rowcount > 0
    
    finally:
        cursor.close()
        conn.close()


def update_action_item_due_date(
    action_item_id: int,
    new_due_date,
    updated_by_user_id: int
) -> bool:
    """
    Update the DueDate of an action item.
    NO validation. Service layer handles workflow rules.
    
    Args:
        action_item_id: ID of the action item
        new_due_date: New due date (date or datetime object)
        updated_by_user_id: ID of the user making the change
    
    Returns:
        True if updated, False if action item not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_SubcaseActionItem
            SET DueDate = ?,
                UpdatedAt = ?,
                UpdatedByUserID = ?
            WHERE ActionItemID = ?
        """
        
        cursor.execute(query, (
            new_due_date,
            datetime.now(),
            updated_by_user_id,
            action_item_id
        ))
        
        conn.commit()
        return cursor.rowcount > 0
    
    finally:
        cursor.close()
        conn.close()


def bulk_update_action_items_status_by_subcase(
    subcase_id: int,
    to_status: str,
    updated_by_user_id: int,
    from_statuses: Optional[List[str]] = None
) -> int:
    """
    Bulk-update the status of ALL action items belonging to a subcase.
    
    Used by workflow transitions to move action items in parallel with
    their parent subcase through the approval pipeline.
    
    Args:
        subcase_id: Parent subcase ID
        to_status: New status to set (must exist in APP_Lookup_SubcaseActionItemStatus)
        updated_by_user_id: User performing the transition
        from_statuses: Optional list of current statuses to filter on.
                       If None, updates ALL non-final action items for this subcase.
                       
    Returns:
        Number of rows updated
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        now = datetime.now()
        
        if from_statuses:
            placeholders = ','.join(['?' for _ in from_statuses])
            query = f"""
                UPDATE dbo.APP_SubcaseActionItem
                SET Status = ?,
                    UpdatedAt = ?,
                    UpdatedByUserID = ?
                WHERE SubcaseID = ?
                  AND Status IN ({placeholders})
            """
            params = [to_status, now, updated_by_user_id, subcase_id] + list(from_statuses)
        else:
            # Update all non-final items (exclude VERIFIED, CANCELLED)
            query = """
                UPDATE dbo.APP_SubcaseActionItem
                SET Status = ?,
                    UpdatedAt = ?,
                    UpdatedByUserID = ?
                WHERE SubcaseID = ?
                  AND Status NOT IN ('VERIFIED', 'CANCELLED')
            """
            params = [to_status, now, updated_by_user_id, subcase_id]
        
        cursor.execute(query, params)
        conn.commit()
        return cursor.rowcount
    
    finally:
        cursor.close()
        conn.close()


def set_action_item_started(
    action_item_id: int,
    updated_by_user_id: int
) -> bool:
    """
    Mark action item as started (set StartedAt timestamp and Status).
    
    Returns:
        True if updated, False if action item not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_SubcaseActionItem
            SET Status = 'IN_PROGRESS',
                StartedAt = ?,
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


def set_action_item_completed(
    action_item_id: int,
    updated_by_user_id: int
) -> bool:
    """
    Mark action item as completed (set CompletedAt timestamp and Status to DONE).
    
    Returns:
        True if updated, False if action item not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_SubcaseActionItem
            SET Status = 'DONE',
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


def set_action_item_verified(
    action_item_id: int,
    updated_by_user_id: int
) -> bool:
    """
    Mark action item as verified (set VerifiedAt timestamp).
    
    Returns:
        True if updated, False if action item not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_SubcaseActionItem
            SET VerifiedAt = ?,
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


def reassign_action_item(
    action_item_id: int,
    new_user_id: int,
    updated_by_user_id: int
) -> bool:
    """
    Reassign action item to a different user.
    
    Returns:
        True if updated, False if action item not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_SubcaseActionItem
            SET AssignedToUserID = ?,
                UpdatedAt = ?,
                UpdatedByUserID = ?
            WHERE ActionItemID = ?
        """
        
        cursor.execute(query, (
            new_user_id,
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
# ADMINISTRATION
# ============================================================

def delete_action_item(action_item_id: int) -> bool:
    """
    Delete an action item.
    Simple delete, no cascade.
    
    Returns:
        True if deleted, False if action item not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            DELETE FROM dbo.APP_SubcaseActionItem
            WHERE ActionItemID = ?
        """
        
        cursor.execute(query, (action_item_id,))
        
        conn.commit()
        return cursor.rowcount > 0
    
    finally:
        cursor.close()
        conn.close()


# ============================================================
# PHASE F — ACTION LOG REPORT QUERIES
# ============================================================

def get_action_items_by_due_date_range(
    conn: pyodbc.Connection,
    start_date: date,
    end_date: date
) -> List[Dict[str, Any]]:
    """
    Get all action items with DueDate in specified range (PHASE F — ACTION LOG REPORT).
    
    This query applies the following filters (LOCKED BY DECISION BLOCK):
    - DueDate BETWEEN start_date AND end_date (inclusive)
    - Parent subcase status >= ADMIN_APPROVED (passed workflow gate)
    - Action item status != CANCELLED (excluded from report)
    - DueDate IS NOT NULL
    
    This is a FLAT query with no grouping, no status classification, no overdue logic.
    Those are handled at the service layer.
    
    No role/scope filtering is applied here - that is handled at service/router layer.
    
    Args:
        conn: Database connection
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)
        
    Returns:
        List of action item dicts with joined user and org unit data
        Each dict contains:
        - action_item_id
        - subcase_id
        - title
        - description
        - status_code (action item status)
        - due_date
        - started_at
        - completed_at
        - verified_at
        - assigned_to_user_id
        - assigned_to_display_name (from APP_Users)
        - subcase_status
        - target_org_unit_id
        - org_unit_name (from AdminsrationUnit)
    """
    cursor = conn.cursor()
    
    query = """
        SELECT 
            a.ActionItemID,
            a.SubcaseID,
            a.Title,
            a.Description,
            a.Status AS StatusCode,
            a.DueDate,
            a.StartedAt,
            a.CompletedAt,
            a.VerifiedAt,
            a.AssignedToUserID,
            u.DisplayName AS AssignedToDisplayName,
            s.Status AS SubcaseStatus,
            s.TargetOrgUnitID,
            ou.Name AS OrgUnitName
        FROM dbo.APP_SubcaseActionItem a
        INNER JOIN dbo.APP_AdministrativeSubcase s 
            ON s.SubcaseID = a.SubcaseID
        LEFT JOIN dbo.APP_Users u 
            ON u.UserID = a.AssignedToUserID
        LEFT JOIN dbo.AdminsrationUnit ou 
            ON ou.UniqueID = s.TargetOrgUnitID
        WHERE
            a.DueDate IS NOT NULL
            AND a.DueDate >= ?
            AND a.DueDate <= ?
            AND a.Status <> 'CANCELLED'
            AND s.Status IN ('ADMIN_APPROVED', 'IN_PROGRESS', 'DONE', 'VERIFIED')
        ORDER BY a.DueDate, a.ActionItemID
    """
    
    cursor.execute(query, (start_date, end_date))
    rows = cursor.fetchall()
    cursor.close()
    
    action_items = []
    for row in rows:
        action_items.append({
            "action_item_id": row.ActionItemID,
            "subcase_id": row.SubcaseID,
            "title": row.Title,
            "description": row.Description,
            "status_code": row.StatusCode,
            "due_date": row.DueDate,
            "started_at": row.StartedAt,
            "completed_at": row.CompletedAt,
            "verified_at": row.VerifiedAt,
            "assigned_to_user_id": row.AssignedToUserID,
            "assigned_to_display_name": row.AssignedToDisplayName,
            "subcase_status": row.SubcaseStatus,
            "target_org_unit_id": row.TargetOrgUnitID,
            "org_unit_name": row.OrgUnitName
        })
    
    return action_items


# ============================================================
# WORKER ACTION ITEMS (Phase B — B-B4)
# ============================================================

def get_worker_action_items(
    employee_id: int,
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get action items for a specific worker (employee) with pagination.
    
    Phase B — B-B4 — Worker action list endpoint V2.
    Joins with subcase table to include incident_case_id.
    
    Args:
        employee_id: Employee ID (same as AssignedToUserID)
        limit: Maximum number of items to return
        offset: Number of items to skip
        status: Optional status filter (e.g., 'DRAFT', 'IN_PROGRESS', 'COMPLETED')
    
    Returns:
        Dict with:
        - items: List of action item dicts
        - total_count: Total number of matching action items (before pagination)
        - limit: Applied limit
        - offset: Applied offset
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Build status filter
        status_filter = ""
        params_count = [employee_id]
        params_data = [employee_id]
        
        if status:
            status_filter = "AND ai.Status = ?"
            params_count.append(status)
            params_data.append(status)
        
        # Count total matching items
        count_query = f"""
            SELECT COUNT(*)
            FROM dbo.APP_SubcaseActionItem ai
            WHERE ai.AssignedToUserID = ?
            {status_filter}
        """
        
        cursor.execute(count_query, params_count)
        total_count = cursor.fetchone()[0]
        
        # Get paginated data with subcase join
        data_query = f"""
            SELECT 
                ai.ActionItemID,
                ai.SubcaseID,
                ai.Status,
                ai.Title,
                ai.Description,
                ai.DueDate,
                ai.AssignedToUserID,
                ai.StartedAt,
                ai.CompletedAt,
                ai.VerifiedAt,
                ai.CreatedAt,
                ai.CreatedByUserID,
                ai.UpdatedAt,
                ai.UpdatedByUserID,
                sc.IncidentRequestCaseID
            FROM dbo.APP_SubcaseActionItem ai
            LEFT JOIN dbo.APP_AdministrativeSubcase sc ON ai.SubcaseID = sc.SubcaseID
            WHERE ai.AssignedToUserID = ?
            {status_filter}
            ORDER BY ai.DueDate ASC, ai.CreatedAt DESC
            OFFSET ? ROWS
            FETCH NEXT ? ROWS ONLY
        """
        
        params_data.extend([offset, limit])
        cursor.execute(data_query, params_data)
        rows = cursor.fetchall()
        
        items = [
            {
                "action_item_id": row.ActionItemID,
                "subcase_id": row.SubcaseID,
                "status": row.Status,
                "title": row.Title,
                "description": row.Description,
                "due_date": row.DueDate,
                "assigned_to_user_id": row.AssignedToUserID,
                "started_at": row.StartedAt,
                "completed_at": row.CompletedAt,
                "verified_at": row.VerifiedAt,
                "created_at": row.CreatedAt,
                "created_by_user_id": row.CreatedByUserID,
                "updated_at": row.UpdatedAt,
                "updated_by_user_id": row.UpdatedByUserID,
                "incident_case_id": row.IncidentRequestCaseID
            }
            for row in rows
        ]
        
        return {
            "items": items,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }
    
    finally:
        cursor.close()
        conn.close()
