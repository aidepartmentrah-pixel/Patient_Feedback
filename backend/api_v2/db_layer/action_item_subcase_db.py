"""
Action Item Subcase Database Layer (API V2)
Handles SQL operations for APP_SubcaseActionItem table.

This is part of Phase 3 parallel workflow system.
NO business logic. NO authorization. ONLY SQL operations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, date
import pyodbc


def get_db_connection():
    """Get database connection using project standard."""
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn


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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
    
    Returns:
        List of action item dicts
    """
    conn = get_db_connection()
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
            WHERE AssignedToUserID = ?
            ORDER BY DueDate ASC, CreatedAt DESC
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
                "updated_by_user_id": row.UpdatedByUserID
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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


def set_action_item_started(
    action_item_id: int,
    updated_by_user_id: int
) -> bool:
    """
    Mark action item as started (set StartedAt timestamp and Status).
    
    Returns:
        True if updated, False if action item not found
    """
    conn = get_db_connection()
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
