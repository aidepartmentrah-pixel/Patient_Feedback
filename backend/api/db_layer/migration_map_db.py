"""
PHASE K — Migration Mapping DB Layer

Database layer functions for migration mappings.

This module provides read and write operations for the APP_DataMigration_Map table,
tracking legacy case → new case migrations.
"""

from core.database import get_connection


def get_migration_map_by_legacy_id(legacy_case_id: int) -> dict:
    """
    Retrieve migration mapping by legacy case ID.
    
    Args:
        legacy_case_id: ID from legacy IncidentRequestCase table
        
    Returns:
        dict: {
            "exists": True,
            "map_id": int,
            "legacy_case_id": int,
            "new_case_id": int,
            "migrated_by_user_id": int,
            "migrated_at": datetime
        }
        
        OR
        
        dict: {
            "exists": False
        }
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MapID, legacy_case_id, new_case_id, migrated_by_user_id, migrated_at
            FROM dbo.APP_DataMigration_Map
            WHERE legacy_case_id = ?
        """, legacy_case_id)
        
        row = cursor.fetchone()
        
        if row:
            return {
                "exists": True,
                "map_id": row[0],
                "legacy_case_id": row[1],
                "new_case_id": row[2],
                "migrated_by_user_id": row[3],
                "migrated_at": row[4]
            }
        else:
            return {"exists": False}
            
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def insert_migration_mapping(
    legacy_case_id: int,
    new_case_id: int,
    migrated_by_user_id: int
) -> dict:
    """
    Insert a migration mapping row into APP_DataMigration_Map.
    
    Performs proactive duplicate check before insert to prevent duplicate
    migrations of the same legacy case.
    
    Args:
        legacy_case_id: ID from legacy IncidentRequestCase table
        new_case_id: ID from APP_IncidentCase table
        migrated_by_user_id: User ID who performed the migration
        
    Returns:
        dict: {
            "success": True,
            "legacy_case_id": int,
            "new_case_id": int
        }
        
    Raises:
        ValueError: If legacy_case_id already has a mapping
        Exception: If database operation fails (FK violation, etc.)
    """
    conn = None
    cursor = None
    
    try:
        # Step 1: Open connection
        conn = get_connection()
        cursor = conn.cursor()
        
        # Step 2: Proactive duplicate check
        cursor.execute("""
            SELECT COUNT(*)
            FROM dbo.APP_DataMigration_Map
            WHERE legacy_case_id = ?
        """, legacy_case_id)
        
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0:
            raise ValueError("Legacy case already migrated")
        
        # Step 3: Insert mapping row
        cursor.execute("""
            INSERT INTO dbo.APP_DataMigration_Map
            (
                legacy_case_id,
                new_case_id,
                migrated_by_user_id,
                migrated_at
            )
            VALUES (?, ?, ?, GETDATE())
        """, legacy_case_id, new_case_id, migrated_by_user_id)
        
        # Step 4: Commit transaction
        conn.commit()
        
        # Step 5: Return structured result
        return {
            "success": True,
            "legacy_case_id": legacy_case_id,
            "new_case_id": new_case_id
        }
        
    except ValueError:
        # Duplicate mapping - rollback and re-raise
        if conn:
            conn.rollback()
        raise
        
    except Exception as e:
        # Generic database error - rollback and wrap
        if conn:
            conn.rollback()
        raise Exception("Failed to insert migration mapping: " + str(e))
        
    finally:
        # Always clean up resources
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def find_case_by_complaint_text(complaint_text: str) -> dict:
    """
    Search for an existing case by complaint text.
    
    This is a fallback mechanism for partial failure recovery:
    If mapping write failed after case creation, we can find the orphaned case
    by its complaint text on retry.
    
    Args:
        complaint_text: The exact complaint text to search for
        
    Returns:
        dict: {
            "exists": True,
            "case_id": int
        }
        
        OR
        
        dict: {
            "exists": False
        }
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Search for cases with exact complaint text match
        # Use TOP 1 to get most recent if multiple exist
        cursor.execute("""
            SELECT TOP 1 IncidentRequestCaseID
            FROM dbo.APP_IncidentCase
            WHERE ComplaintText = ?
            ORDER BY IncidentRequestCaseID DESC
        """, complaint_text)
        
        row = cursor.fetchone()
        
        if row:
            return {
                "exists": True,
                "case_id": row[0]
            }
        else:
            return {"exists": False}
            
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
