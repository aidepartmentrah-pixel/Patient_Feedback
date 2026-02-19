"""
PHASE K — Migration Progress DB Layer

Database layer functions for migration progress reporting.

This module provides read-only operations to report migration statistics:
- Total legacy case count (from IncidentRequestCase - the OLD table)
- Migrated case count (from APP_DataMigration_Map - tracking table)

NO WRITES - Read-only reporting only.
"""

from core.database import get_connection


def get_migration_progress_counts() -> dict:
    """
    Retrieve migration progress counts from database.
    
    This function runs two queries:
    1. Count total cases in IncidentRequestCase (LEGACY table - source)
    2. Count migrated cases in APP_DataMigration_Map (tracking table)
    
    Returns:
        dict: {
            "total_cases": int,
            "migrated_cases": int
        }
        
    Raises:
        Exception: If database query fails
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # QUERY 1: Total legacy cases (from IncidentRequestCase - the OLD table)
        cursor.execute("""
            SELECT COUNT(*) AS total_cases
            FROM dbo.IncidentRequestCase
        """)
        
        total_row = cursor.fetchone()
        total_cases = total_row[0] if total_row else 0
        
        # QUERY 2: Migrated cases (tracked in APP_DataMigration_Map)
        cursor.execute("""
            SELECT COUNT(*) AS migrated_cases
            FROM dbo.APP_DataMigration_Map
        """)
        
        migrated_row = cursor.fetchone()
        migrated_cases = migrated_row[0] if migrated_row else 0
        
        return {
            "total_cases": total_cases,
            "migrated_cases": migrated_cases
        }
        
    except Exception as e:
        # Read-only operation - no rollback needed
        raise Exception("Failed to read migration progress: " + str(e))
        
    finally:
        # Always clean up resources
        if cursor:
            cursor.close()
        if conn:
            conn.close()
