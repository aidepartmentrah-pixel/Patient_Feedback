"""
Migration Script: Add PatientAdmissionID column to APP_DrawerNote
Date: 2026-02-25

This script adds the PatientAdmissionID column to support linking 
drawer notes to patients.
"""

import pyodbc
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

from core.database import get_connection


def run_migration():
    """Add PatientAdmissionID column to APP_DrawerNote if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check if column already exists
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = 'APP_DrawerNote' 
            AND COLUMN_NAME = 'PatientAdmissionID'
            AND TABLE_SCHEMA = 'dbo'
        """)
        exists = cursor.fetchone()[0] > 0
        
        if exists:
            print("✓ PatientAdmissionID column already exists in APP_DrawerNote")
            return True
        
        # Add the column
        print("Adding PatientAdmissionID column to APP_DrawerNote table...")
        cursor.execute("""
            ALTER TABLE dbo.APP_DrawerNote
            ADD PatientAdmissionID INT NULL
        """)
        conn.commit()
        
        print("✓ Successfully added PatientAdmissionID column to APP_DrawerNote")
        return True
        
    except Exception as e:
        print(f"✗ Error during migration: {e}")
        conn.rollback()
        return False
        
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Migration: Add PatientAdmissionID to APP_DrawerNote")
    print("=" * 60)
    
    success = run_migration()
    
    if success:
        print("\n✓ Migration completed successfully!")
    else:
        print("\n✗ Migration failed!")
        sys.exit(1)
