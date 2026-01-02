"""
FIX 1: ADD AUTO_INCREMENT TO ML DATABASE
=========================================
SQLite doesn't support traditional AUTO_INCREMENT on existing tables easily.
Solution: Use the insert function to handle NULL id (SQLite will use rowid)
"""

import sqlite3
import os
from pathlib import Path

ML_DB_PATH = Path(__file__).resolve().parent.parent / "models_directory" / "patient_feedback_ml.db"

def add_autoincrement_support():
    """Add AUTOINCREMENT-like behavior by ensuring id is handled properly."""
    try:
        conn = sqlite3.connect(str(ML_DB_PATH))
        cursor = conn.cursor()
        
        # Check current max id
        cursor.execute("SELECT MAX(id) FROM patient_feedback_encoded")
        max_id = cursor.fetchone()[0] or 0
        
        print(f"Current max id: {max_id}")
        print(f"Next record will have id: {max_id + 1}")
        
        # Solution: Modify insert function to handle id generation
        print("\nSOLUTION:")
        print("1. Modify ml_insert_adapter.py _insert_row() function")
        print("2. If 'id' is missing or None, calculate max(id)+1")
        print("3. Insert with explicit id value")
        
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    add_autoincrement_support()
