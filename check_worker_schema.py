"""
Check APP_IncidentCaseEmployee and related tables schema
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from core.database import get_connection

def check_schema():
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        print("\n" + "="*70)
        print("CHECKING APP_IncidentCaseEmployee SCHEMA")
        print("="*70)
        
        # Get column info
        cursor.execute("""
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'APP_IncidentCaseEmployee'
            ORDER BY ORDINAL_POSITION
        """)
        
        columns = cursor.fetchall()
        
        if columns:
            print(f"\nFound {len(columns)} columns:")
            print(f"\n{'Column Name':<30} {'Data Type':<20} {'Nullable':<10} {'Max Length':<10}")
            print("-" * 70)
            for col in columns:
                print(f"{col.COLUMN_NAME:<30} {col.DATA_TYPE:<20} {col.IS_NULLABLE:<10} {str(col.CHARACTER_MAXIMUM_LENGTH or ''):<10}")
        else:
            print("\n⚠ Table APP_IncidentCaseEmployee not found or no columns")
        
        # Check if table exists and get sample data
        print("\n" + "="*70)
        print("SAMPLE DATA (first 5 rows)")
        print("="*70)
        
        cursor.execute("SELECT TOP 5 * FROM APP_IncidentCaseEmployee")
        rows = cursor.fetchall()
        
        if rows:
            # Get column names
            col_names = [col[0] for col in cursor.description]
            print(f"\nColumns: {', '.join(col_names)}")
            print("\nData:")
            for idx, row in enumerate(rows, 1):
                print(f"\nRow {idx}:")
                for col_name, value in zip(col_names, row):
                    print(f"  {col_name}: {value}")
        else:
            print("\nNo data in table")
        
        # Check APP_LOOKUP_EXPLANATION_STATUS
        print("\n" + "="*70)
        print("APP_LOOKUP_EXPLANATION_STATUS")
        print("="*70)
        
        cursor.execute("SELECT * FROM APP_LOOKUP_EXPLANATION_STATUS")
        statuses = cursor.fetchall()
        
        if statuses:
            col_names = [col[0] for col in cursor.description]
            print(f"\nColumns: {', '.join(col_names)}")
            for row in statuses:
                print(f"\n  {dict(zip(col_names, row))}")
        
        # Check HR employees view
        print("\n" + "="*70)
        print("APP_VIEWTABLE_HR_EMPLOYEES (schema)")
        print("="*70)
        
        cursor.execute("""
            SELECT TOP 5
                COLUMN_NAME,
                DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'APP_VIEWTABLE_HR_EMPLOYEES'
            ORDER BY ORDINAL_POSITION
        """)
        
        hr_cols = cursor.fetchall()
        if hr_cols:
            print("\nFirst 5 columns:")
            for col in hr_cols:
                print(f"  {col.COLUMN_NAME}: {col.DATA_TYPE}")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    check_schema()
