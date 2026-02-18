"""
Check how employees link to incidents
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from core.database import get_connection

def check_linkage():
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        print("\n" + "="*70)
        print("CHECKING EMPLOYEE-INCIDENT LINKAGE")
        print("="*70)
        
        # Check APP_IncidentCase columns related to employees
        print("\n1. APP_IncidentCase - Employee-related columns:")
        cursor.execute("""
            SELECT 
                COLUMN_NAME,
                DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'APP_IncidentCase'
            AND (COLUMN_NAME LIKE '%Employee%' OR COLUMN_NAME LIKE '%User%' OR COLUMN_NAME LIKE '%Assigned%' OR COLUMN_NAME LIKE '%Responsible%')
            ORDER BY ORDINAL_POSITION
        """)
        
        emp_cols = cursor.fetchall()
        if emp_cols:
            for col in emp_cols:
                print(f"  {col.COLUMN_NAME}: {col.DATA_TYPE}")
        else:
            print("  No employee-related columns found")
        
        # Check if APP_IncidentCaseEmployee is a junction table or view
        print("\n2. Check if APP_IncidentCaseEmployee is a view or table:")
        cursor.execute("""
            SELECT 
                TABLE_TYPE
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_NAME = 'APP_IncidentCaseEmployee'
        """)
        
        table_type = cursor.fetchone()
        if table_type:
            print(f"  Type: {table_type.TABLE_TYPE}")
        
        # Try to find any junction table
        print("\n3. Looking for junction/link tables:")
        cursor.execute("""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_NAME LIKE '%Incident%Employee%' 
               OR TABLE_NAME LIKE '%Employee%Incident%'
               OR TABLE_NAME LIKE '%Case%Employee%'
        """)
        
        junction_tables = cursor.fetchall()
        if junction_tables:
            for table in junction_tables:
                print(f"  Found: {table.TABLE_NAME}")
                
                # Get columns
                cursor.execute(f"""
                    SELECT COLUMN_NAME, DATA_TYPE
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = '{table.TABLE_NAME}'
                    ORDER BY ORDINAL_POSITION
                """)
                cols = cursor.fetchall()
                for col in cols:
                    print(f"    - {col.COLUMN_NAME}: {col.DATA_TYPE}")
        
        # Check CreatedByUserID approach (current fallback)
        print("\n4. Sample APP_IncidentCase with CreatedByUserID:")
        cursor.execute("""
            SELECT TOP 5
                IncidentID,
                CreatedByUserID,
                FeedbackRecievedDate,
                Category
            FROM APP_IncidentCase
            WHERE CreatedByUserID IS NOT NULL
        """)
        
        cases = cursor.fetchall()
        if cases:
            print("  IncidentID | CreatedByUserID | Date | Category")
            print("  " + "-"*60)
            for case in cases:
                print(f"  {case.IncidentID} | {case.CreatedByUserID} | {case.FeedbackRecievedDate} | {case.Category}")
        
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
    check_linkage()
