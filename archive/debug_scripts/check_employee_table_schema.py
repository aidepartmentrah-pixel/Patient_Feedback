"""
Check APP_IncidentCaseEmployee table schema
"""
import pyodbc
from backend.core.database import get_connection

def check_employee_table():
    """Check current schema of APP_IncidentCaseEmployee"""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        print("=" * 60)
        print("APP_IncidentCaseEmployee Table Schema")
        print("=" * 60)
        
        # Get table schema
        cursor.execute("""
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'APP_IncidentCaseEmployee'
            ORDER BY ORDINAL_POSITION
        """)
        
        columns = cursor.fetchall()
        
        if not columns:
            print("\n❌ Table does not exist!")
            return False
        
        print(f"\nFound {len(columns)} columns:")
        print("-" * 60)
        for col in columns:
            nullable = "NULL" if col.IS_NULLABLE == "YES" else "NOT NULL"
            default = f" DEFAULT {col.COLUMN_DEFAULT}" if col.COLUMN_DEFAULT else ""
            print(f"{col.COLUMN_NAME:30} {col.DATA_TYPE:15} {nullable:10}{default}")
        
        # Check for required columns
        column_names = [col.COLUMN_NAME for col in columns]
        required = ['IncidentRequestCaseID', 'IsPrimary', 'AssignedAt', 'AssignedByUserID']
        
        print("\n" + "=" * 60)
        print("Required Columns Status:")
        print("=" * 60)
        
        all_present = True
        for req in required:
            if req in column_names:
                print(f"✅ {req} - Present")
            else:
                print(f"❌ {req} - MISSING")
                all_present = False
        
        if all_present:
            print("\n✅ All required columns are present!")
            print("You can skip running ALTER_EMPLOYEE_TABLE.sql")
        else:
            print("\n⚠️  Missing columns detected!")
            print("You need to run ALTER_EMPLOYEE_TABLE.sql")
        
        return all_present
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    check_employee_table()
