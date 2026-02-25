"""
Check IncidentCase Schema for Org Unit Columns
"""
from backend.core.database import get_connection


def check_incident_schema():
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get all columns from APP_IncidentCase
        cursor.execute("""
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'APP_IncidentCase'
            AND (COLUMN_NAME LIKE '%Org%' OR COLUMN_NAME LIKE '%Department%')
            ORDER BY ORDINAL_POSITION
        """)
        
        columns = cursor.fetchall()
        
        print("=" * 60)
        print("Org/Department Columns in APP_IncidentCase:")
        print("=" * 60)
        for col in columns:
            print(f"  - {col.COLUMN_NAME} ({col.DATA_TYPE}, {col.IS_NULLABLE})")
        
        # Try to get a sample value
        cursor.execute("SELECT TOP 1 IssuingOrgUnitID FROM dbo.APP_IncidentCase")
        row = cursor.fetchone()
        if row:
            print(f"\n✅ Sample IssuingOrgUnitID: {row.IssuingOrgUnitID}")
        
        return columns
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    check_incident_schema()
