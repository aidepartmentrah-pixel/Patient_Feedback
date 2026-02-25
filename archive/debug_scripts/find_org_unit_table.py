"""
Find Org Unit Table Name
"""
from backend.core.database import get_connection


def find_org_unit_table():
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Search for tables with ORGUNIT or DEPARTMENT in the name
        cursor.execute("""
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME LIKE '%ORG%' OR TABLE_NAME LIKE '%DEPARTMENT%'
            ORDER BY TABLE_NAME
        """)
        
        tables = cursor.fetchall()
        
        print("=" * 60)
        print("Tables with ORG or DEPARTMENT:")
        print("=" * 60)
        for table in tables:
            print(f"  - {table.TABLE_NAME}")
        
        return tables
        
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    find_org_unit_table()
