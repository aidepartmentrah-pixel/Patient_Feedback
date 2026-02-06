"""
Query the VW_AdminstrationUnit view for organizational structure
"""
import pyodbc

def get_connection():
    """Get SQL Server database connection."""
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn

def main():
    conn = get_connection()
    cursor = conn.cursor()
    
    print("\n" + "="*80)
    print("🏢 ORGANIZATIONAL STRUCTURE FROM VW_AdminstrationUnit")
    print("="*80)
    
    # Get column structure
    print("\n--- Columns in VW_AdminstrationUnit ---")
    cursor.execute("""
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'VW_AdminstrationUnit'
        AND TABLE_SCHEMA = 'dbo'
        ORDER BY ORDINAL_POSITION
    """)
    columns = cursor.fetchall()
    print(f"\nFound {len(columns)} columns:")
    for col in columns:
        max_len = f"({col.CHARACTER_MAXIMUM_LENGTH})" if col.CHARACTER_MAXIMUM_LENGTH else ""
        print(f"  - {col.COLUMN_NAME}: {col.DATA_TYPE}{max_len}")
    
    # Get sample data
    print("\n\n--- Sample Data (First 10 Records) ---")
    try:
        cursor.execute("""
            SELECT TOP 10 *
            FROM dbo.VW_AdminstrationUnit
            ORDER BY SectionID
        """)
        records = cursor.fetchall()
        
        if records and cursor.description:
            col_names = [desc[0] for desc in cursor.description]
            print(f"\nColumns: {', '.join(col_names)}\n")
            
            for record in records:
                print("Record:")
                for i, col_name in enumerate(col_names):
                    print(f"  {col_name}: {record[i]}")
                print()
    except Exception as e:
        print(f"Error querying data: {e}")
    
    # Get unique sections, departments, administrations
    print("\n--- Summary Statistics ---")
    try:
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT SectionID) as UniqueSections,
                COUNT(DISTINCT DepartmentName) as UniqueDepartments
            FROM dbo.VW_AdminstrationUnit
            WHERE SectionID IS NOT NULL
        """)
        stats = cursor.fetchone()
        print(f"\n  Unique Sections: {stats.UniqueSections}")
        print(f"  Unique Departments: {stats.UniqueDepartments}")
    except Exception as e:
        print(f"Error getting statistics: {e}")
    
    print("\n" + "="*80)
    
    cursor.close()
    conn.close()

if __name__ == '__main__':
    main()
