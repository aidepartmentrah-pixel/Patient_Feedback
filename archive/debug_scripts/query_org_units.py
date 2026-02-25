"""
Query organizational unit data
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
    print("🏢 ORGANIZATIONAL UNITS ANALYSIS")
    print("="*80)
    
    # Check AdminsrationUnit table (note the typo in the actual table name)
    print("\n--- 1. AdminsrationUnit Table ---")
    try:
        cursor.execute("""
            SELECT TOP 20
                AdministrationID,
                AdministrationNameAr,
                AdministrationNameEn
            FROM dbo.AdminsrationUnit
            ORDER BY AdministrationID
        """)
        admin_units = cursor.fetchall()
        print(f"\nFound {len(admin_units)} administration units (showing first 20):\n")
        for unit in admin_units:
            print(f"  ID: {unit.AdministrationID} - {unit.AdministrationNameEn} / {unit.AdministrationNameAr}")
    except Exception as e:
        print(f"Error querying AdminsrationUnit: {e}")
    
    # Check for Department/Section tables
    print("\n\n--- 2. Looking for Department/Section Tables ---")
    cursor.execute("""
        SELECT 
            c.TABLE_NAME,
            c.COLUMN_NAME,
            c.DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS c
        WHERE c.TABLE_SCHEMA = 'dbo'
        AND (
            c.COLUMN_NAME LIKE '%Department%'
            OR c.COLUMN_NAME LIKE '%Section%'
        )
        AND c.TABLE_NAME NOT LIKE 'APP_%'
        ORDER BY c.TABLE_NAME, c.ORDINAL_POSITION
    """)
    dept_cols = cursor.fetchall()
    
    if dept_cols:
        print(f"\nFound columns related to Department/Section:")
        current_table = None
        for col in dept_cols:
            if col.TABLE_NAME != current_table:
                print(f"\n  Table: {col.TABLE_NAME}")
                current_table = col.TABLE_NAME
            print(f"    - {col.COLUMN_NAME} ({col.DATA_TYPE})")
    else:
        print("\n⚠️  No Department/Section columns found in non-APP tables")
    
    # Check APP_LOOKUP tables for organizational data
    print("\n\n--- 3. APP_LOOKUP Tables (Potential Org Data) ---")
    cursor.execute("""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = 'dbo'
        AND TABLE_NAME LIKE 'APP_LOOKUP_%'
        ORDER BY TABLE_NAME
    """)
    lookup_tables = cursor.fetchall()
    print(f"\nFound {len(lookup_tables)} lookup tables:")
    for table in lookup_tables:
        print(f"  - {table.TABLE_NAME}")
    
    # Check IncidentRequest table structure
    print("\n\n--- 4. IncidentRequest Table (Original System) ---")
    try:
        cursor.execute("""
            SELECT TOP 3
                COLUMN_NAME,
                DATA_TYPE,
                CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'IncidentRequest'
            AND TABLE_SCHEMA = 'dbo'
            ORDER BY ORDINAL_POSITION
        """)
        inc_cols = cursor.fetchall()
        print(f"\nIncidentRequest columns (first 3):")
        for col in inc_cols:
            max_len = f"({col.CHARACTER_MAXIMUM_LENGTH})" if col.CHARACTER_MAXIMUM_LENGTH else ""
            print(f"  - {col.COLUMN_NAME}: {col.DATA_TYPE}{max_len}")
        
        # Sample data
        cursor.execute("""
            SELECT TOP 1
                *
            FROM dbo.IncidentRequest
        """)
        sample = cursor.fetchone()
        if sample:
            print(f"\n  Sample record exists (columns: {len(sample)} fields)")
    except Exception as e:
        print(f"Error querying IncidentRequest: {e}")
    
    print("\n" + "="*80)
    
    cursor.close()
    conn.close()

if __name__ == '__main__':
    main()
