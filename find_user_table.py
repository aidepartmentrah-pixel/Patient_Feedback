"""
Find the correct User table name in the database.
"""
import sys
sys.path.insert(0, 'backend')

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

try:
    print("Searching for User table...")
    cursor.execute("""
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME LIKE '%User%'
        ORDER BY TABLE_NAME
    """)
    
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\nFound {len(tables)} table(s) with 'User' in the name:")
    for table in tables:
        print(f"  - {table}")
    
    # Check AdminsrationUnit which we know exists
    print("\n\nChecking CreatedByUserID references in existing tables...")
    cursor.execute("""
        SELECT 
            fk.name AS FK_name,
            tp.name AS parent_table,
            ref.name AS referenced_table,
            cp.name AS parent_column,
            cr.name AS referenced_column
        FROM sys.foreign_keys fk
        INNER JOIN sys.tables tp ON fk.parent_object_id = tp.object_id
        INNER JOIN sys.tables ref ON fk.referenced_object_id = ref.object_id
        INNER JOIN sys.foreign_key_columns fkc ON fkc.constraint_object_id = fk.object_id
        INNER JOIN sys.columns cp ON fkc.parent_column_id = cp.column_id AND fkc.parent_object_id = cp.object_id
        INNER JOIN sys.columns cr ON fkc.referenced_column_id = cr.column_id AND fkc.referenced_object_id = cr.object_id
        WHERE cp.name LIKE '%User%'
        ORDER BY tp.name
    """)
    
    print("\nExisting FK constraints on User columns:")
    fks = cursor.fetchall()
    if fks:
        for fk in fks[:10]:  # Show first 10
            print(f"  {fk.parent_table}.{fk.parent_column} -> {fk.referenced_table}.{fk.referenced_column}")
    else:
        print("  No FK constraints found on User columns")
    
    # Check CreatedByUserID in APP_AdministrativeSubcase
    print("\n\nChecking APP_AdministrativeSubcase columns...")
    cursor.execute("""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'APP_AdministrativeSubcase' 
        AND COLUMN_NAME LIKE '%User%'
    """)
    cols = cursor.fetchall()
    if cols:
        print("User-related columns in APP_AdministrativeSubcase:")
        for col in cols:
            print(f"  {col.COLUMN_NAME} ({col.DATA_TYPE}, nullable={col.IS_NULLABLE})")
    
finally:
    cursor.close()
    conn.close()
