"""
PHASE K — KDB1 — Quick table verification and documentation
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from core.database import get_connection


def verify_table():
    """Query and display table structure"""
    conn = get_connection()
    cursor = conn.cursor()
    
    print("=" * 80)
    print("APP_DataMigration_Map — Table Structure")
    print("=" * 80)
    
    # Columns
    print("\n📋 COLUMNS:")
    cursor.execute("""
        SELECT 
            COLUMN_NAME, 
            DATA_TYPE, 
            CHARACTER_MAXIMUM_LENGTH,
            IS_NULLABLE,
            COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'APP_DataMigration_Map'
        ORDER BY ORDINAL_POSITION
    """)
    
    for row in cursor.fetchall():
        col_name, data_type, max_len, nullable, default = row
        len_info = f"({max_len})" if max_len else ""
        null_info = "NULL" if nullable == 'YES' else "NOT NULL"
        default_info = f" DEFAULT {default}" if default else ""
        print(f"  {col_name:25} {data_type}{len_info:15} {null_info:10}{default_info}")
    
    # Primary Key
    print("\n🔑 PRIMARY KEY:")
    cursor.execute("""
        SELECT kcu.COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
        INNER JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
        ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
        WHERE tc.TABLE_NAME = 'APP_DataMigration_Map'
        AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
    """)
    
    for row in cursor.fetchall():
        print(f"  {row[0]}")
    
    # Unique Constraints
    print("\n🔒 UNIQUE CONSTRAINTS:")
    cursor.execute("""
        SELECT kcu.CONSTRAINT_NAME, kcu.COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
        INNER JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
        ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
        WHERE tc.TABLE_NAME = 'APP_DataMigration_Map'
        AND tc.CONSTRAINT_TYPE = 'UNIQUE'
    """)
    
    for row in cursor.fetchall():
        print(f"  {row[0]} ON {row[1]}")
    
    # Foreign Keys
    print("\n🔗 FOREIGN KEYS:")
    cursor.execute("""
        SELECT 
            fk.name AS FK_Name,
            OBJECT_NAME(fk.parent_object_id) AS TableName,
            COL_NAME(fc.parent_object_id, fc.parent_column_id) AS ColumnName,
            OBJECT_NAME(fk.referenced_object_id) AS ReferencedTable,
            COL_NAME(fc.referenced_object_id, fc.referenced_column_id) AS ReferencedColumn,
            fk.delete_referential_action_desc AS OnDelete
        FROM sys.foreign_keys AS fk
        INNER JOIN sys.foreign_key_columns AS fc 
        ON fk.object_id = fc.constraint_object_id
        WHERE OBJECT_NAME(fk.parent_object_id) = 'APP_DataMigration_Map'
    """)
    
    for row in cursor.fetchall():
        fk_name, table, col, ref_table, ref_col, on_delete = row
        print(f"  {fk_name}")
        print(f"    {table}.{col} → {ref_table}.{ref_col}")
        print(f"    ON DELETE: {on_delete}")
    
    # Indexes
    print("\n📊 INDEXES:")
    cursor.execute("""
        SELECT 
            i.name AS IndexName,
            i.type_desc AS IndexType,
            COL_NAME(ic.object_id, ic.column_id) AS ColumnName,
            i.is_unique
        FROM sys.indexes i
        INNER JOIN sys.index_columns ic 
        ON i.object_id = ic.object_id AND i.index_id = ic.index_id
        WHERE i.object_id = OBJECT_ID('dbo.APP_DataMigration_Map')
        AND i.name IS NOT NULL
        ORDER BY i.name, ic.key_ordinal
    """)
    
    current_index = None
    for row in cursor.fetchall():
        idx_name, idx_type, col, is_unique = row
        unique_text = "(UNIQUE)" if is_unique else ""
        if idx_name != current_index:
            print(f"  {idx_name} — {idx_type} {unique_text}")
            current_index = idx_name
        print(f"    → {col}")
    
    # Row count
    print("\n📈 ROW COUNT:")
    cursor.execute("SELECT COUNT(*) FROM APP_DataMigration_Map")
    count = cursor.fetchone()[0]
    print(f"  {count} rows")
    
    print("\n" + "=" * 80)
    print("✅ Table APP_DataMigration_Map is ready for Phase K migration")
    print("=" * 80)
    
    cursor.close()
    conn.close()


if __name__ == "__main__":
    verify_table()
