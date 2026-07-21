"""
Check IncidentRequestCase table schema
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from core.database import get_connection


def check_schema():
    """Check table schema"""
    conn = get_connection()
    cursor = conn.cursor()
    
    print("=" * 80)
    print("IncidentRequestCase TABLE SCHEMA")
    print("=" * 80)
    
    cursor.execute("""
        SELECT 
            COLUMN_NAME, 
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'IncidentRequestCase'
        ORDER BY ORDINAL_POSITION
    """)
    
    print("\n📋 Columns:")
    for row in cursor.fetchall():
        col_name, data_type, nullable, default = row
        null_text = "NULL" if nullable == 'YES' else "NOT NULL"
        default_text = f" DEFAULT {default}" if default else ""
        print(f"  {col_name:30} {data_type:15} {null_text:10}{default_text}")
    
    # Check foreign keys
    print("\n🔗 Foreign Keys:")
    cursor.execute("""
        SELECT 
            OBJECT_NAME(fkc.constraint_object_id) AS ConstraintName,
            COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS ColumnName,
            OBJECT_NAME(fkc.referenced_object_id) AS ReferencedTable,
            COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS ReferencedColumn
        FROM sys.foreign_key_columns fkc
        WHERE fkc.parent_object_id = OBJECT_ID('IncidentRequestCase')
    """)
    
    for row in cursor.fetchall():
        print(f"  {row[1]} → {row[2]}.{row[3]}")
    
    print("\n" + "=" * 80)
    
    cursor.close()
    conn.close()


if __name__ == "__main__":
    check_schema()
