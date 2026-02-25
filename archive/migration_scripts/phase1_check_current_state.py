"""
Phase 1 Pre-Check: Examine current APP_IncidentCaseEmployee table state
"""
import sys
sys.path.insert(0, '.')
from backend.core.database import get_connection

def check_current_state():
    conn = get_connection()
    cursor = conn.cursor()
    
    print("=" * 70)
    print("PHASE 1 PRE-CHECK: APP_IncidentCaseEmployee Current State")
    print("=" * 70)
    
    # 1. Check existing data
    print("\n1. EXISTING DATA:")
    cursor.execute("SELECT * FROM dbo.APP_IncidentCaseEmployee ORDER BY EmployeeID")
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    print(f"   Columns: {columns}")
    print(f"   Row count: {len(rows)}")
    for row in rows:
        row_dict = dict(zip(columns, row))
        print(f"   - EmpID={row_dict.get('EmployeeID')}, "
              f"Name={row_dict.get('FullName')}, "
              f"IncidentID={row_dict.get('IncidentRequestCaseID')}, "
              f"IsPrimary={row_dict.get('IsPrimary')}, "
              f"AssignedAt={row_dict.get('AssignedAt')}")
    
    # 2. Check current PK constraint name
    print("\n2. PRIMARY KEY CONSTRAINT:")
    cursor.execute("""
        SELECT 
            kc.name AS constraint_name,
            c.name AS column_name
        FROM sys.key_constraints kc
        JOIN sys.index_columns ic ON kc.unique_index_id = ic.index_id 
            AND kc.parent_object_id = ic.object_id
        JOIN sys.columns c ON ic.object_id = c.object_id 
            AND ic.column_id = c.column_id
        WHERE kc.parent_object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee')
            AND kc.type = 'PK'
    """)
    pk_rows = cursor.fetchall()
    for row in pk_rows:
        print(f"   Constraint: {row.constraint_name}, Column: {row.column_name}")
    
    # 3. Check FK constraints
    print("\n3. FOREIGN KEY CONSTRAINTS:")
    cursor.execute("""
        SELECT 
            fk.name AS fk_name,
            COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS fk_column,
            OBJECT_NAME(fkc.referenced_object_id) AS referenced_table,
            COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS referenced_column
        FROM sys.foreign_keys fk
        JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
        WHERE fk.parent_object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee')
    """)
    fk_rows = cursor.fetchall()
    for row in fk_rows:
        print(f"   FK: {row.fk_name} ({row.fk_column} -> {row.referenced_table}.{row.referenced_column})")
    
    # 4. Check all indexes
    print("\n4. INDEXES:")
    cursor.execute("""
        SELECT 
            i.name AS index_name,
            i.type_desc,
            i.is_unique,
            i.is_primary_key,
            STRING_AGG(c.name, ', ') AS columns
        FROM sys.indexes i
        JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
        JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
        WHERE i.object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee')
        GROUP BY i.name, i.type_desc, i.is_unique, i.is_primary_key
    """)
    idx_rows = cursor.fetchall()
    for row in idx_rows:
        print(f"   Index: {row.index_name} (type={row.type_desc}, unique={row.is_unique}, pk={row.is_primary_key}, cols={row.columns})")
    
    # 5. Check if ID column already exists
    print("\n5. COLUMN CHECK (does ID already exist?):")
    cursor.execute("""
        SELECT c.name, t.name AS type_name, c.is_identity, c.is_nullable
        FROM sys.columns c
        JOIN sys.types t ON c.user_type_id = t.user_type_id
        WHERE c.object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee')
        ORDER BY c.column_id
    """)
    col_rows = cursor.fetchall()
    id_exists = False
    for row in col_rows:
        is_id = " <-- ID EXISTS!" if row.name == 'ID' else ""
        if row.name == 'ID':
            id_exists = True
        print(f"   {row.name}: {row.type_name}, identity={row.is_identity}, nullable={row.is_nullable}{is_id}")
    
    print(f"\n   ID column exists: {id_exists}")
    
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 70)
    print("PRE-CHECK COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    check_current_state()
