"""
Phase 1 Migration: Fix APP_IncidentCaseEmployee for Many-to-Many

Problem: EmployeeID is the PK, so each employee can only link to ONE incident.
Fix:     Add auto-increment ID as PK, allow same employee in multiple incidents.

Steps:
1. Drop existing PK constraint on EmployeeID
2. Add new ID column (IDENTITY auto-increment) as new PK
3. Add UNIQUE constraint on (EmployeeID, IncidentRequestCaseID) to prevent duplicates
4. Keep existing FK, indexes, and data intact
"""
import sys
sys.path.insert(0, '.')
from backend.core.database import get_connection


def run_migration():
    conn = get_connection()
    cursor = conn.cursor()
    
    print("=" * 70)
    print("PHASE 1 MIGRATION: APP_IncidentCaseEmployee → Many-to-Many")
    print("=" * 70)
    
    try:
        # ============================================================
        # STEP 1: Backup existing data count
        # ============================================================
        cursor.execute("SELECT COUNT(*) FROM dbo.APP_IncidentCaseEmployee")
        before_count = cursor.fetchone()[0]
        print(f"\n[STEP 1] Existing rows before migration: {before_count}")
        
        # ============================================================
        # STEP 2: Drop existing PK constraint
        # ============================================================
        print("\n[STEP 2] Dropping existing PK constraint...")
        
        # Find the PK constraint name dynamically
        cursor.execute("""
            SELECT kc.name AS constraint_name
            FROM sys.key_constraints kc
            WHERE kc.parent_object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee')
                AND kc.type = 'PK'
        """)
        pk_row = cursor.fetchone()
        
        if pk_row:
            pk_name = pk_row.constraint_name
            print(f"   Found PK constraint: {pk_name}")
            cursor.execute(f"ALTER TABLE dbo.APP_IncidentCaseEmployee DROP CONSTRAINT [{pk_name}]")
            print(f"   Dropped PK constraint: {pk_name}")
        else:
            print("   No PK constraint found (may already be dropped)")
        
        # ============================================================
        # STEP 3: Drop the old IX_IncidentCaseEmployee_EmployeeID index
        #         (redundant after we make a unique constraint)
        # ============================================================
        print("\n[STEP 3] Dropping old EmployeeID index if exists...")
        try:
            cursor.execute("""
                IF EXISTS (SELECT 1 FROM sys.indexes 
                           WHERE name = 'IX_IncidentCaseEmployee_EmployeeID' 
                           AND object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee'))
                    DROP INDEX IX_IncidentCaseEmployee_EmployeeID ON dbo.APP_IncidentCaseEmployee
            """)
            print("   Dropped IX_IncidentCaseEmployee_EmployeeID")
        except Exception as e:
            print(f"   Index drop skipped: {e}")
        
        # ============================================================
        # STEP 4: Add new ID column as IDENTITY
        # ============================================================
        print("\n[STEP 4] Adding new ID column (IDENTITY)...")
        
        # Check if ID column already exists
        cursor.execute("""
            SELECT COUNT(*) FROM sys.columns 
            WHERE object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee') 
            AND name = 'ID'
        """)
        id_exists = cursor.fetchone()[0] > 0
        
        if not id_exists:
            cursor.execute("""
                ALTER TABLE dbo.APP_IncidentCaseEmployee 
                ADD ID INT IDENTITY(1,1) NOT NULL
            """)
            print("   Added ID column (INT IDENTITY)")
        else:
            print("   ID column already exists, skipping")
        
        # ============================================================
        # STEP 5: Add new PK on ID column
        # ============================================================
        print("\n[STEP 5] Adding new PK constraint on ID...")
        
        # Check if any PK exists now
        cursor.execute("""
            SELECT COUNT(*) FROM sys.key_constraints 
            WHERE parent_object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee') 
            AND type = 'PK'
        """)
        has_pk = cursor.fetchone()[0] > 0
        
        if not has_pk:
            cursor.execute("""
                ALTER TABLE dbo.APP_IncidentCaseEmployee 
                ADD CONSTRAINT PK_IncidentCaseEmployee_ID PRIMARY KEY CLUSTERED (ID)
            """)
            print("   Added PK: PK_IncidentCaseEmployee_ID on (ID)")
        else:
            print("   PK already exists, skipping")
        
        # ============================================================
        # STEP 6: Add UNIQUE constraint on (EmployeeID, IncidentRequestCaseID)
        # ============================================================
        print("\n[STEP 6] Adding UNIQUE constraint on (EmployeeID, IncidentRequestCaseID)...")
        
        # Check if unique constraint already exists
        cursor.execute("""
            SELECT COUNT(*) FROM sys.indexes 
            WHERE name = 'UQ_Employee_Incident' 
            AND object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee')
        """)
        uq_exists = cursor.fetchone()[0] > 0
        
        if not uq_exists:
            cursor.execute("""
                ALTER TABLE dbo.APP_IncidentCaseEmployee 
                ADD CONSTRAINT UQ_Employee_Incident 
                UNIQUE (EmployeeID, IncidentRequestCaseID)
            """)
            print("   Added UNIQUE: UQ_Employee_Incident on (EmployeeID, IncidentRequestCaseID)")
        else:
            print("   UNIQUE constraint already exists, skipping")
        
        # ============================================================
        # STEP 7: Commit
        # ============================================================
        conn.commit()
        print("\n[STEP 7] Migration COMMITTED successfully!")
        
        # ============================================================
        # STEP 8: Verify
        # ============================================================
        print("\n[STEP 8] Verification...")
        
        # Verify data preserved
        cursor.execute("SELECT COUNT(*) FROM dbo.APP_IncidentCaseEmployee")
        after_count = cursor.fetchone()[0]
        print(f"   Rows after migration: {after_count} (was {before_count})")
        assert after_count == before_count, f"DATA LOSS! Before={before_count}, After={after_count}"
        
        # Verify new schema
        cursor.execute("""
            SELECT c.name, t.name AS type_name, c.is_identity, c.is_nullable
            FROM sys.columns c
            JOIN sys.types t ON c.user_type_id = t.user_type_id
            WHERE c.object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee')
            ORDER BY c.column_id
        """)
        print("\n   New column structure:")
        for row in cursor.fetchall():
            marker = " *** NEW PK ***" if row.name == 'ID' else ""
            print(f"     {row.name}: {row.type_name}, identity={row.is_identity}, nullable={row.is_nullable}{marker}")
        
        # Verify constraints
        cursor.execute("""
            SELECT kc.name, kc.type_desc
            FROM sys.key_constraints kc
            WHERE kc.parent_object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee')
        """)
        print("\n   Constraints:")
        for row in cursor.fetchall():
            print(f"     {row.name} ({row.type_desc})")
        
        # Verify unique constraint
        cursor.execute("""
            SELECT i.name, i.is_unique
            FROM sys.indexes i
            WHERE i.object_id = OBJECT_ID('dbo.APP_IncidentCaseEmployee')
            AND i.name = 'UQ_Employee_Incident'
        """)
        uq_row = cursor.fetchone()
        if uq_row:
            print(f"\n   UNIQUE constraint verified: {uq_row.name} (is_unique={uq_row.is_unique})")
        
        # Verify existing data with new ID column
        cursor.execute("""
            SELECT ID, EmployeeID, FullName, IncidentRequestCaseID, IsPrimary 
            FROM dbo.APP_IncidentCaseEmployee 
            ORDER BY ID
        """)
        print("\n   Data with new ID column:")
        for row in cursor.fetchall():
            print(f"     ID={row.ID}, EmpID={row.EmployeeID}, Name={row.FullName}, "
                  f"IncidentID={row.IncidentRequestCaseID}, Primary={row.IsPrimary}")
        
        print("\n" + "=" * 70)
        print("PHASE 1 MIGRATION COMPLETE - ALL VERIFICATIONS PASSED!")
        print("=" * 70)
        
    except Exception as e:
        conn.rollback()
        print(f"\n[ERROR] Migration FAILED and ROLLED BACK: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    run_migration()
