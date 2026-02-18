"""
Execute Force Close Migration Script
Applies database schema changes for force-close feature.
"""
import sys
sys.path.insert(0, 'backend')

from core.database import get_connection

def run_migration():
    """Execute the force close migration."""
    print("=" * 70)
    print("FORCE CLOSE MIGRATION - STARTING")
    print("=" * 70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Step 1: Add columns to APP_AdministrativeSubcase
        print("\n[1/8] Adding columns to APP_AdministrativeSubcase...")
        try:
            cursor.execute("""
                ALTER TABLE dbo.APP_AdministrativeSubcase
                ADD ForceClosedAt DATETIME NULL,
                    ForceClosedByUserID INT NULL,
                    ForceCloseReason NVARCHAR(MAX) NULL
            """)
            conn.commit()
            print("✅ Columns added to APP_AdministrativeSubcase")
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "duplicate" in error_msg or "specified more than once" in error_msg:
                print("⚠️  Columns already exist in APP_AdministrativeSubcase (skipping)")
                conn.rollback()
            else:
                raise
        
        # Step 2: Add FK constraint for AdministrativeSubcase
        print("\n[2/8] Adding FK constraint to APP_AdministrativeSubcase...")
        try:
            cursor.execute("""
                ALTER TABLE dbo.APP_AdministrativeSubcase
                ADD CONSTRAINT FK_AdministrativeSubcase_ForceClosedByUser
                FOREIGN KEY (ForceClosedByUserID) REFERENCES dbo.APP_Users(UserID)
            """)
            conn.commit()
            print("✅ FK constraint added to APP_AdministrativeSubcase")
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "duplicate" in error_msg:
                print("⚠️  FK constraint already exists (skipping)")
                conn.rollback()
            else:
                raise
        
        # Step 3: Add columns to APP_IncidentCase
        print("\n[3/8] Adding columns to APP_IncidentCase...")
        try:
            cursor.execute("""
                ALTER TABLE dbo.APP_IncidentCase
                ADD ForceClosedAt DATETIME NULL,
                    ForceClosedByUserID INT NULL,
                    ForceCloseReason NVARCHAR(MAX) NULL
            """)
            conn.commit()
            error_msg = str(e).lower()
            if "already exists" in error_msg or "duplicate" in error_msg or "specified more than once" in error_msg:
                print("⚠️  Columns already exist in APP_IncidentCase (skipping)")
                conn.rollback(
            if "already exists" in str(e) or "duplicate" in str(e).lower():
                print("⚠️  Columns already exist in APP_IncidentCase (skipping)")
            else:
                raise
        
        # Step 4: Add FK constraint for IncidentCase
        print("\n[4/8] Adding FK constraint to APP_IncidentCase...")
        try:
            cursor.execute("""
                ALTER TABLE dbo.APP_IncidentCase
                ADD CONSTRAINT FK_IncidentCase_ForceClosedByUser
                FOREIGN KEY (ForceClosedByUserID) REFERENCES dbo.APP_Users(UserID)
            """)
            conn.commit()
            error_msg = str(e).lower()
            if "already exists" in error_msg or "duplicate" in error_msg:
                print("⚠️  FK constraint already exists (skipping)")
                conn.rollback(
            if "already exists" in str(e) or "duplicate" in str(e).lower():
                print("⚠️  FK constraint already exists (skipping)")
            else:
                raise
        
        # Step 5: Create index on AdministrativeSubcase
        print("\n[5/8] Creating index on APP_AdministrativeSubcase...")
        try:
            cursor.execute("""
                CREATE NONCLUSTERED INDEX IX_AdministrativeSubcase_ForceClosedAt
                ON dbo.APP_AdministrativeSubcase(ForceClosedAt)
                WHERE ForceClosedAt IS NOT NULL
            """)
            conn.commit()
            error_msg = str(e).lower()
            if "already exists" in error_msg or "duplicate" in error_msg:
                print("⚠️  Index already exists (skipping)")
                conn.rollback(
            if "already exists" in str(e) or "duplicate" in str(e).lower():
                print("⚠️  Index already exists (skipping)")
            else:
                raise
        
        # Step 6: Create index on IncidentCase
        print("\n[6/8] Creating index on APP_IncidentCase...")
        try:
            cursor.execute("""
                CREATE NONCLUSTERED INDEX IX_IncidentCase_ForceClosedAt
                ON dbo.APP_IncidentCase(ForceClosedAt)
                WHERE ForceClosedAt IS NOT NULL
            """)
            conn.commit()
            error_msg = str(e).lower()
            if "already exists" in error_msg or "duplicate" in error_msg:
                print("⚠️  Index already exists (skipping)")
                conn.rollback(
            if "already exists" in str(e) or "duplicate" in str(e).lower():
                print("⚠️  Index already exists (skipping)")
            else:
                raise
        
        # Step 7: Verify columns exist
        print("\n[7/8] Verifying schema changes...")
        cursor.execute("""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = 'APP_AdministrativeSubcase' 
            AND COLUMN_NAME IN ('ForceClosedAt', 'ForceClosedByUserID', 'ForceCloseReason')
        """)
        subcase_cols = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = 'APP_IncidentCase' 
            AND COLUMN_NAME IN ('ForceClosedAt', 'ForceClosedByUserID', 'ForceCloseReason')
        """)
        incident_cols = [row[0] for row in cursor.fetchall()]
        
        print(f"   APP_AdministrativeSubcase columns: {', '.join(subcase_cols)}")
        print(f"   APP_IncidentCase columns: {', '.join(incident_cols)}")
        
        if len(subcase_cols) == 3 and len(incident_cols) == 3:
            print("✅ All columns verified")
        else:
            print("⚠️  Some columns may be missing")
        
        # Step 8: Test write/read
        print("\n[8/8] Testing column accessibility...")
        cursor.execute("""
            SELECT TOP 1 
                SubcaseID,
                ForceClosedAt,
                ForceClosedByUserID,
                ForceCloseReason
            FROM dbo.APP_AdministrativeSubcase
        """)
        cursor.fetchone()
        print("✅ Subcase columns are accessible")
        
        cursor.execute("""
            SELECT TOP 1 
                IncidentRequestCaseID,
                ForceClosedAt,
                ForceClosedByUserID,
                ForceCloseReason
            FROM dbo.APP_IncidentCase
        """)
        cursor.fetchone()
        print("✅ Incident columns are accessible")
        
        print("\n" + "=" * 70)
        print("✅ MIGRATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return False
        
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
