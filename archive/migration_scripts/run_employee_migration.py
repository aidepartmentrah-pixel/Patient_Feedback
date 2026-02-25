"""
Run Employee Table Migration
Adds columns to APP_IncidentCaseEmployee table
"""
from backend.core.database import get_connection


def run_migration():
    """Execute the ALTER TABLE migration"""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        print("=" * 60)
        print("Running Employee Table Migration")
        print("=" * 60)
        
        # Step 1: Add IncidentRequestCaseID column
        print("\n1. Adding IncidentRequestCaseID column...")
        try:
            cursor.execute("""
                ALTER TABLE dbo.APP_IncidentCaseEmployee
                ADD IncidentRequestCaseID INT NULL
            """)
            print("✅ IncidentRequestCaseID column added")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print("⚠️  Column already exists, skipping")
            else:
                raise
        
        # Step 2: Add IsPrimary column
        print("\n2. Adding IsPrimary column...")
        try:
            cursor.execute("""
                ALTER TABLE dbo.APP_IncidentCaseEmployee
                ADD IsPrimary BIT DEFAULT 0
            """)
            print("✅ IsPrimary column added")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print("⚠️  Column already exists, skipping")
            else:
                raise
        
        # Step 3: Add AssignedAt column
        print("\n3. Adding AssignedAt column...")
        try:
            cursor.execute("""
                ALTER TABLE dbo.APP_IncidentCaseEmployee
                ADD AssignedAt DATETIME DEFAULT GETDATE()
            """)
            print("✅ AssignedAt column added")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print("⚠️  Column already exists, skipping")
            else:
                raise
        
        # Step 4: Add AssignedByUserID column
        print("\n4. Adding AssignedByUserID column...")
        try:
            cursor.execute("""
                ALTER TABLE dbo.APP_IncidentCaseEmployee
                ADD AssignedByUserID INT NULL
            """)
            print("✅ AssignedByUserID column added")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print("⚠️  Column already exists, skipping")
            else:
                raise
        
        # Step 5: Add foreign key constraint
        print("\n5. Adding foreign key constraint...")
        try:
            cursor.execute("""
                ALTER TABLE dbo.APP_IncidentCaseEmployee
                ADD CONSTRAINT FK_IncidentCaseEmployee_Incident
                FOREIGN KEY (IncidentRequestCaseID) 
                REFERENCES dbo.APP_IncidentCase(IncidentRequestCaseID)
            """)
            print("✅ Foreign key constraint added")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print("⚠️  Constraint already exists, skipping")
            else:
                raise
        
        # Step 6: Add index for IncidentID
        print("\n6. Adding index on IncidentRequestCaseID...")
        try:
            cursor.execute("""
                CREATE NONCLUSTERED INDEX IX_IncidentCaseEmployee_IncidentID
                ON dbo.APP_IncidentCaseEmployee(IncidentRequestCaseID)
            """)
            print("✅ Index on IncidentRequestCaseID created")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print("⚠️  Index already exists, skipping")
            else:
                raise
        
        # Step 7: Add index for EmployeeID
        print("\n7. Adding index on EmployeeID...")
        try:
            cursor.execute("""
                CREATE NONCLUSTERED INDEX IX_IncidentCaseEmployee_EmployeeID
                ON dbo.APP_IncidentCaseEmployee(EmployeeID)
            """)
            print("✅ Index on EmployeeID created")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print("⚠️  Index already exists, skipping")
            else:
                raise
        
        conn.commit()
        
        print("\n" + "=" * 60)
        print("✅ Migration completed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Migration failed: {str(e)}")
        if conn:
            conn.rollback()
        import traceback
        traceback.print_exc()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    print("\n🔧 Employee Table Migration")
    print("This will add columns to APP_IncidentCaseEmployee table\n")
    
    success = run_migration()
    
    if success:
        print("\n✅ Ready to test employee linkage!")
        print("\nNext steps:")
        print("1. Start the backend server: cd backend && uvicorn main:app --reload")
        print("2. Run the test: python test_employee_linkage.py")
    else:
        print("\n❌ Migration failed! Check errors above.")
