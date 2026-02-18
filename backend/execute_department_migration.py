"""
Quick execution script for department migration.
Automatically executes the migration without interactive prompts.
"""

from core.database import get_connection


def execute_migration_direct():
    """Execute the migration directly without prompts."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        print("\n" + "=" * 80)
        print("EXECUTING DEPARTMENT MIGRATION FROM ORG UNITS")
        print("=" * 80)
        
        # Step 1: Check current state
        query = """
            SELECT COUNT(*) AS count
            FROM dbo.APP_Users
            WHERE DepartmentDisplayName IS NULL 
               OR DepartmentDisplayName = ''
               OR DepartmentDisplayName = 'Unknown'
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        before_count = result.count if result else 0
        
        print(f"\n📊 Users with Unknown/NULL departments: {before_count}")
        
        # Step 2: Get users to update
        select_query = """
            SELECT 
                u.UserID,
                u.Username,
                u.DepartmentDisplayName AS OldDept,
                org.Name AS NewDept
            FROM dbo.APP_Users u
            INNER JOIN dbo.APP_UserRoleScope urs ON u.UserID = urs.UserID
            INNER JOIN dbo.AdminsrationUnit org ON urs.OrgUnitID = org.UniqueID
            WHERE (u.DepartmentDisplayName IS NULL 
                   OR u.DepartmentDisplayName = ''
                   OR u.DepartmentDisplayName = 'Unknown')
              AND org.Name IS NOT NULL
        """
        
        cursor.execute(select_query)
        users_to_update = cursor.fetchall()
        
        if not users_to_update:
            print("\n✅ No users need department updates!")
            return
        
        print(f"\n🔄 Updating {len(users_to_update)} users...")
        print("-" * 80)
        
        # Step 3: Update each user
        update_query = """
            UPDATE dbo.APP_Users
            SET DepartmentDisplayName = ?
            WHERE UserID = ?
        """
        
        updated_count = 0
        failed_count = 0
        
        for user in users_to_update:
            try:
                cursor.execute(update_query, (user.NewDept, user.UserID))
                updated_count += 1
                
                old_dept = user.OldDept if user.OldDept else 'NULL'
                if updated_count <= 20:  # Show first 20
                    print(f"  ✓ UserID {user.UserID:3d} | {user.Username:<25s} | "
                          f"{old_dept:<15s} → {user.NewDept}")
                elif updated_count == 21:
                    print(f"  ... ({len(users_to_update) - 20} more users)")
                
            except Exception as e:
                failed_count += 1
                print(f"  ✗ UserID {user.UserID:3d} | {user.Username:<25s} | ERROR: {str(e)}")
        
        # Step 4: Commit changes
        conn.commit()
        print("\n✅ Changes committed to database")
        
        # Step 5: Verify
        cursor.execute(query)
        result = cursor.fetchone()
        after_count = result.count if result else 0
        
        # Summary
        print("\n" + "=" * 80)
        print("MIGRATION SUMMARY")
        print("=" * 80)
        print(f"  Before migration:       {before_count} users with Unknown departments")
        print(f"  ✓ Successfully updated: {updated_count} users")
        print(f"  ✗ Failed updates:       {failed_count} users")
        print(f"  After migration:        {after_count} users with Unknown departments")
        print(f"\n  📊 Improvement: {before_count - after_count} users now have departments!")
        print("\n✅ Migration completed successfully!")
        
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        print("   Transaction rolled back - no changes made")
        raise
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DEPARTMENT MIGRATION - AUTO EXECUTE" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")
    
    execute_migration_direct()
