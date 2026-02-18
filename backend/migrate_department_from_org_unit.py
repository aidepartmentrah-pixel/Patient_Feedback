"""
Migrate Department Names from Org Units
Copies Org Unit names to DepartmentDisplayName for users where department is Unknown/NULL.

This script:
1. Finds all users with Unknown or NULL DepartmentDisplayName
2. Looks up their assigned Org Unit(s) from APP_UserRoleScope
3. Updates DepartmentDisplayName with the Org Unit name
4. Reports on changes made

Usage:
    python migrate_department_from_org_unit.py
"""

from core.database import get_connection


def check_users_with_unknown_departments():
    """Check how many users have Unknown or NULL departments."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        print("\n" + "=" * 80)
        print("CHECKING USERS WITH UNKNOWN/NULL DEPARTMENTS")
        print("=" * 80)
        
        # Count users with Unknown or NULL departments
        query = """
            SELECT COUNT(*) AS count
            FROM dbo.APP_Users
            WHERE DepartmentDisplayName IS NULL 
               OR DepartmentDisplayName = ''
               OR DepartmentDisplayName = 'Unknown'
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        unknown_count = result.count if result else 0
        
        print(f"\n📊 Total users with Unknown/NULL departments: {unknown_count}")
        
        # Show sample of users
        if unknown_count > 0:
            print("\n📋 Sample users (first 10):")
            print("-" * 80)
            
            query = """
                SELECT TOP 10
                    u.UserID,
                    u.Username,
                    u.DisplayName,
                    u.DepartmentDisplayName,
                    r.RoleCode,
                    org.Name AS OrgUnitName
                FROM dbo.APP_Users u
                LEFT JOIN dbo.APP_UserRoleScope urs ON u.UserID = urs.UserID
                LEFT JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
                LEFT JOIN dbo.AdminsrationUnit org ON urs.OrgUnitID = org.UniqueID
                WHERE u.DepartmentDisplayName IS NULL 
                   OR u.DepartmentDisplayName = ''
                   OR u.DepartmentDisplayName = 'Unknown'
                ORDER BY u.UserID
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            for row in rows:
                dept = row.DepartmentDisplayName if row.DepartmentDisplayName else 'NULL'
                org_unit = row.OrgUnitName if row.OrgUnitName else 'NO ORG UNIT'
                print(f"  UserID {row.UserID:3d} | {row.Username:20s} | "
                      f"Dept: {dept:15s} | Org: {org_unit}")
        
        return unknown_count
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return 0
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def preview_migration():
    """Preview what changes will be made without actually updating."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        print("\n" + "=" * 80)
        print("MIGRATION PREVIEW - NO CHANGES WILL BE MADE")
        print("=" * 80)
        
        # Query to show what would be updated
        query = """
            SELECT 
                u.UserID,
                u.Username,
                u.DisplayName,
                u.DepartmentDisplayName AS CurrentDepartment,
                org.Name AS NewDepartment,
                r.RoleCode
            FROM dbo.APP_Users u
            INNER JOIN dbo.APP_UserRoleScope urs ON u.UserID = urs.UserID
            INNER JOIN dbo.AdminsrationUnit org ON urs.OrgUnitID = org.UniqueID
            LEFT JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
            WHERE (u.DepartmentDisplayName IS NULL 
                   OR u.DepartmentDisplayName = ''
                   OR u.DepartmentDisplayName = 'Unknown')
              AND org.Name IS NOT NULL
            ORDER BY u.UserID
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            print("\n✅ No users need department updates!")
            return 0
        
        print(f"\n📋 {len(rows)} users will be updated:")
        print("-" * 80)
        print(f"{'UserID':<8} {'Username':<20} {'Current Dept':<20} {'→ New Dept':<30}")
        print("-" * 80)
        
        for row in rows:
            current = row.CurrentDepartment if row.CurrentDepartment else 'NULL'
            new_dept = row.NewDepartment
            print(f"{row.UserID:<8} {row.Username:<20} {current:<20} → {new_dept:<30}")
        
        return len(rows)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return 0
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def execute_migration(dry_run=True):
    """
    Execute the migration to copy Org Unit names to Department names.
    
    Args:
        dry_run: If True, shows changes but doesn't commit. If False, commits changes.
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        print("\n" + "=" * 80)
        if dry_run:
            print("DRY RUN MODE - CHANGES WILL BE ROLLED BACK")
        else:
            print("EXECUTING MIGRATION - CHANGES WILL BE COMMITTED")
        print("=" * 80)
        
        # Step 1: Get list of users to update
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
        
        print(f"\n📊 Found {len(users_to_update)} users to update")
        
        # Step 2: Update each user
        update_query = """
            UPDATE dbo.APP_Users
            SET DepartmentDisplayName = ?
            WHERE UserID = ?
        """
        
        updated_count = 0
        failed_count = 0
        
        print("\n🔄 Updating users...")
        print("-" * 80)
        
        for user in users_to_update:
            try:
                cursor.execute(update_query, (user.NewDept, user.UserID))
                updated_count += 1
                
                old_dept = user.OldDept if user.OldDept else 'NULL'
                print(f"  ✓ UserID {user.UserID:3d} | {user.Username:<20s} | "
                      f"{old_dept:<15s} → {user.NewDept}")
                
            except Exception as e:
                failed_count += 1
                print(f"  ✗ UserID {user.UserID:3d} | {user.Username:<20s} | ERROR: {str(e)}")
        
        # Step 3: Commit or rollback
        if dry_run:
            conn.rollback()
            print("\n🔄 Changes rolled back (dry run mode)")
        else:
            conn.commit()
            print("\n✅ Changes committed to database")
        
        # Step 4: Summary
        print("\n" + "=" * 80)
        print("MIGRATION SUMMARY")
        print("=" * 80)
        print(f"  ✓ Successfully updated: {updated_count}")
        print(f"  ✗ Failed updates:       {failed_count}")
        print(f"  📊 Total processed:     {len(users_to_update)}")
        
        if dry_run:
            print("\n⚠️  This was a DRY RUN - no actual changes were made")
            print("   Run with dry_run=False to commit changes")
        else:
            print("\n✅ Migration completed successfully!")
        
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        print("   Transaction rolled back - no changes made")
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def verify_migration():
    """Verify the migration was successful."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        print("\n" + "=" * 80)
        print("VERIFICATION - CHECKING REMAINING UNKNOWN DEPARTMENTS")
        print("=" * 80)
        
        # Count remaining users with Unknown departments
        query = """
            SELECT COUNT(*) AS count
            FROM dbo.APP_Users
            WHERE DepartmentDisplayName IS NULL 
               OR DepartmentDisplayName = ''
               OR DepartmentDisplayName = 'Unknown'
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        remaining = result.count if result else 0
        
        print(f"\n📊 Remaining users with Unknown/NULL departments: {remaining}")
        
        if remaining > 0:
            print("\n⚠️  Some users still have Unknown departments")
            print("   These may be users without org unit assignments")
            
            # Show remaining users
            query = """
                SELECT TOP 10
                    u.UserID,
                    u.Username,
                    u.DepartmentDisplayName,
                    COUNT(urs.OrgUnitID) AS OrgUnitCount
                FROM dbo.APP_Users u
                LEFT JOIN dbo.APP_UserRoleScope urs ON u.UserID = urs.UserID
                WHERE u.DepartmentDisplayName IS NULL 
                   OR u.DepartmentDisplayName = ''
                   OR u.DepartmentDisplayName = 'Unknown'
                GROUP BY u.UserID, u.Username, u.DepartmentDisplayName
                ORDER BY u.UserID
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            print("\n📋 Remaining users (sample):")
            print("-" * 80)
            for row in rows:
                dept = row.DepartmentDisplayName if row.DepartmentDisplayName else 'NULL'
                print(f"  UserID {row.UserID:3d} | {row.Username:<20s} | "
                      f"Dept: {dept:<15s} | Org Units: {row.OrgUnitCount}")
        else:
            print("\n✅ All users now have department names!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def main():
    """Main execution function."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "DEPARTMENT MIGRATION FROM ORG UNITS" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Step 1: Check current state
    unknown_count = check_users_with_unknown_departments()
    
    if unknown_count == 0:
        print("\n✅ No migration needed - all users have departments!")
        return
    
    # Step 2: Preview changes
    preview_count = preview_migration()
    
    if preview_count == 0:
        print("\n⚠️  Users with Unknown departments don't have org unit assignments")
        print("   Manual intervention may be required")
        return
    
    # Step 3: Ask for confirmation
    print("\n" + "=" * 80)
    print("MIGRATION OPTIONS")
    print("=" * 80)
    print("\n1. Run DRY RUN (preview changes, don't commit)")
    print("2. EXECUTE MIGRATION (commit changes to database)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        print("\nRunning DRY RUN...")
        execute_migration(dry_run=True)
        
    elif choice == "2":
        print("\n⚠️  WARNING: This will update the database!")
        confirm = input("Type 'YES' to confirm migration: ").strip()
        
        if confirm == "YES":
            print("\nExecuting migration...")
            execute_migration(dry_run=False)
            
            # Verify
            verify_migration()
        else:
            print("\n❌ Migration cancelled")
            
    else:
        print("\n👋 Exiting without changes")


if __name__ == "__main__":
    main()
