"""
Create a UNIVERSAL_SECTION user for operational use.

This creates a user that can:
- See ALL section-level subcases in inbox (no scope filter)
- Process subcases across all sections in one login
- Direct approve from submitted_to_section to admin_approved

Run this once to create the operational bridge user.
"""
import sys
sys.path.insert(0, "backend")

from backend.core.database import get_connection
import bcrypt


def create_universal_section_user(
    username: str = "universal_section_user",
    password: str = "Hospital2026!",
    display_name: str = "Universal Section Admin"
):
    """Create a user with UNIVERSAL_SECTION role."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check if user already exists
        cursor.execute("SELECT UserID FROM dbo.APP_Users WHERE Username = ?", (username,))
        existing = cursor.fetchone()
        
        if existing:
            print(f"ℹ️ User '{username}' already exists with UserID={existing.UserID}")
            return existing.UserID
        
        # Hash the password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Create the user
        cursor.execute("""
            INSERT INTO dbo.APP_Users (Username, PasswordHash, DisplayName, IsActive)
            OUTPUT INSERTED.UserID
            VALUES (?, ?, ?, 1)
        """, (username, password_hash, display_name))
        
        user_id = cursor.fetchone().UserID
        print(f"✅ Created user '{username}' with UserID={user_id}")
        
        # Get UNIVERSAL_SECTION role ID
        cursor.execute("SELECT RoleID FROM dbo.APP_Roles WHERE RoleCode = 'UNIVERSAL_SECTION'")
        role_row = cursor.fetchone()
        
        if not role_row:
            print("❌ UNIVERSAL_SECTION role not found in database!")
            return None
        
        role_id = role_row.RoleID
        
        # Get a top-level Administration unit for the scope
        # UNIVERSAL_SECTION doesn't use scope filtering, but we need a valid org unit for the system
        cursor.execute("""
            SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Type = 323
            ORDER BY UniqueID
        """)
        org_row = cursor.fetchone()
        
        if not org_row:
            print("❌ No Administration unit found for scope assignment!")
            return None
        
        org_unit_id = org_row.UniqueID
        
        # Create role scope assignment (OrgUnitType is required - use ADMINISTRATION for universal scope)
        cursor.execute("""
            INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
            VALUES (?, ?, ?, 'ADMINISTRATION')
        """, (user_id, role_id, org_unit_id))
        
        print(f"✅ Assigned UNIVERSAL_SECTION role (RoleID={role_id}) with OrgUnitID={org_unit_id}")
        
        conn.commit()
        
        print(f"\n{'='*50}")
        print("UNIVERSAL_SECTION User Created Successfully")
        print(f"{'='*50}")
        print(f"  Username: {username}")
        print(f"  Password: {password}")
        print(f"  Role: UNIVERSAL_SECTION")
        print(f"\nThis user can now:")
        print("  • See ALL section-level subcases in workflow inbox")
        print("  • Process subcases across all sections")
        print("  • Direct approve to admin_approved status")
        print(f"{'='*50}")
        
        return user_id
    
    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    create_universal_section_user()
