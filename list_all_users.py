"""
List all users with their credentials and roles from the database.
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

import pyodbc

def get_connection():
    """Get SQL Server database connection."""
    return pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=192.168.68.110;'
        'DATABASE=PatientFeedback;'
        'UID=sa;'
        'PWD=Ph@rmacy123;'
        'TrustServerCertificate=yes;'
    )

def list_all_users():
    """Get all users from database with their roles."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get all users
        cursor.execute("""
            SELECT 
                u.UserID,
                u.Username,
                u.PasswordHash,
                u.IsActive
            FROM dbo.APP_Users u
            ORDER BY u.UserID
        """)
        
        users = cursor.fetchall()
        
        print("\n" + "="*80)
        print("🔐 USER ACCOUNTS IN DATABASE")
        print("="*80)
        print(f"\nTotal Users: {len(users)}\n")
        
        for user in users:
            user_id = user.UserID
            username = user.Username
            password_hash = user.PasswordHash
            is_active = bool(user.IsActive)
            
            # Get user's roles
            cursor.execute("""
                SELECT DISTINCT r.RoleCode, r.RoleNameEn
                FROM dbo.APP_UserRoleScope urs
                INNER JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
                WHERE urs.UserID = ?
                ORDER BY r.RoleCode
            """, (user_id,))
            
            roles = cursor.fetchall()
            
            # Extract password from TEMP_HASH if applicable
            if password_hash.startswith('TEMP_HASH_'):
                actual_password = password_hash.replace('TEMP_HASH_', '')
                password_type = "TEMP (Plaintext)"
            else:
                actual_password = "[BCRYPT HASH - Use original password]"
                password_type = "BCRYPT"
            
            status = "✅ ACTIVE" if is_active else "❌ INACTIVE"
            
            print(f"{'─'*80}")
            print(f"👤 User ID: {user_id}")
            print(f"   Username:  {username}")
            print(f"   Password:  {actual_password}")
            print(f"   Hash Type: {password_type}")
            print(f"   Status:    {status}")
            
            if roles:
                print(f"   Roles:")
                for role in roles:
                    print(f"     • {role.RoleCode} ({role.RoleNameEn})")
            else:
                print(f"   Roles:     [None assigned]")
            print()
        
        print("="*80)
        print("\n📝 QUICK REFERENCE - LOGIN CREDENTIALS:")
        print("="*80)
        
        # Re-query for clean summary
        cursor.execute("""
            SELECT 
                u.Username,
                u.PasswordHash
            FROM dbo.APP_Users u
            WHERE u.IsActive = 1
            ORDER BY u.UserID
        """)
        
        active_users = cursor.fetchall()
        
        for user in active_users:
            username = user.Username
            password_hash = user.PasswordHash
            
            if password_hash.startswith('TEMP_HASH_'):
                password = password_hash.replace('TEMP_HASH_', '')
            else:
                password = "[See original password]"
            
            print(f"  Username: {username:25} | Password: {password}")
        
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == '__main__':
    list_all_users()
