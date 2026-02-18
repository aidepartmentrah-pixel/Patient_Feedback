"""
Quick script to find valid admin credentials for testing
"""
import sys
sys.path.insert(0, 'backend')

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

print("="*60)
print("Finding admin users for testing:")
print("="*60)

cursor.execute("""
    SELECT TOP 5 
        u.UserID,
        u.Username,
        r.RoleCode,
        u.IsActive
    FROM dbo.APP_Users u
    JOIN dbo.APP_UserRoleScope scope ON u.UserID = scope.UserID
    JOIN dbo.APP_Roles r ON scope.RoleID = r.RoleID
    WHERE r.RoleCode IN ('SECTION_ADMIN', 'DEPARTMENT_ADMIN', 'ADMINISTRATION_ADMIN', 'SOFTWARE_ADMIN')
      AND u.IsActive = 1
    ORDER BY u.UserID
""")

print("\nAvailable admin users:")
print("-" * 60)
for row in cursor.fetchall():
    print(f"Username: {row.Username:20} Role: {row.RoleCode:25} Active: {row.IsActive}")

cursor.close()
conn.close()

print("\nNote: You'll need to know the password for one of these users")
print("Common test passwords: 'password', '123456', 'admin123', etc.")
