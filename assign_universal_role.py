"""Assign UNIVERSAL_SECTION role to user 343."""
import sys
sys.path.insert(0, "backend")
from backend.core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# Get role ID
cursor.execute("SELECT RoleID FROM dbo.APP_Roles WHERE RoleCode = 'UNIVERSAL_SECTION'")
role_id = cursor.fetchone().RoleID
print(f'Role ID: {role_id}')

# Get admin unit
cursor.execute('SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Type = 323 ORDER BY UniqueID')
org_unit_id = cursor.fetchone().UniqueID
print(f'Org Unit ID: {org_unit_id}')

# Check if already assigned
cursor.execute('SELECT 1 FROM dbo.APP_UserRoleScope WHERE UserID = 343 AND RoleID = ?', (role_id,))
if cursor.fetchone():
    print('Role already assigned to user 343')
else:
    # Create scope for user 343
    cursor.execute('''
        INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
        VALUES (343, ?, ?, 'ADMINISTRATION')
    ''', (role_id, org_unit_id))
    conn.commit()
    print('Assigned role to user 343')

cursor.close()
conn.close()
print('Done!')
