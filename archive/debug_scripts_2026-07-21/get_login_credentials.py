"""
Get all login credentials for web login
"""
from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# Get all users with their password hashes
cursor.execute("""
    SELECT 
        u.UserID,
        u.Username,
        u.PasswordHash,
        u.IsActive,
        r.RoleCode,
        urs.OrgUnitType
    FROM dbo.APP_Users u
    LEFT JOIN dbo.APP_UserRoleScope urs ON u.UserID = urs.UserID
    LEFT JOIN dbo.APP_Roles r ON r.RoleID = urs.RoleID
    ORDER BY u.UserID
""")

print("\n" + "="*80)
print("AVAILABLE LOGIN CREDENTIALS FOR WEBPAGE")
print("="*80)
print("\nUsername                  | Password          | Role                      | Active")
print("-" * 80)

credentials = {}
for row in cursor.fetchall():
    if row.Username not in credentials:
        # Extract password from TEMP_HASH_ format
        if row.PasswordHash and row.PasswordHash.startswith('TEMP_HASH_'):
            password = row.PasswordHash.replace('TEMP_HASH_', '')
        else:
            password = '[bcrypt - use actual password]'
        
        credentials[row.Username] = {
            'password': password,
            'role': row.RoleCode or 'N/A',
            'active': row.IsActive
        }

# Sort and display
for username in sorted(credentials.keys()):
    info = credentials[username]
    active_status = "✅ Yes" if info['active'] else "❌ No"
    print(f"{username:<25} | {info['password']:<17} | {info['role']:<25} | {active_status}")

print("\n" + "="*80)
print("KEY TEST ACCOUNTS (Use these first):")
print("="*80)

key_accounts = [
    'software_admin',
    'worker', 
    'supervisor',
    'section',
    'sec_100_admin',
    'dept_15_admin',
    'adm_1_admin'
]

print("\nUsername              Password              Role                      ")
print("-" * 75)
for username in key_accounts:
    if username in credentials:
        info = credentials[username]
        print(f"{username:<20}  {info['password']:<20}  {info['role']:<25}")

print("\n" + "="*80)
print("HOW TO LOGIN:")
print("="*80)
print("""
1. Start the backend server:
   cd backend
   python main.py

2. Open your web browser and go to:
   http://localhost:8000

3. Login with any username and password from above

Example logins:
   - Admin access:     software_admin / admin123
   - Worker access:    worker / worker123
   - Section admin:    sec_100_admin / Hospital2026!
   - Department admin: dept_15_admin / Hospital2026!

Note: You use the PASSWORD directly, NOT the hash from database!
""")

print("="*80)
print(f"Total available accounts: {len(credentials)}")
print("="*80 + "\n")

cursor.close()
conn.close()
