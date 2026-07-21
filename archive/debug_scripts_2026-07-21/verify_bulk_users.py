"""
Verify MODULE 5.2 Bulk User Creation
"""
from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# Count bulk admin users
cursor.execute("""
    SELECT COUNT(*) as count 
    FROM dbo.APP_Users 
    WHERE Username LIKE 'adm_%_admin' 
       OR Username LIKE 'dept_%_admin' 
       OR Username LIKE 'sec_%_admin'
""")
result = cursor.fetchone()
print(f"\n{'='*70}")
print(f"MODULE 5.2 — Bulk User Generator Verification")
print(f"{'='*70}")
print(f"Total bulk admin users created: {result.count}")

# Get sample users by type
print(f"\n{'='*70}")
print("Sample Users by Type:")
print(f"{'='*70}")

# Administration admins
cursor.execute("SELECT TOP 5 Username FROM APP_Users WHERE Username LIKE 'adm_%_admin' ORDER BY Username")
adm_users = cursor.fetchall()
print(f"\nAdministration Admins: {len(adm_users)}")
for row in adm_users:
    print(f"  - {row.Username}")

# Department admins
cursor.execute("SELECT TOP 5 Username FROM APP_Users WHERE Username LIKE 'dept_%_admin' ORDER BY Username")
dept_users = cursor.fetchall()
print(f"\nDepartment Admins: {len(dept_users)}")
for row in dept_users:
    print(f"  - {row.Username}")

# Section admins
cursor.execute("SELECT TOP 5 Username FROM APP_Users WHERE Username LIKE 'sec_%_admin' ORDER BY Username")
sec_users = cursor.fetchall()
print(f"\nSection Admins: {len(sec_users)}")
for row in sec_users:
    print(f"  - {row.Username}")

# Verify role assignments
cursor.execute("""
    SELECT COUNT(*) as count
    FROM dbo.APP_UserRoleScope urs
    INNER JOIN dbo.APP_Users u ON urs.UserID = u.UserID
    WHERE u.Username LIKE 'adm_%_admin' 
       OR u.Username LIKE 'dept_%_admin' 
       OR u.Username LIKE 'sec_%_admin'
""")
result = cursor.fetchone()
print(f"\n{'='*70}")
print(f"Total role scope assignments: {result.count}")
print(f"{'='*70}")

# Check organizational unit coverage
cursor.execute("""
    SELECT 
        au.Type,
        CASE 
            WHEN au.Type = 323 THEN 'ADMINISTRATION'
            WHEN au.Type = 325 THEN 'DEPARTMENT'
            WHEN au.Type = 324 THEN 'SECTION'
        END AS TypeName,
        COUNT(*) as TotalUnits,
        COUNT(urs.UserRoleScopeID) as UnitsWithAdmin
    FROM dbo.AdminsrationUnit au
    LEFT JOIN dbo.APP_UserRoleScope urs ON urs.OrgUnitID = au.UniqueID
    LEFT JOIN dbo.APP_Users u ON u.UserID = urs.UserID AND (
        u.Username LIKE 'adm_%_admin' OR 
        u.Username LIKE 'dept_%_admin' OR 
        u.Username LIKE 'sec_%_admin'
    )
    WHERE au.Type IN (323, 324, 325)
      AND au.Frozen = 0
    GROUP BY au.Type
    ORDER BY au.Type
""")

print(f"\n{'='*70}")
print("Organizational Unit Coverage:")
print(f"{'='*70}")
print(f"{'Type':<20} {'Total Units':<15} {'With Admin':<15} {'Coverage'}")
print(f"{'-'*70}")

for row in cursor.fetchall():
    coverage = (row.UnitsWithAdmin / row.TotalUnits * 100) if row.TotalUnits > 0 else 0
    print(f"{row.TypeName:<20} {row.TotalUnits:<15} {row.UnitsWithAdmin:<15} {coverage:.1f}%")

print(f"\n{'='*70}")
print("✅ MODULE 5.2 Verification Complete")
print(f"{'='*70}\n")

cursor.close()
conn.close()
