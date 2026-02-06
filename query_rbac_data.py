"""
Query current RBAC data from database
"""
import pyodbc

def get_connection():
    """Get SQL Server database connection."""
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn

def main():
    conn = get_connection()
    cursor = conn.cursor()
    
    print("\n" + "="*80)
    print("🔐 RBAC DATABASE ANALYSIS")
    print("="*80)
    
    # 1. Users
    print("\n--- 1. APP_Users ---")
    cursor.execute("""
        SELECT 
            UserID,
            Username,
            PasswordHash,
            IsActive,
            CreatedAt
        FROM dbo.APP_Users
        ORDER BY UserID
    """)
    users = cursor.fetchall()
    print(f"\nTotal Users: {len(users)}\n")
    for user in users:
        print(f"  UserID: {user.UserID}")
        print(f"    Username: {user.Username}")
        print(f"    PasswordHash: {user.PasswordHash}")
        print(f"    IsActive: {user.IsActive}")
        print(f"    CreatedAt: {user.CreatedAt}")
        print()
    
    # 2. Roles
    print("\n--- 2. APP_Roles ---")
    cursor.execute("""
        SELECT 
            RoleID,
            RoleCode,
            RoleNameEn,
            RoleNameAr
        FROM dbo.APP_Roles
        ORDER BY RoleID
    """)
    roles = cursor.fetchall()
    print(f"\nTotal Roles: {len(roles)}\n")
    for role in roles:
        print(f"  RoleID: {role.RoleID} - {role.RoleCode}")
        print(f"    English: {role.RoleNameEn}")
        print(f"    Arabic: {role.RoleNameAr}")
        print()
    
    # 3. User Role Scopes
    print("\n--- 3. APP_UserRoleScope (User-Role-OrgUnit Mappings) ---")
    cursor.execute("""
        SELECT 
            urs.UserRoleScopeID,
            u.Username,
            r.RoleCode,
            urs.OrgUnitID,
            urs.OrgUnitType
        FROM dbo.APP_UserRoleScope urs
        INNER JOIN dbo.APP_Users u ON urs.UserID = u.UserID
        INNER JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
        ORDER BY u.Username, r.RoleCode
    """)
    scopes = cursor.fetchall()
    print(f"\nTotal Role Scopes: {len(scopes)}\n")
    for scope in scopes:
        print(f"  {scope.Username} → {scope.RoleCode} → {scope.OrgUnitType}({scope.OrgUnitID})")
    
    # 4. Check for org unit tables
    print("\n\n--- 4. Organizational Unit Tables ---")
    cursor.execute("""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = 'dbo'
        AND TABLE_NAME LIKE '%Administration%' 
        OR TABLE_NAME LIKE '%Department%' 
        OR TABLE_NAME LIKE '%Section%'
        OR TABLE_NAME LIKE '%OrgUnit%'
        ORDER BY TABLE_NAME
    """)
    org_tables = cursor.fetchall()
    if org_tables:
        print(f"\nFound {len(org_tables)} org unit related tables:")
        for table in org_tables:
            print(f"  - {table.TABLE_NAME}")
    else:
        print("\n⚠️  No organizational unit tables found!")
        print("    (No tables with names containing: Administration, Department, Section, OrgUnit)")
    
    # 5. Check what tables exist
    print("\n\n--- 5. All Tables in Database ---")
    cursor.execute("""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = 'dbo'
        AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
    """)
    all_tables = cursor.fetchall()
    print(f"\nTotal Tables: {len(all_tables)}\n")
    for table in all_tables:
        print(f"  - {table.TABLE_NAME}")
    
    # 6. Login Credentials Summary
    print("\n\n" + "="*80)
    print("📋 WORKING LOGIN CREDENTIALS")
    print("="*80)
    cursor.execute("""
        SELECT 
            Username,
            PasswordHash
        FROM dbo.APP_Users
        WHERE IsActive = 1
        ORDER BY UserID
    """)
    active_users = cursor.fetchall()
    for user in active_users:
        if user.PasswordHash.startswith('TEMP_HASH_'):
            password = user.PasswordHash.replace('TEMP_HASH_', '')
            print(f"  Username: {user.Username:<25} | Password: {password}")
        else:
            print(f"  Username: {user.Username:<25} | Password: [BCRYPT HASH]")
    
    print("\n" + "="*80)
    
    cursor.close()
    conn.close()

if __name__ == '__main__':
    main()
