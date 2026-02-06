"""
Test Script for Phase 2: RBAC Tables Creation
===============================================
This script:
1. Executes the SQL script to create RBAC tables
2. Verifies all tables were created
3. Verifies all seed data was inserted
4. Displays comprehensive test results

Run from backend directory:
    python test_phase2_rbac_tables.py
"""

import sys
import os
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from core.database import get_connection


def execute_sql_file(conn, sql_file_path):
    """Execute SQL script file and capture all messages."""
    print(f"\n{'='*70}")
    print(f"Executing SQL script: {sql_file_path}")
    print(f"{'='*70}\n")
    
    # Read SQL file
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        sql_script = f.read()
    
    cursor = conn.cursor()
    
    try:
        # Execute the entire script
        cursor.execute(sql_script)
        
        # Fetch any result sets (from SELECT statements)
        while cursor.description:
            rows = cursor.fetchall()
            if rows:
                # Print column headers
                columns = [desc[0] for desc in cursor.description]
                print(" | ".join(columns))
                print("-" * 70)
                
                # Print rows
                for row in rows:
                    print(" | ".join(str(val) for val in row))
                print()
            
            # Move to next result set
            if not cursor.nextset():
                break
        
        # Get messages from SQL Server (PRINT statements)
        messages = []
        while cursor.nextset():
            pass
        
        conn.commit()
        print("\n✓ SQL script executed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error executing SQL script: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()


def verify_tables(conn):
    """Verify that all RBAC tables exist."""
    print(f"\n{'='*70}")
    print("Verification 1: Checking Table Existence")
    print(f"{'='*70}\n")
    
    cursor = conn.cursor()
    expected_tables = ['APP_Users', 'APP_Roles', 'APP_UserRoleScope']
    
    all_exist = True
    for table_name in expected_tables:
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = ? AND TABLE_SCHEMA = 'dbo'
        """, table_name)
        
        exists = cursor.fetchone()[0] > 0
        status = "✓" if exists else "✗"
        print(f"{status} Table: {table_name} {'EXISTS' if exists else 'MISSING'}")
        
        if not exists:
            all_exist = False
    
    cursor.close()
    
    print(f"\n{'→ All tables exist!' if all_exist else '→ Some tables are missing!'}\n")
    return all_exist


def verify_roles(conn):
    """Verify that all roles were inserted."""
    print(f"\n{'='*70}")
    print("Verification 2: Checking Roles")
    print(f"{'='*70}\n")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT RoleID, RoleCode, RoleNameEn, RoleNameAr
        FROM dbo.APP_Roles
        ORDER BY RoleID
    """)
    
    roles = cursor.fetchall()
    expected_role_codes = [
        'SOFTWARE_ADMIN',
        'WORKER',
        'COMPLAINT_SUPERVISOR',
        'SECTION_ADMIN',
        'DEPARTMENT_ADMIN',
        'ADMINISTRATION_ADMIN'
    ]
    
    print(f"Found {len(roles)} roles:\n")
    print(f"{'ID':<5} {'Role Code':<30} {'Name (EN)':<30} {'Name (AR)':<30}")
    print("-" * 100)
    
    actual_role_codes = []
    for role in roles:
        role_id, role_code, name_en, name_ar = role
        print(f"{role_id:<5} {role_code:<30} {name_en:<30} {name_ar:<30}")
        actual_role_codes.append(role_code)
    
    cursor.close()
    
    # Check if all expected roles exist
    all_exist = all(code in actual_role_codes for code in expected_role_codes)
    print(f"\n{'✓ All 6 roles exist!' if all_exist else '✗ Some roles are missing!'}\n")
    return all_exist


def verify_users(conn):
    """Verify that all test users were created."""
    print(f"\n{'='*70}")
    print("Verification 3: Checking Users")
    print(f"{'='*70}\n")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT UserID, Username, IsActive, CreatedAt
        FROM dbo.APP_Users
        ORDER BY UserID
    """)
    
    users = cursor.fetchall()
    expected_usernames = [
        'software_admin',
        'worker',
        'complaint_supervisor',
        'section_admin',
        'department_admin',
        'administration_admin'
    ]
    
    print(f"Found {len(users)} users:\n")
    print(f"{'ID':<5} {'Username':<30} {'Active':<10} {'Created At':<30}")
    print("-" * 80)
    
    actual_usernames = []
    for user in users:
        user_id, username, is_active, created_at = user
        status = "Yes" if is_active else "No"
        print(f"{user_id:<5} {username:<30} {status:<10} {str(created_at):<30}")
        actual_usernames.append(username)
    
    cursor.close()
    
    # Check if all expected users exist
    all_exist = all(name in actual_usernames for name in expected_usernames)
    print(f"\n{'✓ All 6 test users exist!' if all_exist else '✗ Some users are missing!'}\n")
    return all_exist


def verify_role_scopes(conn):
    """Verify that all user-role-scope mappings were created."""
    print(f"\n{'='*70}")
    print("Verification 4: Checking User Role Scopes")
    print(f"{'='*70}\n")
    
    cursor = conn.cursor()
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
    
    print(f"Found {len(scopes)} role scope assignments:\n")
    print(f"{'ID':<5} {'Username':<30} {'Role':<30} {'OrgUnit':<10} {'Type':<20}")
    print("-" * 100)
    
    expected_mappings = [
        ('software_admin', 'SOFTWARE_ADMIN', 0, 'ADMINISTRATION'),
        ('worker', 'WORKER', 10, 'COMPLAINT'),
        ('complaint_supervisor', 'COMPLAINT_SUPERVISOR', 10, 'COMPLAINT'),
        ('section_admin', 'SECTION_ADMIN', 10, 'SECTION'),
        ('department_admin', 'DEPARTMENT_ADMIN', 5, 'DEPARTMENT'),
        ('administration_admin', 'ADMINISTRATION_ADMIN', 1, 'ADMINISTRATION'),
    ]
    
    actual_mappings = []
    for scope in scopes:
        scope_id, username, role_code, org_unit_id, org_unit_type = scope
        print(f"{scope_id:<5} {username:<30} {role_code:<30} {org_unit_id:<10} {org_unit_type:<20}")
        actual_mappings.append((username, role_code, org_unit_id, org_unit_type))
    
    cursor.close()
    
    # Check if all expected mappings exist
    all_exist = all(mapping in actual_mappings for mapping in expected_mappings)
    print(f"\n{'✓ All 6 role scope mappings exist!' if all_exist else '✗ Some mappings are missing!'}\n")
    return all_exist


def verify_constraints(conn):
    """Verify that all foreign keys and constraints were created."""
    print(f"\n{'='*70}")
    print("Verification 5: Checking Foreign Keys and Constraints")
    print(f"{'='*70}\n")
    
    cursor = conn.cursor()
    
    # Check foreign keys
    cursor.execute("""
        SELECT 
            fk.name AS ForeignKeyName,
            OBJECT_NAME(fk.parent_object_id) AS TableName,
            COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS ColumnName,
            OBJECT_NAME(fk.referenced_object_id) AS ReferencedTable,
            COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS ReferencedColumn
        FROM sys.foreign_keys AS fk
        INNER JOIN sys.foreign_key_columns AS fkc 
            ON fk.object_id = fkc.constraint_object_id
        WHERE OBJECT_NAME(fk.parent_object_id) IN ('APP_UserRoleScope')
        ORDER BY fk.name
    """)
    
    fks = cursor.fetchall()
    print(f"Foreign Keys ({len(fks)}):\n")
    for fk in fks:
        fk_name, table, column, ref_table, ref_column = fk
        print(f"  ✓ {fk_name}: {table}.{column} → {ref_table}.{ref_column}")
    
    # Check unique constraints
    cursor.execute("""
        SELECT 
            tc.CONSTRAINT_NAME,
            tc.TABLE_NAME,
            STRING_AGG(kcu.COLUMN_NAME, ', ') AS Columns
        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
            ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
            AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA
        WHERE tc.TABLE_NAME IN ('APP_Users', 'APP_Roles', 'APP_UserRoleScope')
            AND tc.CONSTRAINT_TYPE = 'UNIQUE'
        GROUP BY tc.CONSTRAINT_NAME, tc.TABLE_NAME
        ORDER BY tc.TABLE_NAME
    """)
    
    unique_constraints = cursor.fetchall()
    print(f"\nUnique Constraints ({len(unique_constraints)}):\n")
    for uc in unique_constraints:
        constraint_name, table_name, columns = uc
        print(f"  ✓ {table_name}.{constraint_name}: [{columns}]")
    
    cursor.close()
    
    expected_fk_count = 2  # FK_UserRoleScope_User, FK_UserRoleScope_Role
    expected_unique_count = 3  # Username, RoleCode, UQ_UserRoleScope
    
    fk_ok = len(fks) >= expected_fk_count
    unique_ok = len(unique_constraints) >= expected_unique_count
    
    print(f"\n{'✓ All constraints created!' if (fk_ok and unique_ok) else '✗ Some constraints missing!'}\n")
    return fk_ok and unique_ok


def run_comprehensive_test():
    """Run all tests and display results."""
    print("\n" + "="*70)
    print(" " * 15 + "PHASE 2: RBAC TABLES TEST")
    print("="*70)
    
    conn = None
    all_passed = True
    
    try:
        # Connect to database
        print("\n→ Connecting to database...")
        conn = get_connection()
        print("✓ Connected successfully!")
        
        # Execute SQL script
        sql_file_path = backend_dir / "database_migrations" / "phase2_create_rbac_tables.sql"
        if not execute_sql_file(conn, sql_file_path):
            all_passed = False
        
        # Run verifications
        verifications = [
            ("Tables", verify_tables),
            ("Roles", verify_roles),
            ("Users", verify_users),
            ("Role Scopes", verify_role_scopes),
            ("Constraints", verify_constraints),
        ]
        
        results = {}
        for name, verify_func in verifications:
            try:
                results[name] = verify_func(conn)
                if not results[name]:
                    all_passed = False
            except Exception as e:
                print(f"\n✗ Error during {name} verification: {e}")
                results[name] = False
                all_passed = False
        
        # Final Summary
        print("\n" + "="*70)
        print(" " * 20 + "TEST SUMMARY")
        print("="*70 + "\n")
        
        for name, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{name:<30} {status}")
        
        print("\n" + "-"*70)
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! RBAC tables created successfully!\n")
        else:
            print("\n⚠️  SOME TESTS FAILED! Please review the output above.\n")
        print("="*70 + "\n")
        
        return all_passed
        
    except Exception as e:
        print(f"\n✗ Critical error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if conn:
            conn.close()
            print("→ Database connection closed.\n")


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
