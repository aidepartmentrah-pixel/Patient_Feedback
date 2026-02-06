"""
Test Phase A Step 6: Backfill User Display Fields
==================================================

Tests the one-time SQL script that backfills DisplayName and DepartmentDisplayName
for existing users.

Test Coverage:
1. DisplayName backfill (NULL → Username)
2. DepartmentDisplayName backfill (NULL → 'Unknown')
3. Idempotency (safe to run multiple times)
4. No modification of non-NULL values
5. No modification of other columns
"""

import sys
import os
import pyodbc

# Add backend and root to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(backend_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, backend_dir)


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


def read_sql_script(script_path):
    """Read the backfill SQL script."""
    with open(script_path, 'r', encoding='utf-8') as f:
        return f.read()


def execute_backfill_script(conn, script_content):
    """
    Execute the backfill script.
    
    The script contains simple UPDATE statements that can be executed directly.
    """
    cursor = conn.cursor()
    
    # Remove comments and extract UPDATE statements
    lines = script_content.split('\n')
    sql_statements = []
    current_statement = []
    in_comment = False
    
    for line in lines:
        stripped = line.strip()
        
        # Handle multi-line comments
        if '/*' in stripped:
            in_comment = True
        if '*/' in stripped:
            in_comment = False
            continue
        if in_comment:
            continue
            
        # Skip single-line comments and empty lines
        if stripped.startswith('--') or not stripped:
            continue
            
        # Collect statement lines
        current_statement.append(line)
        
        # Execute when we hit a semicolon
        if ';' in line:
            statement = '\n'.join(current_statement)
            if statement.strip():
                try:
                    cursor.execute(statement)
                    conn.commit()
                except pyodbc.Error as e:
                    print(f"Error executing statement: {e}")
                    print(f"Statement: {statement[:200]}...")
                    raise
            current_statement = []
    
    cursor.close()


def cleanup_test_users(conn):
    """Clean up test users from previous runs."""
    cursor = conn.cursor()
    try:
        test_usernames = [
            'backfill_test_1',
            'backfill_test_2',
            'backfill_test_3',
            'backfill_test_4',
        ]
        
        for username in test_usernames:
            cursor.execute("DELETE FROM dbo.APP_Users WHERE Username = ?", (username,))
        
        conn.commit()
    finally:
        cursor.close()


def test_backfill_displayname():
    """Test 1: Verify DisplayName backfill (NULL → Username)."""
    print("\n" + "="*60)
    print("TEST 1: DisplayName Backfill (NULL → Username)")
    print("="*60)
    
    conn = get_connection()
    
    try:
        cleanup_test_users(conn)
        
        # Create test user with NULL DisplayName
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive, DisplayName, DepartmentDisplayName)
            VALUES (?, ?, ?, NULL, ?)
        """, ('backfill_test_1', 'dummy_hash', 1, 'Test Dept'))
        conn.commit()
        
        # Get user ID
        cursor.execute("SELECT UserID FROM dbo.APP_Users WHERE Username = ?", ('backfill_test_1',))
        user_id = cursor.fetchone().UserID
        cursor.close()
        
        print(f"✓ Created test user with NULL DisplayName")
        
        # Run backfill script
        script_path = os.path.join(backend_dir, 'database_migrations', 'phase_a_step6_backfill_user_display_fields.sql')
        script_content = read_sql_script(script_path)
        execute_backfill_script(conn, script_content)
        
        print(f"✓ Executed backfill script")
        
        # Verify DisplayName was updated to Username
        cursor = conn.cursor()
        cursor.execute("SELECT DisplayName FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        display_name = cursor.fetchone().DisplayName
        cursor.close()
        
        assert display_name == 'backfill_test_1', \
            f"Expected DisplayName='backfill_test_1', got '{display_name}'"
        
        print(f"✓ DisplayName backfilled correctly: '{display_name}'")
        print("\n✓ TEST 1 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 1 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 1 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_users(conn)
        conn.close()


def test_backfill_department_displayname():
    """Test 2: Verify DepartmentDisplayName backfill (NULL → 'Unknown')."""
    print("\n" + "="*60)
    print("TEST 2: DepartmentDisplayName Backfill (NULL → 'Unknown')")
    print("="*60)
    
    conn = get_connection()
    
    try:
        cleanup_test_users(conn)
        
        # Create test user with NULL DepartmentDisplayName
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive, DisplayName, DepartmentDisplayName)
            VALUES (?, ?, ?, ?, NULL)
        """, ('backfill_test_2', 'dummy_hash', 1, 'Test User'))
        conn.commit()
        
        # Get user ID
        cursor.execute("SELECT UserID FROM dbo.APP_Users WHERE Username = ?", ('backfill_test_2',))
        user_id = cursor.fetchone().UserID
        cursor.close()
        
        print(f"✓ Created test user with NULL DepartmentDisplayName")
        
        # Run backfill script
        script_path = os.path.join(backend_dir, 'database_migrations', 'phase_a_step6_backfill_user_display_fields.sql')
        script_content = read_sql_script(script_path)
        execute_backfill_script(conn, script_content)
        
        print(f"✓ Executed backfill script")
        
        # Verify DepartmentDisplayName was updated to 'Unknown'
        cursor = conn.cursor()
        cursor.execute("SELECT DepartmentDisplayName FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        dept_name = cursor.fetchone().DepartmentDisplayName
        cursor.close()
        
        assert dept_name == 'Unknown', \
            f"Expected DepartmentDisplayName='Unknown', got '{dept_name}'"
        
        print(f"✓ DepartmentDisplayName backfilled correctly: '{dept_name}'")
        print("\n✓ TEST 2 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 2 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 2 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_users(conn)
        conn.close()


def test_backfill_both_null():
    """Test 3: Verify both fields backfilled when both NULL."""
    print("\n" + "="*60)
    print("TEST 3: Both Fields NULL → Backfill Both")
    print("="*60)
    
    conn = get_connection()
    
    try:
        cleanup_test_users(conn)
        
        # Create test user with BOTH fields NULL
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive, DisplayName, DepartmentDisplayName)
            VALUES (?, ?, ?, NULL, NULL)
        """, ('backfill_test_3', 'dummy_hash', 1))
        conn.commit()
        
        # Get user ID
        cursor.execute("SELECT UserID FROM dbo.APP_Users WHERE Username = ?", ('backfill_test_3',))
        user_id = cursor.fetchone().UserID
        cursor.close()
        
        print(f"✓ Created test user with BOTH fields NULL")
        
        # Run backfill script
        script_path = os.path.join(backend_dir, 'database_migrations', 'phase_a_step6_backfill_user_display_fields.sql')
        script_content = read_sql_script(script_path)
        execute_backfill_script(conn, script_content)
        
        print(f"✓ Executed backfill script")
        
        # Verify both fields were updated
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DisplayName, DepartmentDisplayName 
            FROM dbo.APP_Users 
            WHERE UserID = ?
        """, (user_id,))
        row = cursor.fetchone()
        display_name = row.DisplayName
        dept_name = row.DepartmentDisplayName
        cursor.close()
        
        assert display_name == 'backfill_test_3', \
            f"Expected DisplayName='backfill_test_3', got '{display_name}'"
        assert dept_name == 'Unknown', \
            f"Expected DepartmentDisplayName='Unknown', got '{dept_name}'"
        
        print(f"✓ Both fields backfilled correctly:")
        print(f"  DisplayName: '{display_name}'")
        print(f"  DepartmentDisplayName: '{dept_name}'")
        print("\n✓ TEST 3 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 3 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 3 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_users(conn)
        conn.close()


def test_idempotency():
    """Test 4: Verify script is idempotent (safe to run multiple times)."""
    print("\n" + "="*60)
    print("TEST 4: Idempotency (Safe to Run Twice)")
    print("="*60)
    
    conn = get_connection()
    
    try:
        cleanup_test_users(conn)
        
        # Create test user with NULL fields
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive, DisplayName, DepartmentDisplayName)
            VALUES (?, ?, ?, NULL, NULL)
        """, ('backfill_test_4', 'dummy_hash', 1))
        conn.commit()
        
        # Get user ID
        cursor.execute("SELECT UserID FROM dbo.APP_Users WHERE Username = ?", ('backfill_test_4',))
        user_id = cursor.fetchone().UserID
        cursor.close()
        
        print(f"✓ Created test user with NULL fields")
        
        # Run backfill script FIRST TIME
        script_path = os.path.join(backend_dir, 'database_migrations', 'phase_a_step6_backfill_user_display_fields.sql')
        script_content = read_sql_script(script_path)
        execute_backfill_script(conn, script_content)
        
        print(f"✓ Executed backfill script (1st time)")
        
        # Get values after first run
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DisplayName, DepartmentDisplayName 
            FROM dbo.APP_Users 
            WHERE UserID = ?
        """, (user_id,))
        row = cursor.fetchone()
        first_display = row.DisplayName
        first_dept = row.DepartmentDisplayName
        cursor.close()
        
        print(f"  After 1st run: DisplayName='{first_display}', Dept='{first_dept}'")
        
        # Run backfill script SECOND TIME (should not change anything)
        execute_backfill_script(conn, script_content)
        
        print(f"✓ Executed backfill script (2nd time)")
        
        # Get values after second run
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DisplayName, DepartmentDisplayName 
            FROM dbo.APP_Users 
            WHERE UserID = ?
        """, (user_id,))
        row = cursor.fetchone()
        second_display = row.DisplayName
        second_dept = row.DepartmentDisplayName
        cursor.close()
        
        print(f"  After 2nd run: DisplayName='{second_display}', Dept='{second_dept}'")
        
        # Verify values unchanged
        assert first_display == second_display, \
            f"DisplayName changed on 2nd run: '{first_display}' → '{second_display}'"
        assert first_dept == second_dept, \
            f"DepartmentDisplayName changed on 2nd run: '{first_dept}' → '{second_dept}'"
        
        print(f"✓ Values unchanged after 2nd run (idempotent)")
        print("\n✓ TEST 4 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 4 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 4 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_users(conn)
        conn.close()


def test_no_modification_of_existing_values():
    """Test 5: Verify non-NULL values are NOT modified."""
    print("\n" + "="*60)
    print("TEST 5: No Modification of Existing Values")
    print("="*60)
    
    conn = get_connection()
    
    try:
        cleanup_test_users(conn)
        
        # Create test user with EXISTING (non-NULL) values
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive, DisplayName, DepartmentDisplayName)
            VALUES (?, ?, ?, ?, ?)
        """, ('backfill_test_1', 'dummy_hash', 1, 'Dr. Existing Name', 'Existing Department'))
        conn.commit()
        
        # Get user ID
        cursor.execute("SELECT UserID FROM dbo.APP_Users WHERE Username = ?", ('backfill_test_1',))
        user_id = cursor.fetchone().UserID
        cursor.close()
        
        print(f"✓ Created test user with existing values:")
        print(f"  DisplayName: 'Dr. Existing Name'")
        print(f"  DepartmentDisplayName: 'Existing Department'")
        
        # Run backfill script
        script_path = os.path.join(backend_dir, 'database_migrations', 'phase_a_step6_backfill_user_display_fields.sql')
        script_content = read_sql_script(script_path)
        execute_backfill_script(conn, script_content)
        
        print(f"✓ Executed backfill script")
        
        # Verify values UNCHANGED
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DisplayName, DepartmentDisplayName 
            FROM dbo.APP_Users 
            WHERE UserID = ?
        """, (user_id,))
        row = cursor.fetchone()
        display_name = row.DisplayName
        dept_name = row.DepartmentDisplayName
        cursor.close()
        
        assert display_name == 'Dr. Existing Name', \
            f"DisplayName was modified! Expected 'Dr. Existing Name', got '{display_name}'"
        assert dept_name == 'Existing Department', \
            f"DepartmentDisplayName was modified! Expected 'Existing Department', got '{dept_name}'"
        
        print(f"✓ Existing values preserved:")
        print(f"  DisplayName: '{display_name}'")
        print(f"  DepartmentDisplayName: '{dept_name}'")
        print("\n✓ TEST 5 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 5 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 5 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_users(conn)
        conn.close()


def test_no_modification_of_other_columns():
    """Test 6: Verify other columns (Email, IsActive, etc.) are NOT modified."""
    print("\n" + "="*60)
    print("TEST 6: No Modification of Other Columns")
    print("="*60)
    
    conn = get_connection()
    
    try:
        cleanup_test_users(conn)
        
        # Create test user with NULL display fields but other data
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive, DisplayName, DepartmentDisplayName)
            VALUES (?, ?, ?, NULL, NULL)
        """, ('backfill_test_1', 'important_hash_value', 1))
        conn.commit()
        
        # Get user ID
        cursor.execute("SELECT UserID FROM dbo.APP_Users WHERE Username = ?", ('backfill_test_1',))
        user_id = cursor.fetchone().UserID
        cursor.close()
        
        print(f"✓ Created test user with PasswordHash='important_hash_value'")
        
        # Get original values
        cursor = conn.cursor()
        cursor.execute("""
            SELECT Username, PasswordHash, IsActive
            FROM dbo.APP_Users 
            WHERE UserID = ?
        """, (user_id,))
        row = cursor.fetchone()
        orig_username = row.Username
        orig_password_hash = row.PasswordHash
        orig_active = row.IsActive
        cursor.close()
        
        # Run backfill script
        script_path = os.path.join(backend_dir, 'database_migrations', 'phase_a_step6_backfill_user_display_fields.sql')
        script_content = read_sql_script(script_path)
        execute_backfill_script(conn, script_content)
        
        print(f"✓ Executed backfill script")
        
        # Verify other columns UNCHANGED
        cursor = conn.cursor()
        cursor.execute("""
            SELECT Username, PasswordHash, IsActive
            FROM dbo.APP_Users 
            WHERE UserID = ?
        """, (user_id,))
        row = cursor.fetchone()
        new_username = row.Username
        new_password_hash = row.PasswordHash
        new_active = row.IsActive
        cursor.close()
        
        assert orig_username == new_username, \
            f"Username was modified! '{orig_username}' → '{new_username}'"
        assert orig_password_hash == new_password_hash, \
            f"PasswordHash was modified! '{orig_password_hash}' → '{new_password_hash}'"
        assert orig_active == new_active, \
            f"IsActive was modified! '{orig_active}' → '{new_active}'"
        
        print(f"✓ Other columns preserved:")
        print(f"  Username: '{new_username}'")
        print(f"  PasswordHash: '{new_password_hash}'")
        print(f"  IsActive: {new_active}")
        print("\n✓ TEST 6 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 6 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 6 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_users(conn)
        conn.close()


def main():
    """Run all tests."""
    print("="*60)
    print("PHASE A STEP 6: BACKFILL USER DISPLAY FIELDS")
    print("TEST SUITE")
    print("="*60)
    
    tests = [
        test_backfill_displayname,
        test_backfill_department_displayname,
        test_backfill_both_null,
        test_idempotency,
        test_no_modification_of_existing_values,
        test_no_modification_of_other_columns,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"Total Tests: {total}")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {total - passed}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED (100%)")
        print("✅ Backfill script is ready for production use!")
        print("\nKey Features Verified:")
        print("  ✓ DisplayName: NULL → Username")
        print("  ✓ DepartmentDisplayName: NULL → 'Unknown'")
        print("  ✓ Idempotent (safe to run multiple times)")
        print("  ✓ Does not modify existing values")
        print("  ✓ Does not modify other columns")
        print("="*60)
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    exit(main())
