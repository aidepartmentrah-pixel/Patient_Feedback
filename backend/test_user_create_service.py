"""
PHASE B — B-B4 — SERVICE TEST — CREATE USER WITH ROLE SCOPE

Test suite for create_user_with_role_scope service function.
Tests user creation with role+scope assignment at the service layer.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.database import get_connection
from backend.api.services.user_management_service import create_user_with_role_scope


def cleanup_test_users(conn):
    """Clean up test users and their scopes from previous runs."""
    cursor = conn.cursor()
    try:
        # Delete scopes first (foreign key constraint)
        cursor.execute("""
            DELETE FROM dbo.APP_UserRoleScope 
            WHERE UserID IN (
                SELECT UserID FROM dbo.APP_Users 
                WHERE Username LIKE 'bb4_service_user%'
            )
        """)
        
        # Then delete users
        cursor.execute("""
            DELETE FROM dbo.APP_Users 
            WHERE Username LIKE 'bb4_service_user%'
        """)
        conn.commit()
    finally:
        cursor.close()


def get_test_role_id(conn):
    """Get a valid role ID for testing."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT TOP 1 RoleID, RoleCode, RoleNameEn
            FROM dbo.APP_Roles
            ORDER BY RoleID
        """)
        
        row = cursor.fetchone()
        if not row:
            raise Exception("No roles found in APP_Roles table")
        
        return row.RoleID, row.RoleCode, row.RoleNameEn
    finally:
        cursor.close()


def get_test_org_unit_id(conn):
    """Get a valid org unit ID for testing."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT TOP 1 UniqueID, Name
            FROM dbo.AdminsrationUnit
            WHERE Type IN (323, 324, 325)
            ORDER BY UniqueID
        """)
        
        row = cursor.fetchone()
        if not row:
            raise Exception("No org units found in AdminsrationUnit table")
        
        return row.UniqueID, row.Name
    finally:
        cursor.close()


def test_create_user_with_all_fields():
    """Test 1: Create user with all fields provided."""
    print("\n" + "="*60)
    print("TEST 1: Create User With All Fields")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Get test data
        role_id, role_code, role_name = get_test_role_id(conn)
        org_unit_id, org_unit_name = get_test_org_unit_id(conn)
        
        print(f"✓ Using RoleID: {role_id} ({role_code} - {role_name})")
        print(f"✓ Using OrgUnitID: {org_unit_id} ({org_unit_name})")
        
        # Create user
        username = "bb4_service_user"
        password_plain = "Test123!"
        display_name = "Service User"
        department_display_name = "Service Dept"
        
        print(f"\nCreating user: {username}")
        
        user_id = create_user_with_role_scope(
            username=username,
            password_plain=password_plain,
            display_name=display_name,
            department_display_name=department_display_name,
            role_id=role_id,
            org_unit_id=org_unit_id
        )
        
        print(f"✓ User created with ID: {user_id}")
        
        # Verify user exists in APP_Users
        cursor.execute("""
            SELECT UserID, Username, DisplayName, DepartmentDisplayName, IsActive
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        user_row = cursor.fetchone()
        
        assert user_row is not None, "User not found in APP_Users"
        assert user_row.Username == username, f"Expected username '{username}', got '{user_row.Username}'"
        assert user_row.DisplayName == display_name, f"Expected DisplayName '{display_name}', got '{user_row.DisplayName}'"
        assert user_row.DepartmentDisplayName == department_display_name, f"Expected DepartmentDisplayName '{department_display_name}', got '{user_row.DepartmentDisplayName}'"
        assert user_row.IsActive == 1, f"Expected IsActive=1, got {user_row.IsActive}"
        
        print(f"✓ User record verified:")
        print(f"  Username: {user_row.Username}")
        print(f"  DisplayName: {user_row.DisplayName}")
        print(f"  DepartmentDisplayName: {user_row.DepartmentDisplayName}")
        
        # Verify role scope assignment
        cursor.execute("""
            SELECT COUNT(*) AS cnt
            FROM dbo.APP_UserRoleScope
            WHERE UserID = ?
              AND RoleID = ?
              AND OrgUnitID = ?
        """, (user_id, role_id, org_unit_id))
        
        scope_row = cursor.fetchone()
        
        assert scope_row.cnt == 1, f"Expected 1 role scope assignment, found {scope_row.cnt}"
        
        print(f"✓ Role scope assignment verified")
        
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
        cursor.close()
        conn.close()


def test_create_user_with_null_display_fields():
    """Test 2: Create user with NULL display fields."""
    print("\n" + "="*60)
    print("TEST 2: Create User With NULL Display Fields")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Get test data
        role_id, _, _ = get_test_role_id(conn)
        org_unit_id, _ = get_test_org_unit_id(conn)
        
        # Create user with NULL display fields
        username = "bb4_service_user_null"
        password_plain = "Test123!"
        
        print(f"Creating user with NULL display fields: {username}")
        
        user_id = create_user_with_role_scope(
            username=username,
            password_plain=password_plain,
            display_name=None,
            department_display_name=None,
            role_id=role_id,
            org_unit_id=org_unit_id
        )
        
        print(f"✓ User created with ID: {user_id}")
        
        # Verify NULL display fields
        cursor.execute("""
            SELECT DisplayName, DepartmentDisplayName
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        
        assert row.DisplayName is None, f"Expected DisplayName=NULL, got '{row.DisplayName}'"
        assert row.DepartmentDisplayName is None, f"Expected DepartmentDisplayName=NULL, got '{row.DepartmentDisplayName}'"
        
        print(f"✓ DisplayName: NULL (as expected)")
        print(f"✓ DepartmentDisplayName: NULL (as expected)")
        
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
        cursor.close()
        conn.close()


def test_duplicate_username_raises_error():
    """Test 3: Duplicate username raises ValueError."""
    print("\n" + "="*60)
    print("TEST 3: Duplicate Username Raises ValueError")
    print("="*60)
    
    conn = get_connection()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Get test data
        role_id, _, _ = get_test_role_id(conn)
        org_unit_id, _ = get_test_org_unit_id(conn)
        
        username = "bb4_service_user_dup"
        
        # First creation (should succeed)
        print(f"Creating user first time: {username}")
        user_id1 = create_user_with_role_scope(
            username=username,
            password_plain="Test123!",
            display_name="First User",
            department_display_name="First Dept",
            role_id=role_id,
            org_unit_id=org_unit_id
        )
        
        print(f"✓ First user created with ID: {user_id1}")
        
        # Second creation with same username (should fail)
        print(f"\nAttempting to create duplicate username: {username}")
        
        user_id2 = create_user_with_role_scope(
            username=username,
            password_plain="Test456!",
            display_name="Second User",
            department_display_name="Second Dept",
            role_id=role_id,
            org_unit_id=org_unit_id
        )
        
        # Should not reach here
        print(f"\n✗ TEST 3 FAILED: Duplicate username was accepted (UserID: {user_id2})")
        return False
        
    except ValueError as e:
        # Expected behavior
        print(f"✓ ValueError raised as expected: {str(e)}")
        print("\n✓ TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 3 ERROR: Unexpected exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_users(conn)
        conn.close()


def test_empty_username_raises_error():
    """Test 4: Empty username raises ValueError."""
    print("\n" + "="*60)
    print("TEST 4: Empty Username Raises ValueError")
    print("="*60)
    
    conn = get_connection()
    
    try:
        # Get test data
        role_id, _, _ = get_test_role_id(conn)
        org_unit_id, _ = get_test_org_unit_id(conn)
        
        # Try to create user with empty username
        print("Attempting to create user with empty username...")
        
        user_id = create_user_with_role_scope(
            username="   ",  # Only spaces
            password_plain="Test123!",
            display_name="Test",
            department_display_name="Test",
            role_id=role_id,
            org_unit_id=org_unit_id
        )
        
        # Should not reach here
        print(f"\n✗ TEST 4 FAILED: Empty username was accepted (UserID: {user_id})")
        return False
        
    except ValueError as e:
        # Expected behavior
        print(f"✓ ValueError raised as expected: {str(e)}")
        print("\n✓ TEST 4 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 4 ERROR: Unexpected exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()


def test_empty_password_raises_error():
    """Test 5: Empty password raises ValueError."""
    print("\n" + "="*60)
    print("TEST 5: Empty Password Raises ValueError")
    print("="*60)
    
    conn = get_connection()
    
    try:
        # Get test data
        role_id, _, _ = get_test_role_id(conn)
        org_unit_id, _ = get_test_org_unit_id(conn)
        
        # Try to create user with empty password
        print("Attempting to create user with empty password...")
        
        user_id = create_user_with_role_scope(
            username="bb4_empty_pwd",
            password_plain="   ",  # Only spaces
            display_name="Test",
            department_display_name="Test",
            role_id=role_id,
            org_unit_id=org_unit_id
        )
        
        # Should not reach here
        print(f"\n✗ TEST 5 FAILED: Empty password was accepted (UserID: {user_id})")
        return False
        
    except ValueError as e:
        # Expected behavior
        print(f"✓ ValueError raised as expected: {str(e)}")
        print("\n✓ TEST 5 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 5 ERROR: Unexpected exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()


def test_invalid_role_id_raises_error():
    """Test 6: Invalid role_id raises ValueError."""
    print("\n" + "="*60)
    print("TEST 6: Invalid role_id Raises ValueError")
    print("="*60)
    
    conn = get_connection()
    
    try:
        # Get test data
        org_unit_id, _ = get_test_org_unit_id(conn)
        
        # Try with invalid role_id
        print("Attempting to create user with role_id=0...")
        
        user_id = create_user_with_role_scope(
            username="bb4_invalid_role",
            password_plain="Test123!",
            display_name="Test",
            department_display_name="Test",
            role_id=0,
            org_unit_id=org_unit_id
        )
        
        # Should not reach here
        print(f"\n✗ TEST 6 FAILED: Invalid role_id was accepted (UserID: {user_id})")
        return False
        
    except ValueError as e:
        # Expected behavior
        print(f"✓ ValueError raised as expected: {str(e)}")
        print("\n✓ TEST 6 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 6 ERROR: Unexpected exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()


def test_invalid_org_unit_id_raises_error():
    """Test 7: Invalid org_unit_id raises ValueError."""
    print("\n" + "="*60)
    print("TEST 7: Invalid org_unit_id Raises ValueError")
    print("="*60)
    
    conn = get_connection()
    
    try:
        # Get test data
        role_id, _, _ = get_test_role_id(conn)
        
        # Try with invalid org_unit_id
        print("Attempting to create user with org_unit_id=-1...")
        
        user_id = create_user_with_role_scope(
            username="bb4_invalid_org",
            password_plain="Test123!",
            display_name="Test",
            department_display_name="Test",
            role_id=role_id,
            org_unit_id=-1
        )
        
        # Should not reach here
        print(f"\n✗ TEST 7 FAILED: Invalid org_unit_id was accepted (UserID: {user_id})")
        return False
        
    except ValueError as e:
        # Expected behavior
        print(f"✓ ValueError raised as expected: {str(e)}")
        print("\n✓ TEST 7 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 7 ERROR: Unexpected exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()


def test_password_is_hashed():
    """Test 8: Password is properly hashed (not stored as plaintext)."""
    print("\n" + "="*60)
    print("TEST 8: Password Is Properly Hashed")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Get test data
        role_id, _, _ = get_test_role_id(conn)
        org_unit_id, _ = get_test_org_unit_id(conn)
        
        # Create user
        username = "bb4_password_hash"
        password_plain = "MySecretPassword123!"
        
        print(f"Creating user with password: '{password_plain}'")
        
        user_id = create_user_with_role_scope(
            username=username,
            password_plain=password_plain,
            display_name="Hash Test",
            department_display_name="Hash Dept",
            role_id=role_id,
            org_unit_id=org_unit_id
        )
        
        print(f"✓ User created with ID: {user_id}")
        
        # Verify password is hashed (not plaintext)
        cursor.execute("""
            SELECT PasswordHash
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        stored_hash = row.PasswordHash
        
        # Assertions
        assert stored_hash != password_plain, "Password should be hashed, not stored as plaintext"
        assert stored_hash.startswith('$2b$'), f"Password should be bcrypt hash (start with $2b$), got: {stored_hash[:10]}"
        assert len(stored_hash) >= 60, f"Bcrypt hash should be at least 60 chars, got {len(stored_hash)}"
        
        print(f"✓ Password properly hashed (bcrypt)")
        print(f"  Hash starts with: {stored_hash[:10]}...")
        print(f"  Hash length: {len(stored_hash)} chars")
        
        print("\n✓ TEST 8 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 8 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 8 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_users(conn)
        cursor.close()
        conn.close()


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("PHASE B — B-B4 — SERVICE TEST SUITE — CREATE USER")
    print("="*60)
    
    tests = [
        ("Create User With All Fields", test_create_user_with_all_fields),
        ("Create User With NULL Display Fields", test_create_user_with_null_display_fields),
        ("Duplicate Username Raises ValueError", test_duplicate_username_raises_error),
        ("Empty Username Raises ValueError", test_empty_username_raises_error),
        ("Empty Password Raises ValueError", test_empty_password_raises_error),
        ("Invalid role_id Raises ValueError", test_invalid_role_id_raises_error),
        ("Invalid org_unit_id Raises ValueError", test_invalid_org_unit_id_raises_error),
        ("Password Is Properly Hashed", test_password_is_hashed),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
