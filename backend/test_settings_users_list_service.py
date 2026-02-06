"""
PHASE B — B-B6 — SERVICE TEST — SETTINGS USERS LIST ADAPTER

Test suite for list_users_for_settings_service function.
Tests that the adapter correctly flattens user inventory data.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.database import get_connection
from backend.api.db_layer.user_management_db import insert_user_record, insert_user_role_scope
from backend.api.services.user_management_service import list_users_for_settings_service


def cleanup_test_users(conn):
    """Clean up test users from previous runs."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM dbo.APP_UserRoleScope 
            WHERE UserID IN (
                SELECT UserID FROM dbo.APP_Users 
                WHERE Username LIKE 'bb6_settings_%'
            )
        """)
        cursor.execute("""
            DELETE FROM dbo.APP_Users 
            WHERE Username LIKE 'bb6_settings_%'
        """)
        conn.commit()
    finally:
        cursor.close()


def test_returns_flat_structure():
    """Test 1: Returns flattened list with correct keys."""
    print("\n" + "="*60)
    print("TEST 1: Returns Flat Structure with Correct Keys")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user
        user_id = insert_user_record(
            conn,
            username="bb6_settings_user1",
            password_hash="TEST_HASH",
            display_name="Test User One",
            department_display_name="Test Department"
        )
        conn.commit()
        
        print(f"✓ Created test user with ID: {user_id}")
        
        # Get a valid org unit ID
        cursor.execute("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Frozen = 0")
        org_row = cursor.fetchone()
        org_unit_id = org_row.UniqueID
        
        # Get a valid role ID
        cursor.execute("SELECT TOP 1 RoleID FROM dbo.APP_Roles")
        role_row = cursor.fetchone()
        role_id = role_row.RoleID
        
        print(f"✓ Using org_unit_id: {org_unit_id}, role_id: {role_id}")
        
        # Assign role and scope
        insert_user_role_scope(
            conn,
            user_id=user_id,
            role_id=role_id,
            org_unit_id=org_unit_id
        )
        conn.commit()
        
        print(f"✓ Assigned role and scope")
        
        # Call the adapter service
        print(f"\nCalling list_users_for_settings_service()...")
        results = list_users_for_settings_service()
        
        print(f"✓ Service returned {len(results)} rows")
        
        # Find our test user
        test_row = None
        for row in results:
            if row.get("username") == "bb6_settings_user1":
                test_row = row
                break
        
        # Assertions
        assert test_row is not None, "Test user not found in results"
        
        required_keys = [
            "user_id",
            "username",
            "display_name",
            "department_display_name",
            "role_name",
            "org_unit_name",
            "is_active"
        ]
        
        for key in required_keys:
            assert key in test_row, f"Missing required key: {key}"
        
        print(f"✓ Found test user row with all required keys")
        print(f"  user_id: {test_row['user_id']}")
        print(f"  username: {test_row['username']}")
        print(f"  display_name: {test_row['display_name']}")
        print(f"  department_display_name: {test_row['department_display_name']}")
        print(f"  role_name: {test_row['role_name']}")
        print(f"  org_unit_name: {test_row['org_unit_name']}")
        print(f"  is_active: {test_row['is_active']}")
        
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
        conn.rollback()  # Rollback to clean up
        cleanup_test_users(conn)
        cursor.close()
        conn.close()


def test_excludes_null_usernames():
    """Test 2: Does not return rows with null usernames."""
    print("\n" + "="*60)
    print("TEST 2: Excludes Rows with Null Usernames")
    print("="*60)
    
    try:
        # Call the adapter service
        print(f"Calling list_users_for_settings_service()...")
        results = list_users_for_settings_service()
        
        print(f"✓ Service returned {len(results)} rows")
        
        # Check no rows have null username
        null_username_rows = [row for row in results if row.get("username") is None]
        
        assert len(null_username_rows) == 0, f"Found {len(null_username_rows)} rows with null username"
        
        print(f"✓ No rows with null username found")
        
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


def test_returns_correct_user_id():
    """Test 3: Returns correct user_id for created user."""
    print("\n" + "="*60)
    print("TEST 3: Returns Correct user_id")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user
        user_id = insert_user_record(
            conn,
            username="bb6_settings_user_id_test",
            password_hash="TEST_HASH",
            display_name="UserID Test",
            department_display_name="Test Dept"
        )
        conn.commit()
        
        print(f"✓ Created test user with ID: {user_id}")
        
        # Get valid IDs
        cursor.execute("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Frozen = 0")
        org_unit_id = cursor.fetchone().UniqueID
        
        cursor.execute("SELECT TOP 1 RoleID FROM dbo.APP_Roles")
        role_id = cursor.fetchone().RoleID
        
        # Assign role and scope
        insert_user_role_scope(conn, user_id=user_id, role_id=role_id, org_unit_id=org_unit_id)
        conn.commit()
        
        # Call service
        results = list_users_for_settings_service()
        
        # Find our test user
        test_row = None
        for row in results:
            if row.get("username") == "bb6_settings_user_id_test":
                test_row = row
                break
        
        # Assertions
        assert test_row is not None, "Test user not found"
        assert test_row["user_id"] == user_id, f"Expected user_id={user_id}, got {test_row['user_id']}"
        
        print(f"✓ Returned user_id matches: {test_row['user_id']}")
        
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
        conn.rollback()
        cleanup_test_users(conn)
        cursor.close()
        conn.close()


def test_returns_display_fields():
    """Test 4: Returns display_name and department_display_name."""
    print("\n" + "="*60)
    print("TEST 4: Returns Display Fields")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user with display fields
        user_id = insert_user_record(
            conn,
            username="bb6_settings_display_test",
            password_hash="TEST_HASH",
            display_name="Display Name Test",
            department_display_name="Display Dept Test"
        )
        conn.commit()
        
        print(f"✓ Created test user with display fields")
        
        # Get valid IDs
        cursor.execute("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Frozen = 0")
        org_unit_id = cursor.fetchone().UniqueID
        
        cursor.execute("SELECT TOP 1 RoleID FROM dbo.APP_Roles")
        role_id = cursor.fetchone().RoleID
        
        # Assign role
        insert_user_role_scope(conn, user_id=user_id, role_id=role_id, org_unit_id=org_unit_id)
        conn.commit()
        
        # Call service
        results = list_users_for_settings_service()
        
        # Find our test user
        test_row = None
        for row in results:
            if row.get("username") == "bb6_settings_display_test":
                test_row = row
                break
        
        # Assertions
        assert test_row is not None, "Test user not found"
        assert test_row["display_name"] == "Display Name Test", f"Expected display_name='Display Name Test', got '{test_row['display_name']}'"
        assert test_row["department_display_name"] == "Display Dept Test", f"Expected department_display_name='Display Dept Test', got '{test_row['department_display_name']}'"
        
        print(f"✓ Display fields returned correctly:")
        print(f"  display_name: '{test_row['display_name']}'")
        print(f"  department_display_name: '{test_row['department_display_name']}'")
        
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
        conn.rollback()
        cleanup_test_users(conn)
        cursor.close()
        conn.close()


def test_null_display_fields():
    """Test 5: Handles NULL display fields correctly."""
    print("\n" + "="*60)
    print("TEST 5: Handles NULL Display Fields")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user with NULL display fields
        user_id = insert_user_record(
            conn,
            username="bb6_settings_null_display",
            password_hash="TEST_HASH",
            display_name=None,
            department_display_name=None
        )
        conn.commit()
        
        print(f"✓ Created test user with NULL display fields")
        
        # Get valid IDs
        cursor.execute("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Frozen = 0")
        org_unit_id = cursor.fetchone().UniqueID
        
        cursor.execute("SELECT TOP 1 RoleID FROM dbo.APP_Roles")
        role_id = cursor.fetchone().RoleID
        
        # Assign role
        insert_user_role_scope(conn, user_id=user_id, role_id=role_id, org_unit_id=org_unit_id)
        conn.commit()
        
        # Call service
        results = list_users_for_settings_service()
        
        # Find our test user
        test_row = None
        for row in results:
            if row.get("username") == "bb6_settings_null_display":
                test_row = row
                break
        
        # Assertions
        assert test_row is not None, "Test user not found"
        assert test_row["display_name"] is None, f"Expected display_name=None, got '{test_row['display_name']}'"
        assert test_row["department_display_name"] is None, f"Expected department_display_name=None, got '{test_row['department_display_name']}'"
        
        print(f"✓ NULL display fields handled correctly:")
        print(f"  display_name: {test_row['display_name']}")
        print(f"  department_display_name: {test_row['department_display_name']}")
        
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
        conn.rollback()
        cleanup_test_users(conn)
        cursor.close()
        conn.close()


def test_multiple_roles_multiple_rows():
    """Test 6: User with multiple roles returns multiple rows."""
    print("\n" + "="*60)
    print("TEST 6: Multiple Roles Return Multiple Rows")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user
        user_id = insert_user_record(
            conn,
            username="bb6_settings_multi_role",
            password_hash="TEST_HASH",
            display_name="Multi Role User",
            department_display_name="Test"
        )
        conn.commit()
        
        print(f"✓ Created test user with ID: {user_id}")
        
        # Get two different roles
        cursor.execute("SELECT TOP 2 RoleID FROM dbo.APP_Roles")
        roles = cursor.fetchall()
        role_id_1 = roles[0].RoleID
        role_id_2 = roles[1].RoleID
        
        # Get org unit
        cursor.execute("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Frozen = 0")
        org_unit_id = cursor.fetchone().UniqueID
        
        # Assign two different roles
        insert_user_role_scope(conn, user_id=user_id, role_id=role_id_1, org_unit_id=org_unit_id)
        insert_user_role_scope(conn, user_id=user_id, role_id=role_id_2, org_unit_id=org_unit_id)
        conn.commit()
        
        print(f"✓ Assigned two roles to user")
        
        # Call service
        results = list_users_for_settings_service()
        
        # Find all rows for our test user
        test_rows = [row for row in results if row.get("username") == "bb6_settings_multi_role"]
        
        # Assertions
        assert len(test_rows) >= 2, f"Expected at least 2 rows for multi-role user, got {len(test_rows)}"
        
        print(f"✓ Found {len(test_rows)} rows for multi-role user (as expected)")
        
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
        conn.rollback()
        cleanup_test_users(conn)
        cursor.close()
        conn.close()


def test_is_active_field():
    """Test 7: is_active field is boolean."""
    print("\n" + "="*60)
    print("TEST 7: is_active Field is Boolean")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user
        user_id = insert_user_record(
            conn,
            username="bb6_settings_active_test",
            password_hash="TEST_HASH",
            display_name="Active Test",
            department_display_name="Test"
        )
        conn.commit()
        
        # Get valid IDs
        cursor.execute("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Frozen = 0")
        org_unit_id = cursor.fetchone().UniqueID
        
        cursor.execute("SELECT TOP 1 RoleID FROM dbo.APP_Roles")
        role_id = cursor.fetchone().RoleID
        
        # Assign role
        insert_user_role_scope(conn, user_id=user_id, role_id=role_id, org_unit_id=org_unit_id)
        conn.commit()
        
        # Call service
        results = list_users_for_settings_service()
        
        # Find our test user
        test_row = None
        for row in results:
            if row.get("username") == "bb6_settings_active_test":
                test_row = row
                break
        
        # Assertions
        assert test_row is not None, "Test user not found"
        assert isinstance(test_row["is_active"], bool), f"Expected is_active to be bool, got {type(test_row['is_active'])}"
        
        print(f"✓ is_active field is boolean: {test_row['is_active']}")
        
        print("\n✓ TEST 7 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 7 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 7 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.rollback()
        cleanup_test_users(conn)
        cursor.close()
        conn.close()


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("PHASE B — B-B6 — SERVICE TEST SUITE — SETTINGS USERS LIST")
    print("="*60)
    
    tests = [
        ("Returns Flat Structure with Correct Keys", test_returns_flat_structure),
        ("Excludes Rows with Null Usernames", test_excludes_null_usernames),
        ("Returns Correct user_id", test_returns_correct_user_id),
        ("Returns Display Fields", test_returns_display_fields),
        ("Handles NULL Display Fields", test_null_display_fields),
        ("Multiple Roles Return Multiple Rows", test_multiple_roles_multiple_rows),
        ("is_active Field is Boolean", test_is_active_field),
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
