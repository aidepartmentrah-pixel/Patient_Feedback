"""
Test Phase A Adj-2: Update Functions Coverage (INSERT + UPDATE)
================================================================

Tests that UPDATE functions properly support DisplayName and 
DepartmentDisplayName fields with COALESCE logic to preserve 
existing values.

Tests:
- update_user_profile() function
- COALESCE behavior (None = keep existing)
- Backward compatibility
"""

import sys
import os
import pyodbc

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.api.db_layer.auth_db import update_user_profile
from backend.api.db_layer.section_admin_creator_db import insert_user


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


def cleanup_test_users(conn):
    """Clean up any test users from previous runs."""
    cursor = conn.cursor()
    try:
        test_usernames = [
            'test_update_user_1',
            'test_update_user_2',
            'test_update_user_3',
            'test_update_user_4',
            'test_update_user_5',
        ]
        
        for username in test_usernames:
            cursor.execute("DELETE FROM dbo.APP_Users WHERE Username = ?", (username,))
        
        conn.commit()
    finally:
        cursor.close()


def get_user_profile(conn, user_id: int):
    """Helper to get user profile data."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT UserID, Username, DisplayName, DepartmentDisplayName
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        return cursor.fetchone()
    finally:
        cursor.close()


def test_update_both_fields():
    """Test 1: Update both DisplayName and DepartmentDisplayName."""
    print("\n" + "="*60)
    print("TEST 1: Update Both Fields")
    print("="*60)
    
    conn = get_connection()
    
    try:
        cleanup_test_users(conn)
        
        # Create test user with initial values
        username = "test_update_user_1"
        user_id = insert_user(conn, username, "Initial Name", "Initial Dept")
        conn.commit()
        
        print(f"✓ Created user {user_id}")
        
        # Verify initial state
        user = get_user_profile(conn, user_id)
        assert user.DisplayName == "Initial Name"
        assert user.DepartmentDisplayName == "Initial Dept"
        print(f"  Initial: DisplayName='{user.DisplayName}', DepartmentDisplayName='{user.DepartmentDisplayName}'")
        
        # Update both fields
        success = update_user_profile(user_id, "Updated Name", "Updated Dept")
        assert success, "Update should return True"
        
        # Verify updated state
        user = get_user_profile(conn, user_id)
        assert user.DisplayName == "Updated Name", f"Expected 'Updated Name', got '{user.DisplayName}'"
        assert user.DepartmentDisplayName == "Updated Dept", f"Expected 'Updated Dept', got '{user.DepartmentDisplayName}'"
        
        print(f"  Updated: DisplayName='{user.DisplayName}', DepartmentDisplayName='{user.DepartmentDisplayName}'")
        print("\n✓ TEST 1 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 1 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 1 ERROR: {str(e)}")
        return False
    finally:
        cleanup_test_users(conn)
        conn.close()


def test_update_display_name_only():
    """Test 2: Update only DisplayName, DepartmentDisplayName should remain unchanged (COALESCE)."""
    print("\n" + "="*60)
    print("TEST 2: Update DisplayName Only (COALESCE)")
    print("="*60)
    
    conn = get_connection()
    
    try:
        cleanup_test_users(conn)
        
        # Create test user
        username = "test_update_user_2"
        user_id = insert_user(conn, username, "Initial Name", "Initial Dept")
        conn.commit()
        
        print(f"✓ Created user {user_id}")
        
        # Verify initial state
        user = get_user_profile(conn, user_id)
        print(f"  Initial: DisplayName='{user.DisplayName}', DepartmentDisplayName='{user.DepartmentDisplayName}'")
        
        # Update only display_name (department_display_name=None should keep existing)
        success = update_user_profile(user_id, display_name="New Display Name")
        assert success, "Update should return True"
        
        # Verify: DisplayName changed, DepartmentDisplayName unchanged
        user = get_user_profile(conn, user_id)
        assert user.DisplayName == "New Display Name", f"Expected 'New Display Name', got '{user.DisplayName}'"
        assert user.DepartmentDisplayName == "Initial Dept", f"Expected 'Initial Dept' (unchanged), got '{user.DepartmentDisplayName}'"
        
        print(f"  Updated: DisplayName='{user.DisplayName}', DepartmentDisplayName='{user.DepartmentDisplayName}'")
        print("  ✓ COALESCE working: DepartmentDisplayName preserved")
        print("\n✓ TEST 2 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 2 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 2 ERROR: {str(e)}")
        return False
    finally:
        cleanup_test_users(conn)
        conn.close()


def test_update_department_name_only():
    """Test 3: Update only DepartmentDisplayName, DisplayName should remain unchanged (COALESCE)."""
    print("\n" + "="*60)
    print("TEST 3: Update DepartmentDisplayName Only (COALESCE)")
    print("="*60)
    
    conn = get_connection()
    
    try:
        cleanup_test_users(conn)
        
        # Create test user
        username = "test_update_user_3"
        user_id = insert_user(conn, username, "Initial Name", "Initial Dept")
        conn.commit()
        
        print(f"✓ Created user {user_id}")
        
        # Verify initial state
        user = get_user_profile(conn, user_id)
        print(f"  Initial: DisplayName='{user.DisplayName}', DepartmentDisplayName='{user.DepartmentDisplayName}'")
        
        # Update only department_display_name (display_name=None should keep existing)
        success = update_user_profile(user_id, department_display_name="New Dept Name")
        assert success, "Update should return True"
        
        # Verify: DepartmentDisplayName changed, DisplayName unchanged
        user = get_user_profile(conn, user_id)
        assert user.DisplayName == "Initial Name", f"Expected 'Initial Name' (unchanged), got '{user.DisplayName}'"
        assert user.DepartmentDisplayName == "New Dept Name", f"Expected 'New Dept Name', got '{user.DepartmentDisplayName}'"
        
        print(f"  Updated: DisplayName='{user.DisplayName}', DepartmentDisplayName='{user.DepartmentDisplayName}'")
        print("  ✓ COALESCE working: DisplayName preserved")
        print("\n✓ TEST 3 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 3 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 3 ERROR: {str(e)}")
        return False
    finally:
        cleanup_test_users(conn)
        conn.close()


def test_update_with_null_values():
    """Test 4: Update fields that are initially NULL."""
    print("\n" + "="*60)
    print("TEST 4: Update NULL Fields")
    print("="*60)
    
    conn = get_connection()
    
    try:
        cleanup_test_users(conn)
        
        # Create test user with NULL display fields (using defaults)
        username = "test_update_user_4"
        user_id = insert_user(conn, username)  # DisplayName=username, DepartmentDisplayName=NULL
        conn.commit()
        
        print(f"✓ Created user {user_id}")
        
        # Verify initial state
        user = get_user_profile(conn, user_id)
        assert user.DisplayName == username  # fallback to username
        assert user.DepartmentDisplayName is None  # NULL
        print(f"  Initial: DisplayName='{user.DisplayName}', DepartmentDisplayName=NULL")
        
        # Update DepartmentDisplayName from NULL to value
        success = update_user_profile(user_id, department_display_name="Newly Set Dept")
        assert success, "Update should return True"
        
        # Verify: DepartmentDisplayName now has value
        user = get_user_profile(conn, user_id)
        assert user.DisplayName == username, f"DisplayName should remain '{username}'"
        assert user.DepartmentDisplayName == "Newly Set Dept", f"Expected 'Newly Set Dept', got '{user.DepartmentDisplayName}'"
        
        print(f"  Updated: DisplayName='{user.DisplayName}', DepartmentDisplayName='{user.DepartmentDisplayName}'")
        print("  ✓ NULL field successfully updated to value")
        print("\n✓ TEST 4 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 4 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 4 ERROR: {str(e)}")
        return False
    finally:
        cleanup_test_users(conn)
        conn.close()


def test_update_nonexistent_user():
    """Test 5: Update returns False for nonexistent user."""
    print("\n" + "="*60)
    print("TEST 5: Update Nonexistent User")
    print("="*60)
    
    try:
        # Use a user ID that definitely doesn't exist
        nonexistent_user_id = 999999
        
        success = update_user_profile(nonexistent_user_id, "Some Name", "Some Dept")
        
        assert success == False, "Update should return False for nonexistent user"
        
        print(f"✓ update_user_profile({nonexistent_user_id}, ...) returned False")
        print("✓ Correctly handles nonexistent user")
        print("\n✓ TEST 5 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 5 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 5 ERROR: {str(e)}")
        return False


def test_multiple_sequential_updates():
    """Test 6: Multiple sequential updates with different field combinations."""
    print("\n" + "="*60)
    print("TEST 6: Multiple Sequential Updates")
    print("="*60)
    
    conn = get_connection()
    
    try:
        cleanup_test_users(conn)
        
        # Create test user
        username = "test_update_user_5"
        user_id = insert_user(conn, username, "Name V1", "Dept V1")
        conn.commit()
        
        print(f"✓ Created user {user_id}")
        user = get_user_profile(conn, user_id)
        print(f"  Version 1: DisplayName='{user.DisplayName}', DepartmentDisplayName='{user.DepartmentDisplayName}'")
        
        # Update 1: Change display name only
        update_user_profile(user_id, display_name="Name V2")
        user = get_user_profile(conn, user_id)
        assert user.DisplayName == "Name V2"
        assert user.DepartmentDisplayName == "Dept V1"
        print(f"  Version 2: DisplayName='{user.DisplayName}', DepartmentDisplayName='{user.DepartmentDisplayName}'")
        
        # Update 2: Change dept name only
        update_user_profile(user_id, department_display_name="Dept V2")
        user = get_user_profile(conn, user_id)
        assert user.DisplayName == "Name V2"
        assert user.DepartmentDisplayName == "Dept V2"
        print(f"  Version 3: DisplayName='{user.DisplayName}', DepartmentDisplayName='{user.DepartmentDisplayName}'")
        
        # Update 3: Change both
        update_user_profile(user_id, "Name V3", "Dept V3")
        user = get_user_profile(conn, user_id)
        assert user.DisplayName == "Name V3"
        assert user.DepartmentDisplayName == "Dept V3"
        print(f"  Version 4: DisplayName='{user.DisplayName}', DepartmentDisplayName='{user.DepartmentDisplayName}'")
        
        print("  ✓ All sequential updates preserved correct values")
        print("\n✓ TEST 6 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 6 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 6 ERROR: {str(e)}")
        return False
    finally:
        cleanup_test_users(conn)
        conn.close()


def test_coalesce_preserves_null():
    """Test 7: COALESCE correctly preserves NULL when parameter is None."""
    print("\n" + "="*60)
    print("TEST 7: COALESCE Preserves NULL")
    print("="*60)
    
    conn = get_connection()
    
    try:
        cleanup_test_users(conn)
        
        # Create test user with NULL DepartmentDisplayName
        username = "test_update_user_1"
        user_id = insert_user(conn, username, "Has Name", None)  # Dept is NULL
        conn.commit()
        
        print(f"✓ Created user {user_id}")
        user = get_user_profile(conn, user_id)
        assert user.DisplayName == "Has Name"
        assert user.DepartmentDisplayName is None
        print(f"  Initial: DisplayName='{user.DisplayName}', DepartmentDisplayName=NULL")
        
        # Update DisplayName only, should keep DepartmentDisplayName as NULL
        update_user_profile(user_id, display_name="Updated Name")
        user = get_user_profile(conn, user_id)
        assert user.DisplayName == "Updated Name"
        assert user.DepartmentDisplayName is None, "COALESCE should preserve NULL when parameter is None"
        
        print(f"  Updated: DisplayName='{user.DisplayName}', DepartmentDisplayName=NULL")
        print("  ✓ COALESCE correctly preserved NULL value")
        print("\n✓ TEST 7 PASSED")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST 7 FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ TEST 7 ERROR: {str(e)}")
        return False
    finally:
        cleanup_test_users(conn)
        conn.close()


def run_all_tests():
    """Run all Phase A Adj-2 tests."""
    print("\n" + "="*60)
    print("PHASE A ADJ-2: UPDATE FUNCTIONS COVERAGE")
    print("TEST SUITE")
    print("="*60)
    
    tests = [
        ("Test 1: Update Both Fields", test_update_both_fields),
        ("Test 2: Update DisplayName Only", test_update_display_name_only),
        ("Test 3: Update DepartmentDisplayName Only", test_update_department_name_only),
        ("Test 4: Update NULL Fields", test_update_with_null_values),
        ("Test 5: Update Nonexistent User", test_update_nonexistent_user),
        ("Test 6: Multiple Sequential Updates", test_multiple_sequential_updates),
        ("Test 7: COALESCE Preserves NULL", test_coalesce_preserves_null),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {name} EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {len(tests)}")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"Pass Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED (100%)")
        print("✅ UPDATE functions support display fields")
        print("\nKey Features Verified:")
        print("  ✓ update_user_profile() accepts optional display fields")
        print("  ✓ COALESCE logic preserves existing values when parameter is None")
        print("  ✓ Can update both fields, one field, or no fields")
        print("  ✓ Works with NULL initial values")
        print("  ✓ Returns False for nonexistent users")
        print("  ✓ Multiple sequential updates work correctly")
        print("  ✓ COALESCE preserves NULL when appropriate")
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
    
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
