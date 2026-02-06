"""
PHASE B — B-B3 — DB TEST — UPDATE USER IDENTITY

Test suite for update_user_identity_fields function.
Tests updating user display identity fields at the database layer.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.database import get_connection
from backend.api.db_layer.user_management_db import (
    insert_user_record,
    update_user_identity_fields
)


def cleanup_test_users(conn):
    """Clean up test users from previous runs."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM dbo.APP_Users 
            WHERE Username LIKE 'bb3_test_user%'
        """)
    finally:
        cursor.close()


def test_update_both_fields():
    """Test 1: Update both display fields."""
    print("\n" + "="*60)
    print("TEST 1: Update Both Display Fields")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user with initial values
        user_id = insert_user_record(
            conn,
            username="bb3_test_user",
            password_hash="TEST_HASH",
            display_name="Initial Name",
            department_display_name="Initial Dept"
        )
        
        print(f"✓ Created test user with ID: {user_id}")
        print(f"  Initial DisplayName: 'Initial Name'")
        print(f"  Initial DepartmentDisplayName: 'Initial Dept'")
        
        # Update both fields
        print(f"\nUpdating both fields...")
        update_user_identity_fields(
            conn,
            user_id=user_id,
            display_name="Updated Name",
            department_display_name="Updated Dept"
        )
        
        print(f"✓ Update completed")
        
        # Verify changes
        cursor.execute("""
            SELECT DisplayName, DepartmentDisplayName
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        
        # Assertions
        assert row is not None, "User not found after update"
        assert row.DisplayName == "Updated Name", f"Expected DisplayName='Updated Name', got '{row.DisplayName}'"
        assert row.DepartmentDisplayName == "Updated Dept", f"Expected DepartmentDisplayName='Updated Dept', got '{row.DepartmentDisplayName}'"
        
        print(f"✓ DisplayName: '{row.DisplayName}'")
        print(f"✓ DepartmentDisplayName: '{row.DepartmentDisplayName}'")
        
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
        # Rollback to clean up
        conn.rollback()
        cursor.close()
        conn.close()


def test_partial_update_display_name_only():
    """Test 2: Update display_name only (department unchanged)."""
    print("\n" + "="*60)
    print("TEST 2: Partial Update - DisplayName Only")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user
        user_id = insert_user_record(
            conn,
            username="bb3_test_user_partial1",
            password_hash="TEST_HASH",
            display_name="Original Name",
            department_display_name="Original Dept"
        )
        
        print(f"✓ Created test user with ID: {user_id}")
        print(f"  Initial DisplayName: 'Original Name'")
        print(f"  Initial DepartmentDisplayName: 'Original Dept'")
        
        # Update display_name only (department_display_name = None)
        print(f"\nUpdating DisplayName only (department=None)...")
        update_user_identity_fields(
            conn,
            user_id=user_id,
            display_name="New Name",
            department_display_name=None
        )
        
        print(f"✓ Update completed")
        
        # Verify changes
        cursor.execute("""
            SELECT DisplayName, DepartmentDisplayName
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        
        # Assertions
        assert row is not None, "User not found after update"
        assert row.DisplayName == "New Name", f"Expected DisplayName='New Name', got '{row.DisplayName}'"
        assert row.DepartmentDisplayName == "Original Dept", f"Expected DepartmentDisplayName unchanged='Original Dept', got '{row.DepartmentDisplayName}'"
        
        print(f"✓ DisplayName updated: '{row.DisplayName}'")
        print(f"✓ DepartmentDisplayName unchanged: '{row.DepartmentDisplayName}'")
        
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
        # Rollback to clean up
        conn.rollback()
        cursor.close()
        conn.close()


def test_partial_update_department_only():
    """Test 3: Update department_display_name only (name unchanged)."""
    print("\n" + "="*60)
    print("TEST 3: Partial Update - DepartmentDisplayName Only")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user
        user_id = insert_user_record(
            conn,
            username="bb3_test_user_partial2",
            password_hash="TEST_HASH",
            display_name="Stable Name",
            department_display_name="Original Dept"
        )
        
        print(f"✓ Created test user with ID: {user_id}")
        print(f"  Initial DisplayName: 'Stable Name'")
        print(f"  Initial DepartmentDisplayName: 'Original Dept'")
        
        # Update department_display_name only (display_name = None)
        print(f"\nUpdating DepartmentDisplayName only (name=None)...")
        update_user_identity_fields(
            conn,
            user_id=user_id,
            display_name=None,
            department_display_name="New Dept"
        )
        
        print(f"✓ Update completed")
        
        # Verify changes
        cursor.execute("""
            SELECT DisplayName, DepartmentDisplayName
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        
        # Assertions
        assert row is not None, "User not found after update"
        assert row.DisplayName == "Stable Name", f"Expected DisplayName unchanged='Stable Name', got '{row.DisplayName}'"
        assert row.DepartmentDisplayName == "New Dept", f"Expected DepartmentDisplayName='New Dept', got '{row.DepartmentDisplayName}'"
        
        print(f"✓ DisplayName unchanged: '{row.DisplayName}'")
        print(f"✓ DepartmentDisplayName updated: '{row.DepartmentDisplayName}'")
        
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
        # Rollback to clean up
        conn.rollback()
        cursor.close()
        conn.close()


def test_multiple_updates():
    """Test 4: Multiple sequential updates work correctly."""
    print("\n" + "="*60)
    print("TEST 4: Multiple Sequential Updates")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user
        user_id = insert_user_record(
            conn,
            username="bb3_test_user_multi",
            password_hash="TEST_HASH",
            display_name="Name V1",
            department_display_name="Dept V1"
        )
        
        print(f"✓ Created test user with ID: {user_id}")
        print(f"  V1: DisplayName='Name V1', Dept='Dept V1'")
        
        # First update
        print(f"\nFirst update...")
        update_user_identity_fields(
            conn,
            user_id=user_id,
            display_name="Name V2",
            department_display_name=None
        )
        
        cursor.execute("SELECT DisplayName, DepartmentDisplayName FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        row = cursor.fetchone()
        assert row.DisplayName == "Name V2" and row.DepartmentDisplayName == "Dept V1"
        print(f"✓ After 1st update: DisplayName='Name V2', Dept='Dept V1'")
        
        # Second update
        print(f"\nSecond update...")
        update_user_identity_fields(
            conn,
            user_id=user_id,
            display_name=None,
            department_display_name="Dept V2"
        )
        
        cursor.execute("SELECT DisplayName, DepartmentDisplayName FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        row = cursor.fetchone()
        assert row.DisplayName == "Name V2" and row.DepartmentDisplayName == "Dept V2"
        print(f"✓ After 2nd update: DisplayName='Name V2', Dept='Dept V2'")
        
        # Third update
        print(f"\nThird update...")
        update_user_identity_fields(
            conn,
            user_id=user_id,
            display_name="Name V3",
            department_display_name="Dept V3"
        )
        
        cursor.execute("SELECT DisplayName, DepartmentDisplayName FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        row = cursor.fetchone()
        assert row.DisplayName == "Name V3" and row.DepartmentDisplayName == "Dept V3"
        print(f"✓ After 3rd update: DisplayName='Name V3', Dept='Dept V3'")
        
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
        # Rollback to clean up
        conn.rollback()
        cursor.close()
        conn.close()


def test_user_not_found_raises_error():
    """Test 5: Non-existent user raises ValueError."""
    print("\n" + "="*60)
    print("TEST 5: User Not Found Raises ValueError")
    print("="*60)
    
    conn = get_connection()
    
    try:
        # Try to update non-existent user
        print("Attempting to update user_id=-1 (does not exist)...")
        
        update_user_identity_fields(
            conn,
            user_id=999999,  # Very unlikely to exist
            display_name="Test",
            department_display_name="Test"
        )
        
        # Should not reach here
        print(f"\n✗ TEST 5 FAILED: Non-existent user update was accepted")
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
        conn.rollback()
        conn.close()


def test_invalid_user_id_raises_error():
    """Test 6: Invalid user_id (<=0) raises ValueError."""
    print("\n" + "="*60)
    print("TEST 6: Invalid user_id Raises ValueError")
    print("="*60)
    
    conn = get_connection()
    
    try:
        # Try to update with invalid user_id
        print("Attempting to update user_id=0...")
        
        update_user_identity_fields(
            conn,
            user_id=0,
            display_name="Test",
            department_display_name="Test"
        )
        
        # Should not reach here
        print(f"\n✗ TEST 6 FAILED: Invalid user_id was accepted")
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
        conn.rollback()
        conn.close()


def test_username_unchanged():
    """Test 7: Verify Username is NOT changed by update."""
    print("\n" + "="*60)
    print("TEST 7: Username Remains Unchanged")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user
        original_username = "bb3_test_user_unchanged"
        user_id = insert_user_record(
            conn,
            username=original_username,
            password_hash="TEST_HASH",
            display_name="Name",
            department_display_name="Dept"
        )
        
        print(f"✓ Created test user with username: '{original_username}'")
        
        # Update display fields
        print(f"\nUpdating display fields...")
        update_user_identity_fields(
            conn,
            user_id=user_id,
            display_name="New Name",
            department_display_name="New Dept"
        )
        
        # Verify username unchanged
        cursor.execute("""
            SELECT Username
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        
        # Assertion
        assert row.Username == original_username, f"Username should be unchanged, expected '{original_username}', got '{row.Username}'"
        
        print(f"✓ Username unchanged: '{row.Username}'")
        
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
        # Rollback to clean up
        conn.rollback()
        cursor.close()
        conn.close()


def test_password_hash_unchanged():
    """Test 8: Verify PasswordHash is NOT changed by update."""
    print("\n" + "="*60)
    print("TEST 8: PasswordHash Remains Unchanged")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user
        original_password = "ORIGINAL_HASH_12345"
        user_id = insert_user_record(
            conn,
            username="bb3_test_user_pwd",
            password_hash=original_password,
            display_name="Name",
            department_display_name="Dept"
        )
        
        print(f"✓ Created test user with PasswordHash: '{original_password}'")
        
        # Update display fields
        print(f"\nUpdating display fields...")
        update_user_identity_fields(
            conn,
            user_id=user_id,
            display_name="New Name",
            department_display_name="New Dept"
        )
        
        # Verify password hash unchanged
        cursor.execute("""
            SELECT PasswordHash
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        
        # Assertion
        assert row.PasswordHash == original_password, f"PasswordHash should be unchanged, expected '{original_password}', got '{row.PasswordHash}'"
        
        print(f"✓ PasswordHash unchanged: '{row.PasswordHash}'")
        
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
        # Rollback to clean up
        conn.rollback()
        cursor.close()
        conn.close()


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("PHASE B — B-B3 — DB TEST SUITE — USER IDENTITY UPDATE")
    print("="*60)
    
    tests = [
        ("Update Both Display Fields", test_update_both_fields),
        ("Partial Update - DisplayName Only", test_partial_update_display_name_only),
        ("Partial Update - DepartmentDisplayName Only", test_partial_update_department_only),
        ("Multiple Sequential Updates", test_multiple_updates),
        ("User Not Found Raises ValueError", test_user_not_found_raises_error),
        ("Invalid user_id Raises ValueError", test_invalid_user_id_raises_error),
        ("Username Remains Unchanged", test_username_unchanged),
        ("PasswordHash Remains Unchanged", test_password_hash_unchanged),
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
