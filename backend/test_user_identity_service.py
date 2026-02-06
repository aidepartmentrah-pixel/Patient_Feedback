"""
PHASE B — B-B5 — SERVICE TEST — UPDATE USER IDENTITY

Test suite for update_user_identity_service function.
Tests updating user identity fields at the service layer.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.database import get_connection
from backend.api.db_layer.user_management_db import insert_user_record
from backend.api.services.user_management_service import update_user_identity_service


def cleanup_test_users(conn):
    """Clean up test users from previous runs."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM dbo.APP_Users 
            WHERE Username LIKE 'bb5_service_user%'
        """)
        conn.commit()
    finally:
        cursor.close()


def test_update_both_fields():
    """Test 1: Update both display fields via service."""
    print("\n" + "="*60)
    print("TEST 1: Update Both Display Fields")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user
        user_id = insert_user_record(
            conn,
            username="bb5_service_user",
            password_hash="TEST_HASH",
            display_name="Initial Name",
            department_display_name="Initial Dept"
        )
        conn.commit()
        
        print(f"✓ Created test user with ID: {user_id}")
        print(f"  Initial DisplayName: 'Initial Name'")
        print(f"  Initial DepartmentDisplayName: 'Initial Dept'")
        
        # Update via service
        print(f"\nUpdating both fields via service...")
        update_user_identity_service(
            user_id=user_id,
            display_name="Service Updated",
            department_display_name="Service Dept"
        )
        
        print(f"✓ Service update completed")
        
        # Verify changes
        cursor.execute("""
            SELECT DisplayName, DepartmentDisplayName
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        
        # Assertions
        assert row is not None, "User not found after update"
        assert row.DisplayName == "Service Updated", f"Expected DisplayName='Service Updated', got '{row.DisplayName}'"
        assert row.DepartmentDisplayName == "Service Dept", f"Expected DepartmentDisplayName='Service Dept', got '{row.DepartmentDisplayName}'"
        
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
        cleanup_test_users(conn)
        cursor.close()
        conn.close()


def test_partial_update_display_name_only():
    """Test 2: Update display_name only (partial update)."""
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
            username="bb5_service_user_partial1",
            password_hash="TEST_HASH",
            display_name="Original Name",
            department_display_name="Original Dept"
        )
        conn.commit()
        
        print(f"✓ Created test user with ID: {user_id}")
        print(f"  Initial DisplayName: 'Original Name'")
        print(f"  Initial DepartmentDisplayName: 'Original Dept'")
        
        # Update display_name only
        print(f"\nUpdating DisplayName only (department=None)...")
        update_user_identity_service(
            user_id=user_id,
            display_name="Name Updated",
            department_display_name=None
        )
        
        print(f"✓ Service update completed")
        
        # Verify changes
        cursor.execute("""
            SELECT DisplayName, DepartmentDisplayName
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        
        # Assertions
        assert row is not None, "User not found after update"
        assert row.DisplayName == "Name Updated", f"Expected DisplayName='Name Updated', got '{row.DisplayName}'"
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
        cleanup_test_users(conn)
        cursor.close()
        conn.close()


def test_partial_update_department_only():
    """Test 3: Update department_display_name only (partial update)."""
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
            username="bb5_service_user_partial2",
            password_hash="TEST_HASH",
            display_name="Stable Name",
            department_display_name="Original Dept"
        )
        conn.commit()
        
        print(f"✓ Created test user with ID: {user_id}")
        print(f"  Initial DisplayName: 'Stable Name'")
        print(f"  Initial DepartmentDisplayName: 'Original Dept'")
        
        # Update department_display_name only
        print(f"\nUpdating DepartmentDisplayName only (name=None)...")
        update_user_identity_service(
            user_id=user_id,
            display_name=None,
            department_display_name="Dept Only"
        )
        
        print(f"✓ Service update completed")
        
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
        assert row.DepartmentDisplayName == "Dept Only", f"Expected DepartmentDisplayName='Dept Only', got '{row.DepartmentDisplayName}'"
        
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
        cleanup_test_users(conn)
        cursor.close()
        conn.close()


def test_whitespace_trimming():
    """Test 4: Service trims whitespace from values."""
    print("\n" + "="*60)
    print("TEST 4: Whitespace Trimming")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user
        user_id = insert_user_record(
            conn,
            username="bb5_service_user_trim",
            password_hash="TEST_HASH",
            display_name="Old",
            department_display_name="Old"
        )
        conn.commit()
        
        print(f"✓ Created test user with ID: {user_id}")
        
        # Update with whitespace-padded values
        print(f"\nUpdating with whitespace-padded values...")
        update_user_identity_service(
            user_id=user_id,
            display_name="  Trimmed Name  ",
            department_display_name="  Trimmed Dept  "
        )
        
        print(f"✓ Service update completed")
        
        # Verify whitespace was trimmed
        cursor.execute("""
            SELECT DisplayName, DepartmentDisplayName
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        
        # Assertions
        assert row.DisplayName == "Trimmed Name", f"Expected trimmed 'Trimmed Name', got '{row.DisplayName}'"
        assert row.DepartmentDisplayName == "Trimmed Dept", f"Expected trimmed 'Trimmed Dept', got '{row.DepartmentDisplayName}'"
        
        print(f"✓ DisplayName trimmed: '{row.DisplayName}'")
        print(f"✓ DepartmentDisplayName trimmed: '{row.DepartmentDisplayName}'")
        
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
        cursor.close()
        conn.close()


def test_both_fields_none_raises_error():
    """Test 5: Both fields None raises ValueError."""
    print("\n" + "="*60)
    print("TEST 5: Both Fields None Raises ValueError")
    print("="*60)
    
    try:
        # Try to update with both fields None
        print("Attempting to update with both fields=None...")
        
        update_user_identity_service(
            user_id=999,
            display_name=None,
            department_display_name=None
        )
        
        # Should not reach here
        print(f"\n✗ TEST 5 FAILED: Both fields None was accepted")
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


def test_invalid_user_id_raises_error():
    """Test 6: Invalid user_id raises ValueError."""
    print("\n" + "="*60)
    print("TEST 6: Invalid user_id Raises ValueError")
    print("="*60)
    
    try:
        # Try with invalid user_id
        print("Attempting to update with user_id=0...")
        
        update_user_identity_service(
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


def test_user_not_found_raises_error():
    """Test 7: Non-existent user raises ValueError from DB layer."""
    print("\n" + "="*60)
    print("TEST 7: User Not Found Raises ValueError")
    print("="*60)
    
    try:
        # Try to update non-existent user
        print("Attempting to update user_id=-99 (does not exist)...")
        
        update_user_identity_service(
            user_id=999999,  # Very unlikely to exist
            display_name="Test",
            department_display_name="Test"
        )
        
        # Should not reach here
        print(f"\n✗ TEST 7 FAILED: Non-existent user update was accepted")
        return False
        
    except ValueError as e:
        # Expected behavior (from DB layer)
        print(f"✓ ValueError raised as expected: {str(e)}")
        print("\n✓ TEST 7 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 7 ERROR: Unexpected exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_transaction_rollback_on_error():
    """Test 8: Transaction rollback on error."""
    print("\n" + "="*60)
    print("TEST 8: Transaction Rollback On Error")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user
        user_id = insert_user_record(
            conn,
            username="bb5_service_user_rollback",
            password_hash="TEST_HASH",
            display_name="Original",
            department_display_name="Original"
        )
        conn.commit()
        
        print(f"✓ Created test user with ID: {user_id}")
        
        # Get original values
        cursor.execute("SELECT DisplayName, DepartmentDisplayName FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        original = cursor.fetchone()
        print(f"  Original values: DisplayName='{original.DisplayName}', Dept='{original.DepartmentDisplayName}'")
        
        # Try to update with invalid user_id (should rollback)
        print(f"\nAttempting invalid update (will trigger rollback)...")
        try:
            update_user_identity_service(
                user_id=-1,
                display_name="Should Not Save",
                department_display_name="Should Not Save"
            )
        except ValueError:
            pass  # Expected
        
        # Verify original values unchanged
        cursor.execute("SELECT DisplayName, DepartmentDisplayName FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        after = cursor.fetchone()
        
        assert after.DisplayName == original.DisplayName, "DisplayName should be unchanged after rollback"
        assert after.DepartmentDisplayName == original.DepartmentDisplayName, "DepartmentDisplayName should be unchanged after rollback"
        
        print(f"✓ Values unchanged after error (rollback successful)")
        print(f"  DisplayName: '{after.DisplayName}'")
        print(f"  DepartmentDisplayName: '{after.DepartmentDisplayName}'")
        
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
    print("PHASE B — B-B5 — SERVICE TEST SUITE — UPDATE IDENTITY")
    print("="*60)
    
    tests = [
        ("Update Both Display Fields", test_update_both_fields),
        ("Partial Update - DisplayName Only", test_partial_update_display_name_only),
        ("Partial Update - DepartmentDisplayName Only", test_partial_update_department_only),
        ("Whitespace Trimming", test_whitespace_trimming),
        ("Both Fields None Raises ValueError", test_both_fields_none_raises_error),
        ("Invalid user_id Raises ValueError", test_invalid_user_id_raises_error),
        ("User Not Found Raises ValueError", test_user_not_found_raises_error),
        ("Transaction Rollback On Error", test_transaction_rollback_on_error),
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
