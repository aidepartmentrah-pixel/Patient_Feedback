"""
PHASE B — B-B1 — DB TEST — INSERT USER RECORD

Test suite for insert_user_record function.
Tests user creation at the database layer.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.database import get_connection
from backend.api.db_layer.user_management_db import insert_user_record


def cleanup_test_users(conn):
    """Clean up test users from previous runs."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM dbo.APP_Users 
            WHERE Username LIKE 'test_user_bb1%'
        """)
    finally:
        cursor.close()


def test_insert_user_basic():
    """Test 1: Basic user insertion with all fields."""
    print("\n" + "="*60)
    print("TEST 1: Basic User Insert")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Insert user
        username = "test_user_bb1"
        password_hash = "TEST_HASH"
        display_name = "Test User"
        department_display_name = "Test Dept"
        
        print(f"Inserting user: {username}")
        
        user_id = insert_user_record(
            conn,
            username=username,
            password_hash=password_hash,
            display_name=display_name,
            department_display_name=department_display_name
        )
        
        print(f"✓ User created with ID: {user_id}")
        
        # Verify the inserted data
        cursor.execute("""
            SELECT UserID, Username, DisplayName, DepartmentDisplayName, PasswordHash, IsActive
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        
        # Assertions
        assert row is not None, "User not found after insert"
        assert row.Username == username, f"Expected username '{username}', got '{row.Username}'"
        assert row.DisplayName == display_name, f"Expected DisplayName '{display_name}', got '{row.DisplayName}'"
        assert row.DepartmentDisplayName == department_display_name, f"Expected DepartmentDisplayName '{department_display_name}', got '{row.DepartmentDisplayName}'"
        assert row.PasswordHash == password_hash, f"Expected PasswordHash '{password_hash}', got '{row.PasswordHash}'"
        assert row.IsActive == 1, f"Expected IsActive=1, got {row.IsActive}"
        
        print(f"✓ Username: {row.Username}")
        print(f"✓ DisplayName: {row.DisplayName}")
        print(f"✓ DepartmentDisplayName: {row.DepartmentDisplayName}")
        print(f"✓ PasswordHash: {row.PasswordHash}")
        print(f"✓ IsActive: {row.IsActive}")
        
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


def test_insert_user_with_username_trimming():
    """Test 2: Username trimming - spaces should be removed."""
    print("\n" + "="*60)
    print("TEST 2: Username Trimming")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Insert user with spaces in username
        username_with_spaces = "  test_user_bb1_trim  "
        expected_username = "test_user_bb1_trim"
        password_hash = "TEST_HASH_TRIM"
        
        print(f"Inserting user with spaces: '{username_with_spaces}'")
        
        user_id = insert_user_record(
            conn,
            username=username_with_spaces,
            password_hash=password_hash,
            display_name="Trim Test",
            department_display_name="Trim Dept"
        )
        
        print(f"✓ User created with ID: {user_id}")
        
        # Verify username was trimmed
        cursor.execute("""
            SELECT Username
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        
        # Assertions
        assert row is not None, "User not found after insert"
        assert row.Username == expected_username, f"Expected trimmed username '{expected_username}', got '{row.Username}'"
        assert row.Username.strip() == row.Username, "Username should not have leading/trailing spaces"
        
        print(f"✓ Username trimmed correctly: '{row.Username}'")
        
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


def test_insert_user_with_null_display_fields():
    """Test 3: Insert with NULL display fields."""
    print("\n" + "="*60)
    print("TEST 3: NULL Display Fields")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Insert user with NULL display fields
        username = "test_user_bb1_null"
        password_hash = "TEST_HASH_NULL"
        
        print(f"Inserting user with NULL display fields: {username}")
        
        user_id = insert_user_record(
            conn,
            username=username,
            password_hash=password_hash,
            display_name=None,
            department_display_name=None
        )
        
        print(f"✓ User created with ID: {user_id}")
        
        # Verify NULL values
        cursor.execute("""
            SELECT Username, DisplayName, DepartmentDisplayName
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        
        # Assertions
        assert row is not None, "User not found after insert"
        assert row.Username == username, f"Expected username '{username}', got '{row.Username}'"
        assert row.DisplayName is None, f"Expected DisplayName=NULL, got '{row.DisplayName}'"
        assert row.DepartmentDisplayName is None, f"Expected DepartmentDisplayName=NULL, got '{row.DepartmentDisplayName}'"
        
        print(f"✓ Username: {row.Username}")
        print(f"✓ DisplayName: NULL (as expected)")
        print(f"✓ DepartmentDisplayName: NULL (as expected)")
        
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


def test_insert_user_empty_username_raises_error():
    """Test 4: Empty username should raise ValueError."""
    print("\n" + "="*60)
    print("TEST 4: Empty Username Validation")
    print("="*60)
    
    conn = get_connection()
    
    try:
        # Try to insert user with empty username
        print("Attempting to insert user with empty username...")
        
        user_id = insert_user_record(
            conn,
            username="   ",  # Only spaces, will be empty after trim
            password_hash="TEST_HASH",
            display_name="Test",
            department_display_name="Test"
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
        conn.rollback()
        conn.close()


def test_transaction_rollback():
    """Test 5: Transaction rollback works correctly."""
    print("\n" + "="*60)
    print("TEST 5: Transaction Rollback")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        conn.commit()
        
        # Insert user but don't commit
        username = "test_user_bb1_rollback"
        
        print(f"Inserting user: {username}")
        
        user_id = insert_user_record(
            conn,
            username=username,
            password_hash="TEST_HASH_ROLLBACK",
            display_name="Rollback Test",
            department_display_name="Rollback Dept"
        )
        
        print(f"✓ User created with ID: {user_id}")
        
        # Verify user exists in current transaction
        cursor.execute("""
            SELECT COUNT(*) AS cnt
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        assert row.cnt == 1, "User should exist before rollback"
        print(f"✓ User exists in transaction")
        
        # Rollback
        conn.rollback()
        print(f"✓ Transaction rolled back")
        
        # Verify user does NOT exist after rollback
        cursor.execute("""
            SELECT COUNT(*) AS cnt
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        assert row.cnt == 0, "User should NOT exist after rollback"
        print(f"✓ User does not exist after rollback")
        
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
        cursor.close()
        conn.close()


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("PHASE B — B-B1 — DB TEST SUITE — INSERT USER RECORD")
    print("="*60)
    
    tests = [
        ("Basic User Insert", test_insert_user_basic),
        ("Username Trimming", test_insert_user_with_username_trimming),
        ("NULL Display Fields", test_insert_user_with_null_display_fields),
        ("Empty Username Validation", test_insert_user_empty_username_raises_error),
        ("Transaction Rollback", test_transaction_rollback),
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
