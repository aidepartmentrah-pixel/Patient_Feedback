"""
PHASE B — B-B10 — SERVICE TEST — ADMIN RESET PASSWORD

Integration tests for admin_reset_user_password_service.
Tests password reset service layer functionality.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.database import get_connection
from backend.api.db_layer.user_management_db import insert_user_record
from backend.api.services.user_management_service import admin_reset_user_password_service
from backend.api.db_layer.auth_db import hash_password


def cleanup_test_users(conn):
    """Clean up test users from previous runs."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM dbo.APP_Users 
            WHERE Username LIKE 'bb10_reset_%'
        """)
        conn.commit()
    finally:
        cursor.close()


def test_admin_reset_password_success():
    """Test 1: Successfully reset user password."""
    print("\n" + "="*60)
    print("TEST 1: Admin Reset Password - Success")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user with known password hash
        original_hash = hash_password("OriginalPass123!")
        
        user_id = insert_user_record(
            conn,
            username="bb10_reset_success",
            password_hash=original_hash,
            display_name="Reset Test User",
            department_display_name="Test Dept"
        )
        conn.commit()
        
        print(f"✓ Created test user with ID: {user_id}")
        print(f"  Original hash: {original_hash[:20]}...")
        
        # Read original hash from DB to verify
        cursor.execute("""
            SELECT PasswordHash 
            FROM dbo.APP_Users 
            WHERE UserID = ?
        """, (user_id,))
        
        db_original_hash = cursor.fetchone().PasswordHash
        
        assert db_original_hash == original_hash, "Original hash mismatch in DB"
        print(f"✓ Verified original hash in database")
        
        # Reset password via service
        print(f"\nResetting password via service...")
        
        admin_reset_user_password_service(
            user_id=user_id,
            new_password="NewResetPass123!"
        )
        
        print(f"✓ Password reset service completed")
        
        # Read new hash from DB
        cursor.execute("""
            SELECT PasswordHash 
            FROM dbo.APP_Users 
            WHERE UserID = ?
        """, (user_id,))
        
        new_hash = cursor.fetchone().PasswordHash
        
        print(f"  New hash: {new_hash[:20]}...")
        
        # Assertions
        assert new_hash != original_hash, "Password hash did not change"
        assert new_hash.startswith("$2b$"), "New hash is not bcrypt format"
        assert len(new_hash) >= 60, f"New hash too short: {len(new_hash)}"
        
        print(f"✓ Password hash changed successfully")
        print(f"  Original hash: {original_hash[:20]}...")
        print(f"  New hash:      {new_hash[:20]}...")
        
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


def test_admin_reset_password_user_not_found():
    """Test 2: Reset password for non-existent user raises ValueError."""
    print("\n" + "="*60)
    print("TEST 2: Admin Reset Password - User Not Found")
    print("="*60)
    
    try:
        # Try to reset password for non-existent user
        print(f"Attempting to reset password for user_id=-999...")
        
        admin_reset_user_password_service(
            user_id=999999,  # Very unlikely to exist
            new_password="TestPass123!"
        )
        
        # Should not reach here
        print(f"\n✗ TEST 2 FAILED: Non-existent user was accepted")
        return False
        
    except ValueError as e:
        print(f"✓ ValueError raised as expected: {str(e)}")
        print("\n✓ TEST 2 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 2 ERROR: Unexpected exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_admin_reset_password_invalid_user_id():
    """Test 3: Invalid user_id raises ValueError."""
    print("\n" + "="*60)
    print("TEST 3: Admin Reset Password - Invalid user_id")
    print("="*60)
    
    try:
        # Try with user_id = 0
        print(f"Attempting to reset password with user_id=0...")
        
        admin_reset_user_password_service(
            user_id=0,
            new_password="TestPass123!"
        )
        
        # Should not reach here
        print(f"\n✗ TEST 3 FAILED: user_id=0 was accepted")
        return False
        
    except ValueError as e:
        print(f"✓ ValueError raised as expected: {str(e)}")
        print("\n✓ TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 3 ERROR: Unexpected exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_admin_reset_password_empty_password():
    """Test 4: Empty password raises ValueError."""
    print("\n" + "="*60)
    print("TEST 4: Admin Reset Password - Empty Password")
    print("="*60)
    
    try:
        # Try with empty password
        print(f"Attempting to reset password with empty string...")
        
        admin_reset_user_password_service(
            user_id=1,
            new_password=""
        )
        
        # Should not reach here
        print(f"\n✗ TEST 4 FAILED: Empty password was accepted")
        return False
        
    except ValueError as e:
        print(f"✓ ValueError raised as expected: {str(e)}")
        print("\n✓ TEST 4 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 4 ERROR: Unexpected exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_admin_reset_password_whitespace_only():
    """Test 5: Whitespace-only password raises ValueError."""
    print("\n" + "="*60)
    print("TEST 5: Admin Reset Password - Whitespace Only")
    print("="*60)
    
    try:
        # Try with whitespace-only password
        print(f"Attempting to reset password with whitespace only...")
        
        admin_reset_user_password_service(
            user_id=1,
            new_password="      "
        )
        
        # Should not reach here
        print(f"\n✗ TEST 5 FAILED: Whitespace-only password was accepted")
        return False
        
    except ValueError as e:
        print(f"✓ ValueError raised as expected: {str(e)}")
        print("\n✓ TEST 5 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 5 ERROR: Unexpected exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_admin_reset_password_multiple_resets():
    """Test 6: Multiple consecutive password resets."""
    print("\n" + "="*60)
    print("TEST 6: Admin Reset Password - Multiple Resets")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user
        user_id = insert_user_record(
            conn,
            username="bb10_reset_multiple",
            password_hash="INITIAL_HASH",
            display_name="Multiple Reset Test",
            department_display_name="Test"
        )
        conn.commit()
        
        print(f"✓ Created test user with ID: {user_id}")
        
        # First reset
        print(f"\nFirst reset...")
        admin_reset_user_password_service(
            user_id=user_id,
            new_password="FirstPass123!"
        )
        
        cursor.execute("SELECT PasswordHash FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        hash1 = cursor.fetchone().PasswordHash
        print(f"✓ First hash: {hash1[:20]}...")
        
        # Second reset
        print(f"\nSecond reset...")
        admin_reset_user_password_service(
            user_id=user_id,
            new_password="SecondPass123!"
        )
        
        cursor.execute("SELECT PasswordHash FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        hash2 = cursor.fetchone().PasswordHash
        print(f"✓ Second hash: {hash2[:20]}...")
        
        # Third reset
        print(f"\nThird reset...")
        admin_reset_user_password_service(
            user_id=user_id,
            new_password="ThirdPass123!"
        )
        
        cursor.execute("SELECT PasswordHash FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        hash3 = cursor.fetchone().PasswordHash
        print(f"✓ Third hash: {hash3[:20]}...")
        
        # All hashes should be different
        assert hash1 != hash2, "First and second hashes are the same"
        assert hash2 != hash3, "Second and third hashes are the same"
        assert hash1 != hash3, "First and third hashes are the same"
        
        print(f"\n✓ All three password resets produced different hashes")
        
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
        cursor.close()
        conn.close()


def test_admin_reset_password_hash_format():
    """Test 7: Verify password hash is in correct bcrypt format."""
    print("\n" + "="*60)
    print("TEST 7: Admin Reset Password - Hash Format Validation")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user
        user_id = insert_user_record(
            conn,
            username="bb10_reset_format",
            password_hash="OLD_HASH",
            display_name="Format Test",
            department_display_name="Test"
        )
        conn.commit()
        
        print(f"✓ Created test user with ID: {user_id}")
        
        # Reset password
        admin_reset_user_password_service(
            user_id=user_id,
            new_password="FormatTest123!"
        )
        
        # Read hash
        cursor.execute("SELECT PasswordHash FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        new_hash = cursor.fetchone().PasswordHash
        
        # Validate bcrypt format
        assert new_hash.startswith("$2b$"), f"Hash doesn't start with $2b$: {new_hash[:10]}"
        assert len(new_hash) == 60, f"Hash length is {len(new_hash)}, expected 60"
        
        # Bcrypt format: $2b$<cost>$<salt><hash>
        parts = new_hash.split("$")
        assert len(parts) == 4, f"Hash has {len(parts)} parts, expected 4"
        assert parts[1] == "2b", f"Algorithm version is {parts[1]}, expected 2b"
        assert parts[2].isdigit(), f"Cost factor is not numeric: {parts[2]}"
        assert len(parts[3]) == 53, f"Salt+hash length is {len(parts[3])}, expected 53"
        
        print(f"✓ Hash format validated: {new_hash[:30]}...")
        print(f"  Algorithm: $2b$")
        print(f"  Cost factor: {parts[2]}")
        print(f"  Total length: {len(new_hash)} chars")
        
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
        cleanup_test_users(conn)
        cursor.close()
        conn.close()


def test_admin_reset_password_transaction_rollback():
    """Test 8: Transaction rollback on error."""
    print("\n" + "="*60)
    print("TEST 8: Admin Reset Password - Transaction Rollback")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clean up first
        cleanup_test_users(conn)
        
        # Create test user
        original_hash = "ORIGINAL_HASH_VALUE"
        user_id = insert_user_record(
            conn,
            username="bb10_reset_rollback",
            password_hash=original_hash,
            display_name="Rollback Test",
            department_display_name="Test"
        )
        conn.commit()
        
        print(f"✓ Created test user with ID: {user_id}")
        
        # Try to reset with invalid user_id (should fail and rollback)
        print(f"\nAttempting invalid reset (should rollback)...")
        try:
            admin_reset_user_password_service(
                user_id=-1,
                new_password="ShouldFail123!"
            )
        except ValueError:
            print(f"✓ ValueError raised as expected")
        
        # Verify original user's hash is unchanged
        cursor.execute("SELECT PasswordHash FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        current_hash = cursor.fetchone().PasswordHash
        
        assert current_hash == original_hash, "Hash was modified despite rollback"
        
        print(f"✓ Original user's hash unchanged after rollback")
        
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
    print("PHASE B — B-B10 — SERVICE TEST SUITE — ADMIN RESET PASSWORD")
    print("="*60)
    
    tests = [
        ("Admin Reset Password - Success", test_admin_reset_password_success),
        ("Admin Reset Password - User Not Found", test_admin_reset_password_user_not_found),
        ("Admin Reset Password - Invalid user_id", test_admin_reset_password_invalid_user_id),
        ("Admin Reset Password - Empty Password", test_admin_reset_password_empty_password),
        ("Admin Reset Password - Whitespace Only", test_admin_reset_password_whitespace_only),
        ("Admin Reset Password - Multiple Resets", test_admin_reset_password_multiple_resets),
        ("Admin Reset Password - Hash Format", test_admin_reset_password_hash_format),
        ("Admin Reset Password - Transaction Rollback", test_admin_reset_password_transaction_rollback),
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
