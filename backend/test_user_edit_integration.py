"""
Integration tests for User Edit Feature
Tests the complete flow through service layer
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.database import get_connection
from api.services.user_management_service import (
    create_user_with_role_scope,
    update_user_service,
    delete_user_service
)
from api.services.user_credentials_service import get_all_user_credentials_service


def cleanup_test_user(username):
    """Delete test user if it exists"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get user ID
        cursor.execute("SELECT UserID FROM dbo.APP_Users WHERE Username = ?", (username,))
        row = cursor.fetchone()
        
        if row:
            user_id = row[0]
            # Delete scopes
            cursor.execute("DELETE FROM dbo.APP_UserRoleScope WHERE UserID = ?", (user_id,))
            # Delete user
            cursor.execute("DELETE FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
            conn.commit()
            print(f"  Cleaned up test user: {username}")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"  Warning: Cleanup failed: {e}")


def test_update_display_name():
    """Test updating only display name"""
    print("=" * 60)
    print("TEST 1: Update Display Name Only")
    print("=" * 60)
    
    test_username = "test_edit_user_1"
    
    try:
        # Cleanup any existing test user
        cleanup_test_user(test_username)
        
        # Create test user
        print(f"Creating test user: {test_username}")
        user_id = create_user_with_role_scope(
            username=test_username,
            password_plain="TestPassword123!",
            display_name="Original Name",
            department_display_name=None,
            role_id=2,  # Assuming SECTION_ADMIN role
            org_unit_id=10  # Some org unit
        )
        print(f"✓ Created user with ID: {user_id}")
        
        # Update display name
        print(f"Updating display name...")
        result = update_user_service(
            user_id=user_id,
            display_name="Updated Name"
        )
        
        assert result["success"] == True
        assert result["user"]["user_id"] == user_id
        assert result["user"]["username"] == test_username
        assert result["user"]["display_name"] == "Updated Name"
        
        print(f"✓ Display name updated successfully")
        print(f"  Result: {result}")
        
        # Cleanup
        cleanup_test_user(test_username)
        
        print("\n✅ TEST PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        cleanup_test_user(test_username)
        return False


def test_update_username():
    """Test updating username"""
    print("=" * 60)
    print("TEST 2: Update Username")
    print("=" * 60)
    
    old_username = "test_edit_user_2"
    new_username = "test_edited_user_2"
    
    try:
        # Cleanup any existing test users
        cleanup_test_user(old_username)
        cleanup_test_user(new_username)
        
        # Create test user
        print(f"Creating test user: {old_username}")
        user_id = create_user_with_role_scope(
            username=old_username,
            password_plain="TestPassword123!",
            display_name="Test User",
            department_display_name=None,
            role_id=2,
            org_unit_id=10
        )
        print(f"✓ Created user with ID: {user_id}")
        
        # Update username
        print(f"Updating username to: {new_username}")
        result = update_user_service(
            user_id=user_id,
            username=new_username
        )
        
        assert result["success"] == True
        assert result["user"]["user_id"] == user_id
        assert result["user"]["username"] == new_username
        
        print(f"✓ Username updated successfully")
        print(f"  Result: {result}")
        
        # Cleanup
        cleanup_test_user(new_username)
        
        print("\n✅ TEST PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        cleanup_test_user(old_username)
        cleanup_test_user(new_username)
        return False


def test_update_password():
    """Test updating password"""
    print("=" * 60)
    print("TEST 3: Update Password")
    print("=" * 60)
    
    test_username = "test_edit_user_3"
    
    try:
        # Cleanup any existing test user
        cleanup_test_user(test_username)
        
        # Create test user
        print(f"Creating test user: {test_username}")
        user_id = create_user_with_role_scope(
            username=test_username,
            password_plain="OldPassword123!",
            display_name="Test User",
            department_display_name=None,
            role_id=2,
            org_unit_id=10
        )
        print(f"✓ Created user with ID: {user_id}")
        
        # Update password
        print(f"Updating password...")
        result = update_user_service(
            user_id=user_id,
            password="NewPassword456!"
        )
        
        assert result["success"] == True
        assert result["user"]["user_id"] == user_id
        
        print(f"✓ Password updated successfully")
        print(f"  Result: {result}")
        
        # Verify password was stored in TEMP_HASH format
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT PasswordHash FROM dbo.APP_Users WHERE UserID = ?", (user_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if row and row[0].startswith("TEMP_HASH_"):
            print(f"✓ Password stored in TEMP_HASH format for testing")
        else:
            print(f"⚠ Password not in TEMP_HASH format (may be production-hashed)")
        
        # Cleanup
        cleanup_test_user(test_username)
        
        print("\n✅ TEST PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        cleanup_test_user(test_username)
        return False


def test_update_all_fields():
    """Test updating all fields at once"""
    print("=" * 60)
    print("TEST 4: Update All Fields")
    print("=" * 60)
    
    old_username = "test_edit_user_4"
    new_username = "test_complete_edit_4"
    
    try:
        # Cleanup any existing test users
        cleanup_test_user(old_username)
        cleanup_test_user(new_username)
        
        # Create test user
        print(f"Creating test user: {old_username}")
        user_id = create_user_with_role_scope(
            username=old_username,
            password_plain="OldPassword123!",
            display_name="Old Name",
            department_display_name=None,
            role_id=2,
            org_unit_id=10
        )
        print(f"✓ Created user with ID: {user_id}")
        
        # Update all fields
        print(f"Updating all fields...")
        result = update_user_service(
            user_id=user_id,
            display_name="Completely New Name",
            username=new_username,
            password="CompletelyNewPassword!"
        )
        
        assert result["success"] == True
        assert result["user"]["user_id"] == user_id
        assert result["user"]["username"] == new_username
        assert result["user"]["display_name"] == "Completely New Name"
        
        print(f"✓ All fields updated successfully")
        print(f"  Result: {result}")
        
        # Cleanup
        cleanup_test_user(new_username)
        
        print("\n✅ TEST PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        cleanup_test_user(old_username)
        cleanup_test_user(new_username)
        return False


def test_validation_errors():
    """Test validation error cases"""
    print("=" * 60)
    print("TEST 5: Validation Errors")
    print("=" * 60)
    
    test_username = "test_edit_user_5"
    
    try:
        # Cleanup any existing test user
        cleanup_test_user(test_username)
        
        # Create test user
        print(f"Creating test user: {test_username}")
        user_id = create_user_with_role_scope(
            username=test_username,
            password_plain="TestPassword123!",
            display_name="Test User",
            department_display_name=None,
            role_id=2,
            org_unit_id=10
        )
        print(f"✓ Created user with ID: {user_id}")
        
        # Test 1: Username too short
        print("\n  Test 5a: Username too short...")
        try:
            update_user_service(user_id=user_id, username="ab")
            print(f"  ❌ Should have raised error for short username")
            return False
        except Exception as e:
            if "must be 3-50" in str(e).lower():
                print(f"  ✓ Correctly rejected short username")
            else:
                print(f"  ❌ Wrong error: {e}")
                return False
        
        # Test 2: Username with invalid characters
        print("\n  Test 5b: Username with invalid characters...")
        try:
            update_user_service(user_id=user_id, username="test@user")
            print(f"  ❌ Should have raised error for invalid characters")
            return False
        except Exception as e:
            if "must be 3-50" in str(e).lower() or "alphanumeric" in str(e).lower():
                print(f"  ✓ Correctly rejected invalid username")
            else:
                print(f"  ❌ Wrong error: {e}")
                return False
        
        # Test 3: Password too short
        print("\n  Test 5c: Password too short...")
        try:
            update_user_service(user_id=user_id, password="short")
            print(f"  ❌ Should have raised error for short password")
            return False
        except Exception as e:
            if "at least 8 characters" in str(e).lower():
                print(f"  ✓ Correctly rejected short password")
            else:
                print(f"  ❌ Wrong error: {e}")
                return False
        
        # Test 4: Duplicate username
        print("\n  Test 5d: Duplicate username...")
        try:
            update_user_service(user_id=user_id, username="software_admin")
            print(f"  ❌ Should have raised error for duplicate username")
            return False
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  ✓ Correctly rejected duplicate username")
            else:
                print(f"  ❌ Wrong error: {e}")
                return False
        
        # Cleanup
        cleanup_test_user(test_username)
        
        print("\n✅ TEST PASSED - All validations working correctly\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        cleanup_test_user(test_username)
        return False


def test_protection_rules():
    """Test protection rules (cannot edit SOFTWARE_ADMIN)"""
    print("=" * 60)
    print("TEST 6: Protection Rules")
    print("=" * 60)
    
    try:
        # Try to edit user ID 1 (should be SOFTWARE_ADMIN)
        print("Attempting to edit SOFTWARE_ADMIN user (should be blocked)...")
        try:
            update_user_service(user_id=1, display_name="Hacker")
            print(f"❌ Should have blocked editing SOFTWARE_ADMIN user")
            return False
        except Exception as e:
            if "cannot edit" in str(e).lower() and "software_admin" in str(e).lower():
                print(f"✓ Correctly blocked editing SOFTWARE_ADMIN user")
                print(f"  Error message: {e}")
            else:
                print(f"❌ Wrong error: {e}")
                return False
        
        print("\n✅ TEST PASSED - Protection rules working\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("USER EDIT FEATURE - INTEGRATION TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Update Display Name", test_update_display_name()))
    results.append(("Update Username", test_update_username()))
    results.append(("Update Password", test_update_password()))
    results.append(("Update All Fields", test_update_all_fields()))
    results.append(("Validation Errors", test_validation_errors()))
    results.append(("Protection Rules", test_protection_rules()))
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed! Feature is fully functional.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
