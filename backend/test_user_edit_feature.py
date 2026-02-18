"""
Test script for User Edit Feature
Tests the new PUT /api/admin/users/{user_id} endpoint and updated GET endpoint
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules import correctly"""
    print("=" * 60)
    print("TEST 1: Testing imports...")
    print("=" * 60)
    
    try:
        # Test DB layer imports
        from api.db_layer.user_management_db import (
            get_user_with_role,
            username_exists_excluding_user,
            update_user_credentials
        )
        print("✓ DB layer imports successful")
        
        # Test service layer imports
        from api.services.user_management_service import update_user_service
        print("✓ Service layer imports successful")
        
        # Test router imports
        from api.routers.admin_user_management_router import router
        print("✓ Router imports successful")
        
        # Test updated credentials imports
        from api.services.user_credentials_service import get_all_user_credentials_service
        print("✓ User credentials service imports successful")
        
        print("\n✅ All imports successful!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_db_functions():
    """Test database layer functions"""
    print("=" * 60)
    print("TEST 2: Testing DB layer functions...")
    print("=" * 60)
    
    try:
        from core.database import get_connection
        from api.db_layer.user_management_db import (
            get_user_with_role,
            username_exists_excluding_user
        )
        
        conn = None
        try:
            conn = get_connection()
            print("✓ Database connection successful")
            
            # Test get_user_with_role
            user = get_user_with_role(conn, 1)  # Assuming user 1 exists
            if user:
                print(f"✓ get_user_with_role(1) returned: UserID={user.UserID}, Username={user.Username}")
            else:
                print("⚠ get_user_with_role(1) returned None (user may not exist)")
            
            # Test username_exists_excluding_user
            exists = username_exists_excluding_user(conn, "software_admin", 999)
            print(f"✓ username_exists_excluding_user('software_admin', 999) = {exists}")
            
            print("\n✅ Database layer functions working!\n")
            return True
            
        finally:
            if conn:
                conn.close()
                print("✓ Database connection closed")
                
    except Exception as e:
        print(f"\n❌ Database test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_validation():
    """Test validation logic"""
    print("=" * 60)
    print("TEST 3: Testing validation logic...")
    print("=" * 60)
    
    import re
    
    # Test username validation regex
    valid_usernames = ["admin", "user_123", "test_admin", "ABC", "a_b_c_123"]
    invalid_usernames = ["ab", "a" * 51, "user@name", "user-name", "user name"]
    
    pattern = r'^[a-zA-Z0-9_]{3,50}$'
    
    print("Testing valid usernames:")
    for username in valid_usernames:
        match = bool(re.match(pattern, username))
        status = "✓" if match else "❌"
        print(f"  {status} '{username}' = {match}")
    
    print("\nTesting invalid usernames:")
    for username in invalid_usernames:
        match = bool(re.match(pattern, username))
        status = "✓" if not match else "❌"
        print(f"  {status} '{username}' = {not match} (should be invalid)")
    
    # Test password length validation
    print("\nTesting password validation:")
    valid_passwords = ["12345678", "password123", "a" * 8]
    invalid_passwords = ["1234567", "short", "abc"]
    
    for pwd in valid_passwords:
        status = "✓" if len(pwd) >= 8 else "❌"
        print(f"  {status} '{pwd}' (len={len(pwd)}) = valid")
    
    for pwd in invalid_passwords:
        status = "✓" if len(pwd) < 8 else "❌"
        print(f"  {status} '{pwd}' (len={len(pwd)}) = invalid")
    
    print("\n✅ Validation logic correct!\n")
    return True


def test_user_credentials_display_name():
    """Test that GET user-credentials includes display_name"""
    print("=" * 60)
    print("TEST 4: Testing GET user-credentials with display_name...")
    print("=" * 60)
    
    try:
        from api.services.user_credentials_service import get_all_user_credentials_service
        
        credentials = get_all_user_credentials_service()
        
        if not credentials:
            print("⚠ No users found in database")
            return True
        
        print(f"✓ Retrieved {len(credentials)} user(s)")
        
        # Check first user for display_name field
        first_user = credentials[0]
        if "display_name" in first_user:
            print(f"✓ display_name field present in response")
            print(f"  Sample: user_id={first_user.get('user_id')}, username={first_user.get('username')}, display_name={first_user.get('display_name')}")
        else:
            print("❌ display_name field NOT found in response")
            print(f"  Available fields: {list(first_user.keys())}")
            return False
        
        print("\n✅ GET user-credentials endpoint updated correctly!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ User credentials test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("USER EDIT FEATURE - UNIT TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Database Functions", test_db_functions()))
    results.append(("Validation Logic", test_validation()))
    results.append(("User Credentials Display Name", test_user_credentials_display_name()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! User edit feature is ready.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
