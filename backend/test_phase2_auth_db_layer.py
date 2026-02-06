"""
Comprehensive Test Suite for Auth DB Layer
===========================================
Tests all authentication database functions with real database connections.

Tests:
1. get_user_by_id() - Load user by ID
2. get_user_by_username() - Load user by username
3. get_user_with_scopes() - Load user with all role scopes
4. validate_user_credentials() - Login validation
5. Utility functions - Password hashing and updates

Run from backend directory:
    python test_phase2_auth_db_layer.py
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from api.db_layer.auth_db import (
    get_user_by_id,
    get_user_by_username,
    get_user_with_scopes,
    validate_user_credentials,
    hash_password,
    update_user_password
)


# ==================== TEST UTILITIES ====================

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name):
        self.passed += 1
        print(f"  ✓ {test_name}")
    
    def add_fail(self, test_name, reason):
        self.failed += 1
        self.errors.append((test_name, reason))
        print(f"  ✗ {test_name}: {reason}")
    
    def summary(self):
        total = self.passed + self.failed
        print("\n" + "="*70)
        print(" " * 25 + "TEST SUMMARY")
        print("="*70)
        print(f"Total Tests:  {total}")
        print(f"Passed:       {self.passed} ✓")
        print(f"Failed:       {self.failed} ✗")
        print(f"Success Rate: {(self.passed/total*100) if total > 0 else 0:.1f}%")
        
        if self.errors:
            print("\n" + "-"*70)
            print("FAILED TESTS:")
            for test_name, reason in self.errors:
                print(f"  • {test_name}")
                print(f"    Reason: {reason}")
        
        print("="*70 + "\n")
        
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED! Auth DB layer is working perfectly!\n")
            return True
        else:
            print(f"⚠️  {self.failed} TEST(S) FAILED! Please review errors above.\n")
            return False


def assert_dict_has_keys(data, required_keys, test_name, results):
    """Helper to verify dictionary structure."""
    missing = [k for k in required_keys if k not in data]
    if missing:
        results.add_fail(test_name, f"Missing keys: {missing}")
        return False
    return True


# ==================== TEST CASES ====================

def test_get_user_by_id(results):
    """Test loading user by ID (without scopes)."""
    print("\n" + "="*70)
    print("TEST 1: get_user_by_id()")
    print("="*70)
    
    # Test 1.1: Load existing user
    try:
        user = get_user_by_id(1)  # software_admin
        
        if user is None:
            results.add_fail("Load user by ID 1", "Returned None")
        elif not assert_dict_has_keys(user, ["user_id", "username", "is_active", "scopes"], "User dict structure", results):
            pass  # Already logged
        elif user["user_id"] != 1:
            results.add_fail("User ID matches", f"Expected 1, got {user['user_id']}")
        elif user["username"] != "software_admin":
            results.add_fail("Username matches", f"Expected 'software_admin', got '{user['username']}'")
        elif user["is_active"] != True:
            results.add_fail("User is active", f"Expected True, got {user['is_active']}")
        elif not isinstance(user["scopes"], list):
            results.add_fail("Scopes is list", f"Expected list, got {type(user['scopes'])}")
        else:
            results.add_pass("Load user by ID 1 (software_admin)")
    except Exception as e:
        results.add_fail("Load user by ID 1", str(e))
    
    # Test 1.2: Load non-existent user
    try:
        user = get_user_by_id(99999)
        
        if user is not None:
            results.add_fail("Load non-existent user", f"Expected None, got {user}")
        else:
            results.add_pass("Load non-existent user returns None")
    except Exception as e:
        results.add_fail("Load non-existent user", str(e))
    
    # Test 1.3: Load multiple users
    test_users = [
        (2, "worker"),
        (3, "complaint_supervisor"),
        (4, "section_admin"),
        (5, "department_admin"),
        (6, "administration_admin")
    ]
    
    for user_id, expected_username in test_users:
        try:
            user = get_user_by_id(user_id)
            if user and user["username"] == expected_username:
                results.add_pass(f"Load user ID {user_id} ({expected_username})")
            else:
                results.add_fail(f"Load user ID {user_id}", f"Expected username '{expected_username}', got {user}")
        except Exception as e:
            results.add_fail(f"Load user ID {user_id}", str(e))


def test_get_user_by_username(results):
    """Test loading user by username (without scopes)."""
    print("\n" + "="*70)
    print("TEST 2: get_user_by_username()")
    print("="*70)
    
    # Test 2.1: Load existing user
    try:
        user = get_user_by_username("software_admin")
        
        if user is None:
            results.add_fail("Load user by username 'software_admin'", "Returned None")
        elif not assert_dict_has_keys(user, ["user_id", "username", "is_active", "scopes"], "User dict structure", results):
            pass
        elif user["username"] != "software_admin":
            results.add_fail("Username matches", f"Expected 'software_admin', got '{user['username']}'")
        elif user["user_id"] != 1:
            results.add_fail("User ID matches", f"Expected 1, got {user['user_id']}")
        else:
            results.add_pass("Load user by username 'software_admin'")
    except Exception as e:
        results.add_fail("Load user by username 'software_admin'", str(e))
    
    # Test 2.2: Load non-existent user
    try:
        user = get_user_by_username("nonexistent_user_xyz")
        
        if user is not None:
            results.add_fail("Load non-existent username", f"Expected None, got {user}")
        else:
            results.add_pass("Load non-existent username returns None")
    except Exception as e:
        results.add_fail("Load non-existent username", str(e))
    
    # Test 2.3: Load multiple users
    test_users = [
        "worker",
        "complaint_supervisor",
        "section_admin",
        "department_admin",
        "administration_admin"
    ]
    
    for username in test_users:
        try:
            user = get_user_by_username(username)
            if user and user["username"] == username:
                results.add_pass(f"Load user by username '{username}'")
            else:
                results.add_fail(f"Load username '{username}'", f"Expected username '{username}', got {user}")
        except Exception as e:
            results.add_fail(f"Load username '{username}'", str(e))


def test_get_user_with_scopes(results):
    """Test loading user with all role scopes."""
    print("\n" + "="*70)
    print("TEST 3: get_user_with_scopes()")
    print("="*70)
    
    # Test 3.1: Load user with scopes - software_admin
    try:
        user = get_user_with_scopes(1)
        
        if user is None:
            results.add_fail("Load user ID 1 with scopes", "Returned None")
        elif not assert_dict_has_keys(user, ["user_id", "username", "is_active", "scopes"], "User dict structure", results):
            pass
        elif not isinstance(user["scopes"], list):
            results.add_fail("Scopes is list", f"Expected list, got {type(user['scopes'])}")
        elif len(user["scopes"]) == 0:
            results.add_fail("User has scopes", "Scopes list is empty")
        else:
            # Verify scope structure
            scope = user["scopes"][0]
            if not assert_dict_has_keys(scope, ["role_code", "org_unit_id", "org_unit_type"], "Scope dict structure", results):
                pass
            else:
                results.add_pass(f"Load user ID 1 with {len(user['scopes'])} scope(s)")
                
                # Print scope details for verification
                print(f"    → User: {user['username']}")
                print(f"    → Scopes:")
                for s in user["scopes"]:
                    print(f"      • {s['role_code']} → {s['org_unit_type']}({s['org_unit_id']})")
    except Exception as e:
        results.add_fail("Load user ID 1 with scopes", str(e))
    
    # Test 3.2: Verify specific role mappings
    expected_mappings = [
        (1, "software_admin", "SOFTWARE_ADMIN", 0, "ADMINISTRATION"),
        (2, "worker", "WORKER", 10, "COMPLAINT"),
        (3, "complaint_supervisor", "COMPLAINT_SUPERVISOR", 10, "COMPLAINT"),
        (4, "section_admin", "SECTION_ADMIN", 10, "SECTION"),
        (5, "department_admin", "DEPARTMENT_ADMIN", 5, "DEPARTMENT"),
        (6, "administration_admin", "ADMINISTRATION_ADMIN", 1, "ADMINISTRATION"),
    ]
    
    for user_id, username, role_code, org_unit_id, org_unit_type in expected_mappings:
        try:
            user = get_user_with_scopes(user_id)
            
            if not user:
                results.add_fail(f"Load {username} with scopes", "User not found")
                continue
            
            # Find matching scope
            matching_scope = None
            for scope in user["scopes"]:
                if (scope["role_code"] == role_code and 
                    scope["org_unit_id"] == org_unit_id and 
                    scope["org_unit_type"] == org_unit_type):
                    matching_scope = scope
                    break
            
            if matching_scope:
                results.add_pass(f"Verify {username} → {role_code} → {org_unit_type}({org_unit_id})")
            else:
                results.add_fail(
                    f"Verify {username} role mapping",
                    f"Expected {role_code} → {org_unit_type}({org_unit_id}), not found in {user['scopes']}"
                )
        except Exception as e:
            results.add_fail(f"Verify {username} role mapping", str(e))
    
    # Test 3.3: Non-existent user
    try:
        user = get_user_with_scopes(99999)
        if user is None:
            results.add_pass("Non-existent user with scopes returns None")
        else:
            results.add_fail("Non-existent user with scopes", f"Expected None, got {user}")
    except Exception as e:
        results.add_fail("Non-existent user with scopes", str(e))


def test_validate_user_credentials(results):
    """Test user login validation."""
    print("\n" + "="*70)
    print("TEST 4: validate_user_credentials()")
    print("="*70)
    
    # Test 4.1: Valid credentials (temp hash)
    test_credentials = [
        ("software_admin", "admin123"),
        ("worker", "worker123"),
        ("complaint_supervisor", "sup123"),
        ("section_admin", "section123"),
        ("department_admin", "dept123"),
        ("administration_admin", "adminis123"),
    ]
    
    for username, password in test_credentials:
        try:
            user = validate_user_credentials(username, password)
            
            if user is None:
                results.add_fail(f"Login {username} with correct password", "Returned None")
            elif not assert_dict_has_keys(user, ["user_id", "username", "is_active", "scopes"], "User dict structure", results):
                pass
            elif user["username"] != username:
                results.add_fail(f"Login {username}", f"Username mismatch: {user['username']}")
            elif not user["is_active"]:
                results.add_fail(f"Login {username}", "User not active")
            elif not isinstance(user["scopes"], list):
                results.add_fail(f"Login {username}", f"Scopes not list: {type(user['scopes'])}")
            elif len(user["scopes"]) == 0:
                results.add_fail(f"Login {username}", "No scopes loaded")
            else:
                results.add_pass(f"Login {username} with correct password ({len(user['scopes'])} scope(s))")
        except Exception as e:
            results.add_fail(f"Login {username}", str(e))
    
    # Test 4.2: Invalid passwords
    invalid_tests = [
        ("software_admin", "wrongpassword"),
        ("worker", "incorrect"),
        ("section_admin", ""),
        ("department_admin", "12345"),
    ]
    
    for username, wrong_password in invalid_tests:
        try:
            user = validate_user_credentials(username, wrong_password)
            
            if user is None:
                results.add_pass(f"Login {username} with wrong password returns None")
            else:
                results.add_fail(f"Login {username} with wrong password", f"Expected None, got {user}")
        except Exception as e:
            results.add_fail(f"Login {username} with wrong password", str(e))
    
    # Test 4.3: Non-existent user
    try:
        user = validate_user_credentials("nonexistent_user", "anypassword")
        
        if user is None:
            results.add_pass("Login non-existent user returns None")
        else:
            results.add_fail("Login non-existent user", f"Expected None, got {user}")
    except Exception as e:
        results.add_fail("Login non-existent user", str(e))


def test_password_utilities(results):
    """Test password hashing and update utilities."""
    print("\n" + "="*70)
    print("TEST 5: Password Utility Functions")
    print("="*70)
    
    # Test 5.1: Hash password
    try:
        test_password = "test_password_123"
        hashed = hash_password(test_password)
        
        if not hashed:
            results.add_fail("Hash password", "Returned empty hash")
        elif not hashed.startswith("$2b$"):
            results.add_fail("Hash password", f"Invalid bcrypt format: {hashed[:20]}...")
        else:
            results.add_pass(f"Hash password generates valid bcrypt hash")
            print(f"    → Hash: {hashed[:50]}...")
    except Exception as e:
        results.add_fail("Hash password", str(e))
    
    # Test 5.2: Update user password (use a test user)
    try:
        test_user_id = 2  # worker
        new_password = "new_test_password_456"
        new_hash = hash_password(new_password)
        
        success = update_user_password(test_user_id, new_hash)
        
        if not success:
            results.add_fail("Update user password", "Update returned False")
        else:
            results.add_pass("Update user password returns True")
            
            # Verify the new password works
            user = validate_user_credentials("worker", new_password)
            
            if user:
                results.add_pass("Login with new password succeeds")
                
                # Restore original temp hash for other tests
                update_user_password(test_user_id, "TEMP_HASH_worker123")
                results.add_pass("Restore original password hash")
            else:
                results.add_fail("Login with new password", "Validation failed")
    except Exception as e:
        results.add_fail("Update user password", str(e))
        # Try to restore anyway
        try:
            update_user_password(2, "TEMP_HASH_worker123")
        except:
            pass
    
    # Test 5.3: Update non-existent user
    try:
        success = update_user_password(99999, "some_hash")
        
        if success:
            results.add_fail("Update non-existent user", "Expected False, got True")
        else:
            results.add_pass("Update non-existent user returns False")
    except Exception as e:
        results.add_fail("Update non-existent user", str(e))


# ==================== MAIN TEST RUNNER ====================

def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*70)
    print(" " * 15 + "AUTH DB LAYER - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("\nTesting authentication database layer functions...")
    print("Database: IncidentManager")
    print("Tables: APP_Users, APP_Roles, APP_UserRoleScope")
    print()
    
    results = TestResult()
    
    try:
        # Run all test suites
        test_get_user_by_id(results)
        test_get_user_by_username(results)
        test_get_user_with_scopes(results)
        test_validate_user_credentials(results)
        test_password_utilities(results)
        
        # Print summary
        success = results.summary()
        
        return success
        
    except Exception as e:
        print(f"\n✗ Critical error during test execution: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
