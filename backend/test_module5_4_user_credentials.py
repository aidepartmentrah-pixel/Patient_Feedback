"""
Test for MODULE 5.4 - List All User Credentials (TEST ONLY)
Tests admin endpoint for viewing all user accounts with test passwords.

⚠️ TEST ONLY — This endpoint should be disabled in production

Run from backend directory:
    python test_module5_4_user_credentials.py
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from main import app

# Create test client
client = TestClient(app)


def clear_sessions():
    """Clear all test client sessions."""
    client.cookies.clear()


def login_as_admin():
    """Helper to login as software_admin."""
    response = client.post(
        "/api/auth/login",
        json={"username": "software_admin", "password": "admin123"}
    )
    return response


# ==================== TESTS ====================

def test_credentials_without_login():
    """Test that endpoint requires authentication."""
    print("\n" + "="*70)
    print("TEST 1: User Credentials - Without Login")
    print("="*70)
    
    clear_sessions()
    
    response = client.get("/api/admin/testing/user-credentials")
    
    if response.status_code == 401:
        print("✅ PASS: Returns 401 Unauthorized when not logged in")
    else:
        print(f"❌ FAIL: Expected 401, got {response.status_code}")
        print(f"Response: {response.text}")


def test_credentials_with_admin():
    """Test successful credential listing with admin user."""
    print("\n" + "="*70)
    print("TEST 2: User Credentials - With Admin Login")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed with {login_response.status_code}")
        return
    
    print("✓ Logged in as software_admin")
    
    # Get all user credentials
    response = client.get("/api/admin/testing/user-credentials")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ PASS: Credentials retrieved successfully")
        print(f"\n   Total users returned: {len(data)}")
        
        if len(data) > 0:
            print(f"\n   Sample credentials (first 3 users):")
            for i, user in enumerate(data[:3]):
                print(f"\n   User {i+1}:")
                print(f"     user_id: {user.get('user_id')}")
                print(f"     username: {user.get('username')}")
                print(f"     role: {user.get('role')}")
                print(f"     org_unit: {user.get('org_unit')}")
                print(f"     active: {user.get('active')}")
                print(f"     test_password: {user.get('test_password')}")
            
            # Verify data structure
            required_fields = ['user_id', 'username', 'active', 'test_password']
            first_user = data[0]
            missing_fields = [f for f in required_fields if f not in first_user]
            
            if not missing_fields:
                print(f"\n   ✓ All required fields present")
            else:
                print(f"\n   ⚠️  Missing fields: {missing_fields}")
            
            # Check for TEMP_HASH passwords
            temp_hash_users = [u for u in data if u.get('test_password')]
            print(f"\n   Users with TEMP_HASH passwords: {len(temp_hash_users)}")
            
            # Verify no PasswordHash field is exposed
            has_password_hash = any('PasswordHash' in u or 'password_hash' in u for u in data)
            if not has_password_hash:
                print(f"   ✓ PasswordHash not exposed (secure)")
            else:
                print(f"   ❌ WARNING: PasswordHash field found in response!")
        else:
            print(f"   ⚠️  No users found (database may be empty)")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
        print(f"Response: {response.text}")


def test_credentials_verify_specific_user():
    """Test that specific known users are in the list."""
    print("\n" + "="*70)
    print("TEST 3: Verify Known Users Present")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed")
        return
    
    print("✓ Logged in as software_admin")
    
    # Get credentials
    response = client.get("/api/admin/testing/user-credentials")
    
    if response.status_code != 200:
        print(f"❌ FAIL: Credentials request failed")
        return
    
    data = response.json()
    
    # Check for known test users
    known_users = ['software_admin', 'worker', 'supervisor', 'section_admin', 'dept_admin', 'admin_admin']
    found_users = [u.get('username') for u in data]
    
    print(f"Looking for known test users:")
    for known_user in known_users:
        if known_user in found_users:
            user_data = next(u for u in data if u.get('username') == known_user)
            print(f"   ✓ Found: {known_user} (password: {user_data.get('test_password')})")
        else:
            print(f"   ⚠️  Not found: {known_user}")
    
    if any(user in found_users for user in known_users):
        print(f"\n✅ PASS: At least one known test user found")
    else:
        print(f"\n⚠️  No known test users found (may need to run bulk user creation)")


def test_non_admin_access():
    """Test that non-admin users cannot access credentials."""
    print("\n" + "="*70)
    print("TEST 4: Non-Admin Access Denial")
    print("="*70)
    
    clear_sessions()
    
    # Login as worker (not admin)
    login_response = client.post(
        "/api/auth/login",
        json={"username": "worker", "password": "worker123"}
    )
    
    if login_response.status_code != 200:
        print(f"⚠️  SKIP: Worker login failed, cannot test")
        return
    
    print("✓ Logged in as worker")
    
    # Try to access credentials
    response = client.get("/api/admin/testing/user-credentials")
    
    if response.status_code == 403:
        print("✅ PASS: Returns 403 Forbidden for non-admin user")
    else:
        print(f"❌ FAIL: Expected 403, got {response.status_code}")
        print(f"Response: {response.text}")


def test_password_derivation_logic():
    """Test that password derivation works correctly."""
    print("\n" + "="*70)
    print("TEST 5: Password Derivation Logic")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed")
        return
    
    print("✓ Logged in as software_admin")
    
    # Get credentials
    response = client.get("/api/admin/testing/user-credentials")
    
    if response.status_code != 200:
        print(f"❌ FAIL: Credentials request failed")
        return
    
    data = response.json()
    
    # Check password derivation
    users_with_passwords = [u for u in data if u.get('test_password')]
    users_without_passwords = [u for u in data if not u.get('test_password')]
    
    print(f"Password derivation results:")
    print(f"   Users with test_password: {len(users_with_passwords)}")
    print(f"   Users without test_password: {len(users_without_passwords)}")
    
    if users_with_passwords:
        print(f"\n   Sample derived passwords:")
        for user in users_with_passwords[:3]:
            print(f"     {user.get('username')}: {user.get('test_password')}")
    
    # Verify all passwords are stripped (no TEMP_HASH_ prefix)
    has_temp_prefix = any(
        u.get('test_password', '').startswith('TEMP_HASH_') 
        for u in users_with_passwords
    )
    
    if not has_temp_prefix:
        print(f"\n   ✓ All passwords properly stripped (no TEMP_HASH_ prefix)")
        print(f"✅ PASS: Password derivation working correctly")
    else:
        print(f"\n   ❌ FAIL: Some passwords still have TEMP_HASH_ prefix")


# ==================== MAIN RUNNER ====================

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("MODULE 5.4 - USER CREDENTIALS LISTING TESTS (TEST ONLY)")
    print("="*70)
    print("⚠️  WARNING: This endpoint should be disabled in production")
    
    test_credentials_without_login()
    test_credentials_with_admin()
    test_credentials_verify_specific_user()
    test_non_admin_access()
    test_password_derivation_logic()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
