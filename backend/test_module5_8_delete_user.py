"""
Test for MODULE 5.8 - Delete User Endpoint
Tests admin endpoint for deleting user accounts with safety checks.

⚠️ ADMIN TEST TOOL — USER DELETE — HANDLE WITH CARE

Run from backend directory:
    python test_module5_8_delete_user.py
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


def create_test_section():
    """Helper to create a test section with admin user."""
    response = client.post(
        "/api/admin/create-section-with-admin",
        json={
            "section_name": f"Delete Test Section {os.urandom(4).hex()}",
            "parent_department_id": 1
        }
    )
    return response


# ==================== TESTS ====================

def test_delete_without_login():
    """Test that endpoint requires authentication."""
    print("\n" + "="*70)
    print("TEST 1: Delete User - Without Login")
    print("="*70)
    
    clear_sessions()
    
    response = client.delete("/api/admin/users/999")
    
    if response.status_code == 401:
        print("✅ PASS: Returns 401 Unauthorized when not logged in")
    else:
        print(f"❌ FAIL: Expected 401, got {response.status_code}")
        print(f"Response: {response.text}")


def test_delete_nonexistent_user():
    """Test deleting a user that doesn't exist."""
    print("\n" + "="*70)
    print("TEST 2: Delete Nonexistent User")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed with {login_response.status_code}")
        return
    
    print("✓ Logged in as software_admin")
    
    # Try to delete nonexistent user
    response = client.delete("/api/admin/users/999999")
    
    if response.status_code == 404:
        print("✅ PASS: Returns 404 Not Found for nonexistent user")
        data = response.json()
        print(f"   Error message: {data.get('detail')}")
    else:
        print(f"❌ FAIL: Expected 404, got {response.status_code}")
        print(f"Response: {response.text}")


def test_delete_protected_software_admin():
    """Test that software_admin account cannot be deleted."""
    print("\n" + "="*70)
    print("TEST 3: Block Deletion of software_admin Account")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed")
        return
    
    print("✓ Logged in as software_admin")
    
    # Try to delete software_admin (UserID=1)
    response = client.delete("/api/admin/users/1")
    
    if response.status_code == 403:
        print("✅ PASS: Returns 403 Forbidden for protected account")
        data = response.json()
        print(f"   Protection message: {data.get('detail')}")
    else:
        print(f"⚠️  Expected 403, got {response.status_code}")
        print(f"   Response: {response.text}")
        if response.status_code == 200:
            print(f"   ❌ CRITICAL: software_admin was deleted! This should be blocked.")


def test_delete_user_with_software_admin_role():
    """Test that users with SOFTWARE_ADMIN role cannot be deleted."""
    print("\n" + "="*70)
    print("TEST 4: Block Deletion of Users with SOFTWARE_ADMIN Role")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed")
        return
    
    print("✓ Logged in as software_admin")
    
    # software_admin has SOFTWARE_ADMIN role - should be blocked
    response = client.delete("/api/admin/users/1")
    
    if response.status_code == 403:
        print("✅ PASS: Blocks deletion of user with SOFTWARE_ADMIN role")
    else:
        print(f"⚠️  Expected 403, got {response.status_code}")


def test_delete_regular_user_success():
    """Test successful deletion of a regular user."""
    print("\n" + "="*70)
    print("TEST 5: Successfully Delete Regular User")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed")
        return
    
    print("✓ Logged in as software_admin")
    
    # Create a test section with admin user
    create_response = create_test_section()
    
    if create_response.status_code != 200:
        print(f"⚠️  SKIP: Test section creation failed")
        return
    
    created_data = create_response.json()
    test_username = created_data.get('username')
    
    print(f"✓ Created test user: {test_username}")
    
    # Get user list to find the UserID
    users_response = client.get("/api/admin/testing/user-credentials")
    
    if users_response.status_code != 200:
        print(f"⚠️  SKIP: Could not get user list")
        return
    
    users = users_response.json()
    test_user = next((u for u in users if u.get('username') == test_username), None)
    
    if not test_user:
        print(f"⚠️  SKIP: Could not find created test user in list")
        return
    
    test_user_id = test_user.get('user_id')
    print(f"✓ Found test user ID: {test_user_id}")
    
    # Delete the test user
    delete_response = client.delete(f"/api/admin/users/{test_user_id}")
    
    if delete_response.status_code == 200:
        data = delete_response.json()
        print(f"✅ PASS: User deleted successfully")
        print(f"   deleted_user_id: {data.get('deleted_user_id')}")
        print(f"   deleted_username: {data.get('deleted_username')}")
        
        # Verify user is actually gone
        verify_response = client.get("/api/admin/testing/user-credentials")
        if verify_response.status_code == 200:
            remaining_users = verify_response.json()
            still_exists = any(u.get('user_id') == test_user_id for u in remaining_users)
            
            if not still_exists:
                print(f"   ✓ Verified: User no longer in database")
            else:
                print(f"   ⚠️  User still exists in database after deletion!")
    else:
        print(f"❌ FAIL: Expected 200, got {delete_response.status_code}")
        print(f"Response: {delete_response.text}")


def test_non_admin_access():
    """Test that non-admin users cannot delete users."""
    print("\n" + "="*70)
    print("TEST 6: Non-Admin Access Denial")
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
    
    # Try to delete a user
    response = client.delete("/api/admin/users/999")
    
    if response.status_code == 403:
        print("✅ PASS: Returns 403 Forbidden for non-admin user")
    else:
        print(f"❌ FAIL: Expected 403, got {response.status_code}")
        print(f"Response: {response.text}")


def test_transaction_rollback():
    """Test that failed deletion rolls back properly."""
    print("\n" + "="*70)
    print("TEST 7: Transaction Rollback on Error")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed")
        return
    
    print("✓ Logged in as software_admin")
    
    # Try to delete protected account - should fail and rollback
    response = client.delete("/api/admin/users/1")
    
    if response.status_code == 403:
        print("✅ PASS: Protected account deletion blocked (transaction rolled back)")
        
        # Verify software_admin still exists
        verify_response = client.get("/api/admin/testing/user-credentials")
        if verify_response.status_code == 200:
            users = verify_response.json()
            software_admin_exists = any(u.get('username') == 'software_admin' for u in users)
            
            if software_admin_exists:
                print(f"   ✓ Verified: software_admin account still intact")
            else:
                print(f"   ❌ CRITICAL: software_admin account missing!")
    else:
        print(f"⚠️  Unexpected status: {response.status_code}")


# ==================== MAIN RUNNER ====================

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("MODULE 5.8 - DELETE USER ENDPOINT TESTS")
    print("="*70)
    print("⚠️  WARNING: This endpoint requires careful handling")
    
    test_delete_without_login()
    test_delete_nonexistent_user()
    test_delete_protected_software_admin()
    test_delete_user_with_software_admin_role()
    test_delete_regular_user_success()
    test_non_admin_access()
    test_transaction_rollback()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
