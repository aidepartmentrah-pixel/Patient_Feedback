"""
Test for MODULE 5.9 - Recreate Section Admin User
Tests admin endpoint for recreating section admin users with unique usernames.

⚠️ ADMIN TEST TOOL — RECREATE SECTION ADMIN USER

Run from backend directory:
    python test_module5_9_recreate_section_admin.py
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
            "section_name": f"Recreate Test Section {os.urandom(4).hex()}",
            "parent_department_id": 1
        }
    )
    return response


# ==================== TESTS ====================

def test_recreate_without_login():
    """Test that endpoint requires authentication."""
    print("\n" + "="*70)
    print("TEST 1: Recreate Section Admin - Without Login")
    print("="*70)
    
    clear_sessions()
    
    response = client.post("/api/admin/sections/1/recreate-admin")
    
    if response.status_code == 401:
        print("✅ PASS: Returns 401 Unauthorized when not logged in")
    else:
        print(f"❌ FAIL: Expected 401, got {response.status_code}")
        print(f"Response: {response.text}")


def test_recreate_nonexistent_section():
    """Test recreating admin for nonexistent section."""
    print("\n" + "="*70)
    print("TEST 2: Recreate Admin for Nonexistent Section")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed with {login_response.status_code}")
        return
    
    print("✓ Logged in as software_admin")
    
    # Try to recreate admin for nonexistent section
    response = client.post("/api/admin/sections/999999/recreate-admin")
    
    if response.status_code == 404:
        print("✅ PASS: Returns 404 Not Found for nonexistent section")
        data = response.json()
        print(f"   Error message: {data.get('detail')}")
    else:
        print(f"❌ FAIL: Expected 404, got {response.status_code}")
        print(f"Response: {response.text}")


def test_recreate_for_non_section_unit():
    """Test that recreation fails for non-section org units."""
    print("\n" + "="*70)
    print("TEST 3: Block Recreation for Non-Section Units")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed")
        return
    
    print("✓ Logged in as software_admin")
    
    # Try to recreate admin for ID=1 (likely Administration, Type != 324)
    response = client.post("/api/admin/sections/1/recreate-admin")
    
    if response.status_code == 400:
        print("✅ PASS: Returns 400 Bad Request for non-section unit")
        data = response.json()
        print(f"   Error message: {data.get('detail')}")
    elif response.status_code == 404:
        print("⚠️  Section ID 1 not found (expected for some databases)")
    else:
        print(f"⚠️  Unexpected status: {response.status_code}")
        print(f"   Response: {response.text}")


def test_recreate_section_admin_first_time():
    """Test successful recreation of section admin (first time)."""
    print("\n" + "="*70)
    print("TEST 4: Successfully Recreate Section Admin (First Time)")
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
    section_id = created_data.get('section_id')
    original_username = created_data.get('username')
    
    print(f"✓ Created test section ID: {section_id}")
    print(f"✓ Original admin username: {original_username}")
    
    # Recreate section admin (should generate versioned username)
    recreate_response = client.post(f"/api/admin/sections/{section_id}/recreate-admin")
    
    if recreate_response.status_code == 200:
        data = recreate_response.json()
        print(f"✅ PASS: Section admin recreated successfully")
        print(f"   section_id: {data.get('section_id')}")
        print(f"   username: {data.get('username')}")
        print(f"   password: {data.get('password')}")
        
        new_username = data.get('username')
        
        # Verify username is different from original
        if new_username != original_username:
            print(f"   ✓ New username is unique: {new_username}")
            
            # Check if versioned (should have _v2, _v3, etc.)
            if '_v' in new_username:
                print(f"   ✓ Username has version suffix")
            else:
                print(f"   ⚠️  Username does not have version suffix (unexpected)")
        else:
            print(f"   ⚠️  New username same as original (should be unique)")
        
        # Verify password
        if data.get('password') == "Hospital2026!":
            print(f"   ✓ Password is correct")
    else:
        print(f"❌ FAIL: Expected 200, got {recreate_response.status_code}")
        print(f"Response: {recreate_response.text}")


def test_recreate_multiple_times():
    """Test recreating section admin multiple times (version increment)."""
    print("\n" + "="*70)
    print("TEST 5: Recreate Section Admin Multiple Times")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed")
        return
    
    print("✓ Logged in as software_admin")
    
    # Create a test section
    create_response = create_test_section()
    
    if create_response.status_code != 200:
        print(f"⚠️  SKIP: Test section creation failed")
        return
    
    created_data = create_response.json()
    section_id = created_data.get('section_id')
    
    print(f"✓ Created test section ID: {section_id}")
    
    # Recreate admin 3 times
    usernames = []
    
    for i in range(3):
        recreate_response = client.post(f"/api/admin/sections/{section_id}/recreate-admin")
        
        if recreate_response.status_code == 200:
            data = recreate_response.json()
            username = data.get('username')
            usernames.append(username)
            print(f"   ✓ Recreation {i+1}: {username}")
        else:
            print(f"   ❌ Recreation {i+1} failed: {recreate_response.status_code}")
            break
    
    # Verify all usernames are unique
    if len(usernames) == 3:
        unique_usernames = set(usernames)
        if len(unique_usernames) == 3:
            print(f"\n✅ PASS: All recreated usernames are unique")
            print(f"   Usernames: {usernames}")
        else:
            print(f"\n❌ FAIL: Duplicate usernames found")
            print(f"   Usernames: {usernames}")
    else:
        print(f"\n⚠️  Could not create 3 recreations")


def test_recreated_admin_can_login():
    """Test that recreated section admin can login."""
    print("\n" + "="*70)
    print("TEST 6: Recreated Admin Can Login")
    print("="*70)
    
    clear_sessions()
    
    # Login as software admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Software admin login failed")
        return
    
    print("✓ Logged in as software_admin")
    
    # Create a test section
    create_response = create_test_section()
    
    if create_response.status_code != 200:
        print(f"⚠️  SKIP: Test section creation failed")
        return
    
    created_data = create_response.json()
    section_id = created_data.get('section_id')
    
    # Recreate admin
    recreate_response = client.post(f"/api/admin/sections/{section_id}/recreate-admin")
    
    if recreate_response.status_code != 200:
        print(f"⚠️  SKIP: Recreation failed")
        return
    
    recreate_data = recreate_response.json()
    new_username = recreate_data.get('username')
    new_password = recreate_data.get('password')
    
    print(f"✓ Recreated admin: {new_username}")
    
    # Clear session and try to login with new admin
    clear_sessions()
    
    login_response = client.post(
        "/api/auth/login",
        json={
            "username": new_username,
            "password": new_password
        }
    )
    
    if login_response.status_code == 200:
        login_data = login_response.json()
        print(f"✅ PASS: Recreated admin can login successfully")
        print(f"   username: {login_data.get('user', {}).get('username')}")
        
        # Verify role assignment
        scopes = login_data.get('user', {}).get('scopes', [])
        if scopes:
            print(f"   ✓ Has {len(scopes)} scope(s)")
            for scope in scopes:
                print(f"     - Role: {scope.get('role_code')}, OrgUnit: {scope.get('org_unit_id')}")
            
            # Check for SECTION_ADMIN role
            has_section_admin = any(s.get('role_code') == 'SECTION_ADMIN' for s in scopes)
            if has_section_admin:
                print(f"   ✓ SECTION_ADMIN role assigned correctly")
    else:
        print(f"❌ FAIL: Recreated admin login failed with {login_response.status_code}")
        print(f"Response: {login_response.text}")


def test_non_admin_access():
    """Test that non-admin users cannot recreate section admins."""
    print("\n" + "="*70)
    print("TEST 7: Non-Admin Access Denial")
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
    
    # Try to recreate section admin
    response = client.post("/api/admin/sections/1/recreate-admin")
    
    if response.status_code == 403:
        print("✅ PASS: Returns 403 Forbidden for non-admin user")
    else:
        print(f"❌ FAIL: Expected 403, got {response.status_code}")
        print(f"Response: {response.text}")


# ==================== MAIN RUNNER ====================

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("MODULE 5.9 - RECREATE SECTION ADMIN USER TESTS")
    print("="*70)
    print("⚠️  Tests recreating section admin accounts with versioned usernames")
    
    test_recreate_without_login()
    test_recreate_nonexistent_section()
    test_recreate_for_non_section_unit()
    test_recreate_section_admin_first_time()
    test_recreate_multiple_times()
    test_recreated_admin_can_login()
    test_non_admin_access()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
