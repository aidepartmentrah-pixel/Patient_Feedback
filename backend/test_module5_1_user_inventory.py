"""
Test for MODULE 5.1 - User Inventory Router
Tests read-only inventory endpoints for admin users.

Run from backend directory:
    python test_module5_1_user_inventory.py
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

def test_user_inventory_without_login():
    """Test that inventory endpoint requires authentication."""
    print("\n" + "="*70)
    print("TEST 1: User Inventory - Without Login")
    print("="*70)
    
    clear_sessions()
    
    response = client.get("/api/admin/user-inventory")
    
    if response.status_code == 401:
        print("✅ PASS: Returns 401 Unauthorized when not logged in")
    else:
        print(f"❌ FAIL: Expected 401, got {response.status_code}")
        print(f"Response: {response.json()}")


def test_user_inventory_with_admin():
    """Test that inventory endpoint works for software_admin."""
    print("\n" + "="*70)
    print("TEST 2: User Inventory - With Admin Login")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed with {login_response.status_code}")
        return
    
    print("✓ Logged in as software_admin")
    
    # Get inventory
    response = client.get("/api/admin/user-inventory")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ PASS: Returns 200 OK")
        print(f"   Inventory items: {len(data)}")
        
        if len(data) > 0:
            print(f"\n   Sample item:")
            sample = data[0]
            for key, value in sample.items():
                print(f"     {key}: {value}")
        else:
            print("   ⚠️  No inventory items found (database may be empty)")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
        print(f"Response: {response.text}")


def test_missing_users_endpoint():
    """Test endpoint that shows org units without users."""
    print("\n" + "="*70)
    print("TEST 3: Org Units Without Users")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed")
        return
    
    print("✓ Logged in as software_admin")
    
    # Get missing users
    response = client.get("/api/admin/user-inventory/missing")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ PASS: Returns 200 OK")
        print(f"   Org units without users: {len(data)}")
        
        if len(data) > 0:
            print(f"\n   Sample org unit without user:")
            sample = data[0]
            for key, value in sample.items():
                print(f"     {key}: {value}")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
        print(f"Response: {response.text}")


def test_inventory_summary():
    """Test summary statistics endpoint."""
    print("\n" + "="*70)
    print("TEST 4: Inventory Summary")
    print("="*70)
    
    clear_sessions()
    
    # Login as admin
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed")
        return
    
    print("✓ Logged in as software_admin")
    
    # Get summary
    response = client.get("/api/admin/user-inventory/summary")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ PASS: Returns 200 OK")
        print(f"\n   Summary:")
        for key, value in data.items():
            print(f"     {key}: {value}")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
        print(f"Response: {response.text}")


def test_non_admin_access():
    """Test that non-admin users cannot access inventory."""
    print("\n" + "="*70)
    print("TEST 5: Non-Admin Access Denial")
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
    
    # Try to access inventory
    response = client.get("/api/admin/user-inventory")
    
    if response.status_code == 403:
        print("✅ PASS: Returns 403 Forbidden for non-admin user")
    else:
        print(f"❌ FAIL: Expected 403, got {response.status_code}")
        print(f"Response: {response.json()}")


# ==================== MAIN RUNNER ====================

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("MODULE 5.1 - USER INVENTORY TESTS")
    print("="*70)
    
    test_user_inventory_without_login()
    test_user_inventory_with_admin()
    test_missing_users_endpoint()
    test_inventory_summary()
    test_non_admin_access()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
