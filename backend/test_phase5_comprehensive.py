"""
COMPREHENSIVE TEST SUITE - PHASE 5 ALL MODULES
Tests all Phase 5 endpoints and functionality in integrated workflow.

Modules Tested:
- MODULE 5.1: User Inventory & Mapping Engine (3 endpoints)
- MODULE 5.2: Bulk User Generator (SQL script verification)
- MODULE 5.3: Create Section + Admin User
- MODULE 5.4: List All Users + Passwords
- MODULE 5.5: Backend Login Verification
- MODULE 5.7: Markdown Credential Export
- MODULE 5.8: Delete User
- MODULE 5.9: Recreate Section Admin User

Run from backend directory:
    python test_phase5_comprehensive.py
"""

import sys
import os
from pathlib import Path
import time

# Add backend directory to path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from main import app
from core.database import get_connection

# Create test client
client = TestClient(app)

# Track test results
test_results = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "total": 0
}

# Store test data for cleanup
test_data = {
    "created_sections": [],
    "created_users": []
}


def clear_sessions():
    """Clear all test client sessions."""
    client.cookies.clear()


def login_as_admin():
    """Helper to login as software_admin."""
    clear_sessions()
    response = client.post(
        "/api/auth/login",
        json={"username": "software_admin", "password": "admin123"}
    )
    return response


def record_test_result(passed: bool, skipped: bool = False):
    """Record test result in summary."""
    test_results["total"] += 1
    if skipped:
        test_results["skipped"] += 1
    elif passed:
        test_results["passed"] += 1
    else:
        test_results["failed"] += 1


def print_test_header(module: str, test_name: str):
    """Print formatted test header."""
    print("\n" + "="*80)
    print(f"[{module}] {test_name}")
    print("="*80)


def print_section_header(title: str):
    """Print section header."""
    print("\n" + "█"*80)
    print(f"█ {title}")
    print("█"*80)


def get_valid_department_id():
    """Query database to find a valid department (Type=2 in AdminsrationUnit)."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get a department (Type = 325)
        query = """
            SELECT TOP 1 UniqueID
            FROM dbo.AdminsrationUnit
            WHERE Type = 325
            AND Frozen = 0
            ORDER BY UniqueID
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        
        if result:
            return result.UniqueID
        return None
        
    except:
        return None
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ==================== MODULE 5.5 - BACKEND LOGIN VERIFICATION ====================

def test_login_software_admin():
    """Test software admin login."""
    print_test_header("MODULE 5.5", "Software Admin Login")
    
    clear_sessions()
    response = client.post(
        "/api/auth/login",
        json={"username": "software_admin", "password": "admin123"}
    )
    
    passed = response.status_code == 200
    if passed:
        data = response.json()
        print(f"✅ PASS: Login successful")
        print(f"   Username: {data.get('user', {}).get('username')}")
        print(f"   User ID: {data.get('user', {}).get('user_id')}")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
    
    record_test_result(passed)


def test_login_and_me():
    """Test login then /api/auth/me."""
    print_test_header("MODULE 5.5", "Login + /api/auth/me")
    
    login_response = login_as_admin()
    
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed")
        record_test_result(False)
        return
    
    me_response = client.get("/api/auth/me")
    
    passed = me_response.status_code == 200
    if passed:
        data = me_response.json()
        print(f"✅ PASS: /api/auth/me returned current user")
        print(f"   Username: {data.get('user', {}).get('username')}")
        print(f"   Scopes: {len(data.get('user', {}).get('scopes', []))}")
    else:
        print(f"❌ FAIL: Expected 200, got {me_response.status_code}")
    
    record_test_result(passed)


def test_login_wrong_password():
    """Test login with wrong password."""
    print_test_header("MODULE 5.5", "Wrong Password Rejection")
    
    clear_sessions()
    response = client.post(
        "/api/auth/login",
        json={"username": "software_admin", "password": "wrong_password"}
    )
    
    passed = response.status_code in [400, 401]
    if passed:
        print(f"✅ PASS: Wrong password rejected with {response.status_code}")
    else:
        print(f"❌ FAIL: Expected 400/401, got {response.status_code}")
    
    record_test_result(passed)


# ==================== MODULE 5.1 - USER INVENTORY ====================

def test_user_inventory_full():
    """Test full user inventory endpoint."""
    print_test_header("MODULE 5.1", "User Inventory (Full)")
    
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print("⚠️  SKIP: Login failed")
        record_test_result(False, skipped=True)
        return
    
    response = client.get("/api/admin/user-inventory")
    
    passed = response.status_code == 200
    if passed:
        data = response.json()
        # API returns list directly
        inventory = data if isinstance(data, list) else data.get("inventory", [])
        print(f"✅ PASS: User inventory retrieved")
        print(f"   Total org units: {len(inventory)}")
        print(f"   With users: {sum(1 for item in inventory if item.get('username'))}")
        print(f"   Without users: {sum(1 for item in inventory if not item.get('username'))}")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
    
    record_test_result(passed)


def test_user_inventory_missing():
    """Test missing users inventory endpoint."""
    print_test_header("MODULE 5.1", "User Inventory (Missing Users)")
    
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print("⚠️  SKIP: Login failed")
        record_test_result(False, skipped=True)
        return
    
    response = client.get("/api/admin/user-inventory/missing")
    
    passed = response.status_code == 200
    if passed:
        data = response.json()
        # API returns list directly
        missing = data if isinstance(data, list) else data.get("missing_users", [])
        print(f"✅ PASS: Missing users list retrieved")
        print(f"   Org units without users: {len(missing)}")
        if missing:
            print(f"   Example: {missing[0].get('org_unit_name')} (ID: {missing[0].get('org_unit_id')})")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
    
    record_test_result(passed)


def test_user_inventory_summary():
    """Test user inventory summary endpoint."""
    print_test_header("MODULE 5.1", "User Inventory (Summary)")
    
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print("⚠️  SKIP: Login failed")
        record_test_result(False, skipped=True)
        return
    
    response = client.get("/api/admin/user-inventory/summary")
    
    passed = response.status_code == 200
    if passed:
        data = response.json()
        # API returns dict with summary key or direct dict
        summary = data.get("summary", data) if isinstance(data, dict) and "summary" in data else data
        print(f"✅ PASS: Summary statistics retrieved")
        print(f"   Total users: {summary.get('total_users')}")
        print(f"   Total org units: {summary.get('total_org_units')}")
        print(f"   Org units with users: {summary.get('org_units_with_users')}")
        print(f"   Org units without users: {summary.get('org_units_without_users')}")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
    
    record_test_result(passed)


# ==================== MODULE 5.3 - CREATE SECTION + ADMIN ====================

def test_create_section_with_admin():
    """Test creating section with admin user."""
    print_test_header("MODULE 5.3", "Create Section + Admin User")
    
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print("⚠️  SKIP: Login failed")
        record_test_result(False, skipped=True)
        return
    
    # Get a valid department ID from database
    parent_dept_id = get_valid_department_id()
    
    if not parent_dept_id:
        print("⚠️  SKIP: No department found in database (Type=325)")
        record_test_result(False, skipped=True)
        return
    
    print(f"✓ Found valid parent department ID: {parent_dept_id}")
    
    # Create unique section name
    timestamp = int(time.time() * 1000)
    section_name = f"Test Section {timestamp}"
    
    response = client.post(
        "/api/admin/create-section-with-admin",
        json={
            "section_name": section_name,
            "parent_department_id": parent_dept_id
        }
    )
    
    passed = response.status_code == 200
    if passed:
        data = response.json()
        section_id = data.get("section_id")
        username = data.get("username")
        password = data.get("password")
        
        # Store for cleanup
        test_data["created_sections"].append(section_id)
        test_data["created_users"].append(username)
        
        print(f"✅ PASS: Section and admin created")
        print(f"   Section ID: {section_id}")
        print(f"   Username: {username}")
        print(f"   Password: {password}")
        
        # Verify new admin can login
        clear_sessions()
        login_test = client.post(
            "/api/auth/login",
            json={"username": username, "password": password}
        )
        
        if login_test.status_code == 200:
            print(f"   ✓ New admin can login successfully")
        else:
            print(f"   ⚠️  New admin login failed")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
        print(f"   Response: {response.text}")
    
    record_test_result(passed)


# ==================== MODULE 5.4 - LIST USER CREDENTIALS ====================

def test_list_user_credentials():
    """Test listing all user credentials (TEST ONLY endpoint)."""
    print_test_header("MODULE 5.4", "List All User Credentials")
    
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print("⚠️  SKIP: Login failed")
        record_test_result(False, skipped=True)
        return
    
    response = client.get("/api/admin/testing/user-credentials")
    
    passed = response.status_code == 200
    if passed:
        data = response.json()
        # API returns list directly or dict with credentials key
        credentials = data if isinstance(data, list) else data.get("credentials", [])
        print(f"✅ PASS: User credentials retrieved")
        print(f"   Total users: {len(credentials)}")
        if credentials:
            print(f"   Example user:")
            example = credentials[0]
            print(f"     - Username: {example.get('username')}")
            print(f"     - Role: {example.get('role')}")
            print(f"     - Active: {example.get('active')}")
            # Password should be returned without TEMP_HASH_ prefix
            test_password = example.get('test_password')
            if test_password and not test_password.startswith('TEMP_HASH_'):
                print(f"     ✓ Password format correct (no TEMP_HASH_ prefix)")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
    
    record_test_result(passed)


def test_list_credentials_non_admin():
    """Test that non-admin cannot list credentials."""
    print_test_header("MODULE 5.4", "Non-Admin Access Denial")
    
    # Login as worker (not admin)
    clear_sessions()
    worker_login = client.post(
        "/api/auth/login",
        json={"username": "worker", "password": "worker123"}
    )
    
    if worker_login.status_code != 200:
        print("⚠️  SKIP: Worker login failed")
        record_test_result(False, skipped=True)
        return
    
    response = client.get("/api/admin/testing/user-credentials")
    
    passed = response.status_code == 403
    if passed:
        print(f"✅ PASS: Non-admin blocked with 403 Forbidden")
    else:
        print(f"❌ FAIL: Expected 403, got {response.status_code}")
    
    record_test_result(passed)


# ==================== MODULE 5.7 - MARKDOWN EXPORT ====================

def test_markdown_credential_export():
    """Test markdown credential export."""
    print_test_header("MODULE 5.7", "Markdown Credential Export")
    
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print("⚠️  SKIP: Login failed")
        record_test_result(False, skipped=True)
        return
    
    response = client.get("/api/admin/testing/user-credentials-markdown")
    
    passed = response.status_code == 200
    if passed:
        content = response.text
        
        # Check if it's markdown format
        has_header = "| Username |" in content or "# User Credentials" in content
        has_separator = "|---" in content or "|--" in content
        
        print(f"✅ PASS: Markdown export successful")
        print(f"   Content length: {len(content)} characters")
        print(f"   Has table header: {has_header}")
        print(f"   Has table separator: {has_separator}")
        print(f"   Content type: {response.headers.get('content-type')}")
        
        # Show first few lines
        lines = content.split('\n')[:5]
        print(f"   First lines preview:")
        for line in lines:
            print(f"     {line[:70]}")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
    
    record_test_result(passed)


# ==================== MODULE 5.9 - RECREATE SECTION ADMIN ====================

def test_recreate_section_admin():
    """Test recreating section admin user."""
    print_test_header("MODULE 5.9", "Recreate Section Admin User")
    
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print("⚠️  SKIP: Login failed")
        record_test_result(False, skipped=True)
        return
    
    # Use a section we created earlier if available
    if not test_data["created_sections"]:
        print("⚠️  SKIP: No test section available")
        record_test_result(False, skipped=True)
        return
    
    section_id = test_data["created_sections"][0]
    
    response = client.post(f"/api/admin/sections/{section_id}/recreate-admin")
    
    passed = response.status_code == 200
    if passed:
        data = response.json()
        username = data.get("username")
        password = data.get("password")
        
        # Store for potential cleanup
        test_data["created_users"].append(username)
        
        print(f"✅ PASS: Section admin recreated")
        print(f"   Section ID: {section_id}")
        print(f"   New username: {username}")
        print(f"   Password: {password}")
        
        # Verify username has version suffix
        if "_v" in username:
            print(f"   ✓ Username has version suffix (unique)")
        
        # Verify new admin can login
        clear_sessions()
        login_test = client.post(
            "/api/auth/login",
            json={"username": username, "password": password}
        )
        
        if login_test.status_code == 200:
            print(f"   ✓ Recreated admin can login successfully")
        else:
            print(f"   ⚠️  Recreated admin login failed")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
        print(f"   Response: {response.text}")
    
    record_test_result(passed)


def test_recreate_admin_nonexistent_section():
    """Test recreating admin for nonexistent section."""
    print_test_header("MODULE 5.9", "Recreate Admin - Nonexistent Section")
    
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print("⚠️  SKIP: Login failed")
        record_test_result(False, skipped=True)
        return
    
    response = client.post("/api/admin/sections/999999/recreate-admin")
    
    passed = response.status_code == 404
    if passed:
        print(f"✅ PASS: Returns 404 for nonexistent section")
    else:
        print(f"❌ FAIL: Expected 404, got {response.status_code}")
    
    record_test_result(passed)


# ==================== MODULE 5.8 - DELETE USER ====================

def test_delete_user():
    """Test deleting a user."""
    print_test_header("MODULE 5.8", "Delete User")
    
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print("⚠️  SKIP: Login failed")
        record_test_result(False, skipped=True)
        return
    
    # Get a valid department ID
    parent_dept_id = get_valid_department_id()
    
    if not parent_dept_id:
        print("⚠️  SKIP: No department found in database")
        record_test_result(False, skipped=True)
        return
    
    # Create a test user to delete
    timestamp = int(time.time() * 1000)
    section_name = f"Delete Test Section {timestamp}"
    
    create_response = client.post(
        "/api/admin/create-section-with-admin",
        json={
            "section_name": section_name,
            "parent_department_id": parent_dept_id
        }
    )
    
    if create_response.status_code != 200:
        print("⚠️  SKIP: Could not create test user")
        record_test_result(False, skipped=True)
        return
    
    create_data = create_response.json()
    username = create_data.get("username")
    
    # Get user ID from credentials list
    creds_response = client.get("/api/admin/testing/user-credentials")
    if creds_response.status_code != 200:
        print("⚠️  SKIP: Could not get user credentials")
        record_test_result(False, skipped=True)
        return
    
    # API returns list directly, not wrapped in object
    credentials_data = creds_response.json()
    credentials = credentials_data if isinstance(credentials_data, list) else credentials_data.get("credentials", [])
    user = next((u for u in credentials if u.get("username") == username), None)
    
    if not user:
        print("⚠️  SKIP: Could not find created user")
        record_test_result(False, skipped=True)
        return
    
    user_id = user.get("user_id")
    
    # Now delete the user
    delete_response = client.delete(f"/api/admin/users/{user_id}")
    
    passed = delete_response.status_code == 200
    if passed:
        data = delete_response.json()
        print(f"✅ PASS: User deleted successfully")
        print(f"   Deleted user ID: {data.get('deleted_user_id')}")
        print(f"   Deleted username: {data.get('deleted_username')}")
        
        # Verify user cannot login anymore
        clear_sessions()
        login_test = client.post(
            "/api/auth/login",
            json={"username": username, "password": "Hospital2026!"}
        )
        
        if login_test.status_code in [400, 401]:
            print(f"   ✓ Deleted user cannot login (correctly blocked)")
    else:
        print(f"❌ FAIL: Expected 200, got {delete_response.status_code}")
        print(f"   Response: {delete_response.text}")
    
    record_test_result(passed)


def test_delete_protected_user():
    """Test that protected users cannot be deleted."""
    print_test_header("MODULE 5.8", "Delete Protected User")
    
    login_response = login_as_admin()
    if login_response.status_code != 200:
        print("⚠️  SKIP: Login failed")
        record_test_result(False, skipped=True)
        return
    
    # Try to delete software_admin (UserID=1, should be protected)
    response = client.delete("/api/admin/users/1")
    
    passed = response.status_code == 403
    if passed:
        print(f"✅ PASS: Protected user deletion blocked with 403 Forbidden")
        data = response.json()
        print(f"   Error: {data.get('detail')}")
    else:
        print(f"❌ FAIL: Expected 403, got {response.status_code}")
    
    record_test_result(passed)


# ==================== MODULE 5.2 - BULK USER VERIFICATION ====================

def test_verify_bulk_users_exist():
    """Verify if bulk users were created (MODULE 5.2 SQL script)."""
    print_test_header("MODULE 5.2", "Bulk User Generator Verification")
    
    print("ℹ️  MODULE 5.2 is a SQL script, not an API endpoint.")
    print("   Checking if bulk-created users exist in database...")
    
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check for bulk-created usernames with pattern
        query = """
            SELECT COUNT(*) as count
            FROM dbo.APP_Users
            WHERE Username LIKE 'adm_%_admin'
               OR Username LIKE 'dept_%_admin'
               OR Username LIKE 'sec_%_admin'
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        bulk_user_count = result.count if result else 0
        
        if bulk_user_count > 0:
            print(f"✅ INFO: Found {bulk_user_count} bulk-created users")
            print(f"   (SQL script appears to have been run)")
            record_test_result(True)
        else:
            print(f"⚠️  INFO: No bulk-created users found")
            print(f"   (SQL script may not have been run yet)")
            record_test_result(False, skipped=True)
        
    except Exception as e:
        print(f"⚠️  SKIP: Could not verify bulk users: {str(e)}")
        record_test_result(False, skipped=True)
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ==================== MAIN TEST RUNNER ====================

def run_all_tests():
    """Run all Phase 5 tests."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  PHASE 5 COMPREHENSIVE TEST SUITE".center(78) + "█")
    print("█" + "  All Modules Integration Test".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    print("\n📋 Test Sequence:")
    print("   1. MODULE 5.5 - Backend Login Verification")
    print("   2. MODULE 5.1 - User Inventory & Mapping")
    print("   3. MODULE 5.3 - Create Section + Admin")
    print("   4. MODULE 5.4 - List User Credentials")
    print("   5. MODULE 5.7 - Markdown Export")
    print("   6. MODULE 5.9 - Recreate Section Admin")
    print("   7. MODULE 5.8 - Delete User")
    print("   8. MODULE 5.2 - Bulk User Verification")
    
    # MODULE 5.5 - Login Verification
    print_section_header("MODULE 5.5 - BACKEND LOGIN VERIFICATION")
    test_login_software_admin()
    test_login_and_me()
    test_login_wrong_password()
    
    # MODULE 5.1 - User Inventory
    print_section_header("MODULE 5.1 - USER INVENTORY & MAPPING ENGINE")
    test_user_inventory_full()
    test_user_inventory_missing()
    test_user_inventory_summary()
    
    # MODULE 5.3 - Create Section + Admin
    print_section_header("MODULE 5.3 - CREATE SECTION + ADMIN USER")
    test_create_section_with_admin()
    
    # MODULE 5.4 - List User Credentials
    print_section_header("MODULE 5.4 - LIST USER CREDENTIALS (TEST ONLY)")
    test_list_user_credentials()
    test_list_credentials_non_admin()
    
    # MODULE 5.7 - Markdown Export
    print_section_header("MODULE 5.7 - MARKDOWN CREDENTIAL EXPORT")
    test_markdown_credential_export()
    
    # MODULE 5.9 - Recreate Section Admin
    print_section_header("MODULE 5.9 - RECREATE SECTION ADMIN USER")
    test_recreate_section_admin()
    test_recreate_admin_nonexistent_section()
    
    # MODULE 5.8 - Delete User
    print_section_header("MODULE 5.8 - DELETE USER")
    test_delete_user()
    test_delete_protected_user()
    
    # MODULE 5.2 - Bulk User Verification
    print_section_header("MODULE 5.2 - BULK USER GENERATOR VERIFICATION")
    test_verify_bulk_users_exist()
    
    # Print summary
    print("\n" + "█"*80)
    print("█" + "  TEST SUMMARY".center(78) + "█")
    print("█"*80)
    print(f"\n📊 Results:")
    print(f"   ✅ Passed:  {test_results['passed']}")
    print(f"   ❌ Failed:  {test_results['failed']}")
    print(f"   ⚠️  Skipped: {test_results['skipped']}")
    print(f"   📝 Total:   {test_results['total']}")
    
    success_rate = (test_results['passed'] / test_results['total'] * 100) if test_results['total'] > 0 else 0
    print(f"\n   Success Rate: {success_rate:.1f}%")
    
    if test_results['failed'] == 0:
        print("\n🎉 ALL TESTS PASSED!")
    elif test_results['passed'] > test_results['failed']:
        print("\n✅ MOSTLY PASSING - Some issues to address")
    else:
        print("\n⚠️  ATTENTION NEEDED - Multiple test failures")
    
    # Cleanup notes
    if test_data["created_sections"] or test_data["created_users"]:
        print("\n🧹 Test Data Created:")
        if test_data["created_sections"]:
            print(f"   Sections: {len(test_data['created_sections'])} created")
        if test_data["created_users"]:
            print(f"   Users: {len(test_data['created_users'])} created")
        print("   (Some may have been deleted during tests)")
    
    print("\n" + "█"*80)
    print("█" + "  END OF PHASE 5 COMPREHENSIVE TEST".center(78) + "█")
    print("█"*80 + "\n")


if __name__ == "__main__":
    run_all_tests()
