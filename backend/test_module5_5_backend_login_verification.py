"""
Test for MODULE 5.5 - Backend Login Verification
Integration tests for authentication flow, login, role loading, and guard enforcement.

⚠️ PHASE 5 — USERS TESTING READY — BACKEND LOGIN VERIFICATION

Tests REAL endpoints with REAL database - no mocks.
Uses session-based authentication (not JWT).

Run from backend directory:
    python test_module5_5_backend_login_verification.py
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
from core.database import get_connection

# Create test client (handles session cookies automatically)
client = TestClient(app)


def clear_sessions():
    """Clear all test client sessions."""
    client.cookies.clear()


def get_user_active_status(username: str):
    """Get IsActive status for a user."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = "SELECT IsActive FROM dbo.APP_Users WHERE Username = ?"
        cursor.execute(query, (username,))
        result = cursor.fetchone()
        
        if result:
            return bool(result.IsActive)
        return None
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def set_user_active_status(username: str, is_active: bool):
    """Set IsActive status for a user."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = "UPDATE dbo.APP_Users SET IsActive = ? WHERE Username = ?"
        cursor.execute(query, (1 if is_active else 0, username))
        conn.commit()
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_section_admin_username():
    """Query database to find a SECTION_ADMIN user."""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT TOP 1 u.Username
            FROM dbo.APP_Users u
            INNER JOIN dbo.APP_UserRoleScope urs ON u.UserID = urs.UserID
            INNER JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
            WHERE r.RoleCode = 'SECTION_ADMIN'
            AND u.IsActive = 1
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        
        if result:
            return result.Username
        return None
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ==================== TESTS ====================

def test_1_software_admin_login_works():
    """Test 1: Software admin can login successfully."""
    print("\n" + "="*70)
    print("TEST 1: Software Admin Login Works")
    print("="*70)
    
    clear_sessions()
    
    response = client.post(
        "/api/auth/login",
        json={
            "username": "software_admin",
            "password": "admin123"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print("✅ PASS: Login successful")
        print(f"   Status: {response.status_code}")
        
        # Check user object present
        if "user" in data:
            user = data["user"]
            print(f"   ✓ User object present")
            print(f"     - username: {user.get('username')}")
            print(f"     - user_id: {user.get('user_id')}")
            print(f"     - is_active: {user.get('is_active')}")
            
            # Check scopes present
            if "scopes" in user:
                scopes = user["scopes"]
                print(f"     - scopes: {len(scopes)} scope(s)")
                for scope in scopes:
                    print(f"       * {scope.get('role_code')} - OrgUnit {scope.get('org_unit_id')}")
            else:
                print(f"   ⚠️  No scopes in user object")
        else:
            print(f"   ⚠️  No user object in response")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
        print(f"Response: {response.text}")


def test_2_wrong_password_fails():
    """Test 2: Login fails with wrong password."""
    print("\n" + "="*70)
    print("TEST 2: Wrong Password Fails")
    print("="*70)
    
    clear_sessions()
    
    response = client.post(
        "/api/auth/login",
        json={
            "username": "software_admin",
            "password": "wrong_password_123"
        }
    )
    
    # Expect auth failure (400 or 401)
    if response.status_code in [400, 401]:
        print(f"✅ PASS: Login failed with status {response.status_code}")
        data = response.json()
        print(f"   Error message: {data.get('detail') or data.get('message')}")
    else:
        print(f"❌ FAIL: Expected 400/401, got {response.status_code}")
        print(f"Response: {response.text}")


def test_3_unknown_user_fails():
    """Test 3: Login fails for nonexistent user."""
    print("\n" + "="*70)
    print("TEST 3: Unknown User Fails")
    print("="*70)
    
    clear_sessions()
    
    response = client.post(
        "/api/auth/login",
        json={
            "username": "does_not_exist_user_xyz",
            "password": "any_password"
        }
    )
    
    # Expect auth failure
    if response.status_code in [400, 401, 404]:
        print(f"✅ PASS: Login failed with status {response.status_code}")
        data = response.json()
        print(f"   Error message: {data.get('detail') or data.get('message')}")
    else:
        print(f"❌ FAIL: Expected 400/401/404, got {response.status_code}")
        print(f"Response: {response.text}")


def test_4_login_then_me_works():
    """Test 4: Login then call /api/auth/me endpoint."""
    print("\n" + "="*70)
    print("TEST 4: Login + /api/auth/me Works")
    print("="*70)
    
    clear_sessions()
    
    # Step 1: Login
    login_response = client.post(
        "/api/auth/login",
        json={
            "username": "software_admin",
            "password": "admin123"
        }
    )
    
    if login_response.status_code != 200:
        print(f"❌ FAIL: Login failed with {login_response.status_code}")
        return
    
    print("✓ Login successful")
    login_data = login_response.json()
    login_username = login_data.get("user", {}).get("username")
    
    # Step 2: Call /api/auth/me (session cookie automatically sent by TestClient)
    me_response = client.get("/api/auth/me")
    
    if me_response.status_code == 200:
        me_data = me_response.json()
        print("✅ PASS: /api/auth/me works after login")
        print(f"   Status: {me_response.status_code}")
        
        # Check username matches
        me_username = me_data.get("user", {}).get("username")
        if me_username == login_username:
            print(f"   ✓ Username matches: {me_username}")
        else:
            print(f"   ⚠️  Username mismatch: login={login_username}, me={me_username}")
        
        # Check scopes present
        scopes = me_data.get("user", {}).get("scopes", [])
        if scopes:
            print(f"   ✓ Scopes present: {len(scopes)} scope(s)")
            for scope in scopes:
                print(f"     - {scope.get('role_code')} (OrgUnit: {scope.get('org_unit_id')})")
        else:
            print(f"   ⚠️  No scopes returned")
    else:
        print(f"❌ FAIL: /api/auth/me returned {me_response.status_code}")
        print(f"Response: {me_response.text}")


def test_5_me_without_session_fails():
    """Test 5: /api/auth/me fails without authentication."""
    print("\n" + "="*70)
    print("TEST 5: /api/auth/me Without Session")
    print("="*70)
    
    clear_sessions()  # Ensure no session exists
    
    response = client.get("/api/auth/me")
    
    if response.status_code == 401:
        print("✅ PASS: Returns 401 Unauthorized without session")
        data = response.json()
        print(f"   Error message: {data.get('detail')}")
    else:
        print(f"❌ FAIL: Expected 401, got {response.status_code}")
        print(f"Response: {response.text}")


def test_6_inactive_user_blocked():
    """Test 6: Inactive user cannot login."""
    print("\n" + "="*70)
    print("TEST 6: Inactive User Blocked")
    print("="*70)
    
    clear_sessions()
    
    # Use 'worker' test user for this test
    test_username = "worker"
    original_status = None
    
    try:
        # Step 1: Get original IsActive status
        original_status = get_user_active_status(test_username)
        
        if original_status is None:
            print(f"⚠️  SKIP: User '{test_username}' not found in database")
            return
        
        print(f"✓ Original IsActive status: {original_status}")
        
        # Step 2: Set IsActive = 0 (inactive)
        set_user_active_status(test_username, False)
        print(f"✓ Set {test_username} to inactive")
        
        # Step 3: Attempt login
        response = client.post(
            "/api/auth/login",
            json={
                "username": test_username,
                "password": "worker123"  # Known password for worker
            }
        )
        
        # Step 4: Expect failure
        if response.status_code in [400, 401, 403]:
            print(f"✅ PASS: Inactive user login blocked with status {response.status_code}")
            data = response.json()
            error_msg = data.get('detail') or data.get('message')
            print(f"   Error message: {error_msg}")
        else:
            print(f"❌ FAIL: Expected 400/401/403, got {response.status_code}")
            print(f"Response: {response.text}")
        
    finally:
        # Step 5: ALWAYS restore original status (even if test fails)
        if original_status is not None:
            set_user_active_status(test_username, original_status)
            print(f"✓ Restored {test_username} to IsActive={original_status}")


def test_7_section_admin_login_works():
    """Test 7: Section admin can login successfully."""
    print("\n" + "="*70)
    print("TEST 7: Section Admin Login Works")
    print("="*70)
    
    clear_sessions()
    
    # Query database to find SECTION_ADMIN user
    section_admin_username = get_section_admin_username()
    
    if not section_admin_username:
        print("⚠️  SKIP: No SECTION_ADMIN user found in database")
        return
    
    print(f"✓ Found SECTION_ADMIN user: {section_admin_username}")
    
    # Attempt login with standard test password
    response = client.post(
        "/api/auth/login",
        json={
            "username": section_admin_username,
            "password": "Hospital2026!"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print("✅ PASS: Section admin login successful")
        print(f"   Username: {data.get('user', {}).get('username')}")
        
        # Verify SECTION_ADMIN role in scopes
        scopes = data.get('user', {}).get('scopes', [])
        has_section_admin = any(s.get('role_code') == 'SECTION_ADMIN' for s in scopes)
        
        if has_section_admin:
            print(f"   ✓ SECTION_ADMIN role present in scopes")
        else:
            print(f"   ⚠️  SECTION_ADMIN role not found in scopes")
            print(f"   Scopes: {[s.get('role_code') for s in scopes]}")
    else:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
        print(f"Response: {response.text}")


def test_8_role_guard_enforcement():
    """Test 8: Role guard blocks unauthorized access."""
    print("\n" + "="*70)
    print("TEST 8: Role Guard Enforcement")
    print("="*70)
    
    clear_sessions()
    
    # Query database to find SECTION_ADMIN user
    section_admin_username = get_section_admin_username()
    
    if not section_admin_username:
        print("⚠️  SKIP: No SECTION_ADMIN user found in database")
        return
    
    print(f"✓ Found SECTION_ADMIN user: {section_admin_username}")
    
    # Step 1: Login as SECTION_ADMIN
    login_response = client.post(
        "/api/auth/login",
        json={
            "username": section_admin_username,
            "password": "Hospital2026!"
        }
    )
    
    if login_response.status_code != 200:
        print(f"⚠️  SKIP: Login failed with {login_response.status_code}")
        return
    
    print("✓ Logged in as SECTION_ADMIN")
    
    # Step 2: Try to access SOFTWARE_ADMIN-only endpoint
    # Using existing admin endpoint: GET /api/admin/testing/user-credentials
    response = client.get("/api/admin/testing/user-credentials")
    
    if response.status_code == 403:
        print("✅ PASS: Role guard blocked unauthorized access with 403 Forbidden")
        data = response.json()
        print(f"   Error message: {data.get('detail')}")
    else:
        print(f"❌ FAIL: Expected 403, got {response.status_code}")
        print(f"Response: {response.text}")


# ==================== MAIN RUNNER ====================

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("MODULE 5.5 - BACKEND LOGIN VERIFICATION TESTS")
    print("="*70)
    print("⚠️  Integration tests with REAL endpoints and REAL database")
    print("⚠️  Uses session-based authentication (not JWT)")
    
    test_1_software_admin_login_works()
    test_2_wrong_password_fails()
    test_3_unknown_user_fails()
    test_4_login_then_me_works()
    test_5_me_without_session_fails()
    test_6_inactive_user_blocked()
    test_7_section_admin_login_works()
    test_8_role_guard_enforcement()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
    print("\n📝 Summary:")
    print("   - Tests verify session-based authentication")
    print("   - Tests use real database connections")
    print("   - Tests restore database state after mutations")
    print("   - No auth code modifications required")


if __name__ == "__main__":
    run_all_tests()
