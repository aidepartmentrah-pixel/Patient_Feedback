"""
Comprehensive Test Suite for Auth Service Layer
================================================
Tests session-based authentication service functions.

Tests:
1. login() - Create session with valid/invalid credentials
2. logout() - Clear session
3. get_current_user_from_session() - Load user from session
4. Session persistence across requests
5. Invalid/expired session handling

Run from backend directory:
    python test_phase2_auth_service.py
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, Request, HTTPException
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from api.services.auth_service import (
    login,
    logout,
    get_current_user_from_session
)


# ==================== TEST APP SETUP ====================

# Create test FastAPI app with session middleware
app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key="test_secret_key_for_testing_only",
    session_cookie="test_session"
)


@app.post("/test/login")
async def test_login_endpoint(request: Request, username: str, password: str):
    """Test endpoint for login."""
    result = login(username, password, request)
    return result


@app.post("/test/logout")
async def test_logout_endpoint(request: Request):
    """Test endpoint for logout."""
    logout(request)
    return {"message": "logged out"}


@app.get("/test/me")
async def test_get_current_user_endpoint(request: Request):
    """Test endpoint for getting current user."""
    user = get_current_user_from_session(request)
    return {
        "user_id": user.user_id,
        "username": user.username,
        "is_active": user.is_active,
        "scopes": [
            {
                "role_code": scope.role_code,
                "org_unit_id": scope.org_unit_id,
                "org_unit_type": scope.org_unit_type
            }
            for scope in user.scopes
        ]
    }


# Create test client
client = TestClient(app)


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
            print("🎉 ALL TESTS PASSED! Auth service is working perfectly!\n")
            return True
        else:
            print(f"⚠️  {self.failed} TEST(S) FAILED! Please review errors above.\n")
            return False


# ==================== TEST CASES ====================

def test_login_valid_credentials(results):
    """Test login with valid credentials."""
    print("\n" + "="*70)
    print("TEST 1: login() - Valid Credentials")
    print("="*70)
    
    test_users = [
        ("software_admin", "admin123"),
        ("worker", "worker123"),
        ("section_admin", "section123"),
    ]
    
    for username, password in test_users:
        try:
            response = client.post(
                "/test/login",
                params={"username": username, "password": password}
            )
            
            if response.status_code != 200:
                results.add_fail(
                    f"Login {username}",
                    f"Expected 200, got {response.status_code}: {response.text}"
                )
            elif "message" not in response.json():
                results.add_fail(
                    f"Login {username}",
                    f"Response missing 'message' key: {response.json()}"
                )
            elif response.json()["message"] != "login successful":
                results.add_fail(
                    f"Login {username}",
                    f"Expected 'login successful', got '{response.json()['message']}'"
                )
            else:
                results.add_pass(f"Login {username} with valid credentials")
        except Exception as e:
            results.add_fail(f"Login {username}", str(e))


def test_login_invalid_credentials(results):
    """Test login with invalid credentials."""
    print("\n" + "="*70)
    print("TEST 2: login() - Invalid Credentials")
    print("="*70)
    
    invalid_tests = [
        ("software_admin", "wrongpassword"),
        ("nonexistent_user", "anypassword"),
        ("worker", ""),
        ("", "password"),
    ]
    
    for username, password in invalid_tests:
        try:
            response = client.post(
                "/test/login",
                params={"username": username, "password": password}
            )
            
            if response.status_code != 401:
                results.add_fail(
                    f"Login with invalid credentials ({username}/{password[:3]}...)",
                    f"Expected 401, got {response.status_code}"
                )
            else:
                results.add_pass(f"Login rejects invalid credentials ({username}/{password[:3] if password else 'empty'}...)")
        except Exception as e:
            results.add_fail(f"Login invalid ({username})", str(e))


def test_session_persistence(results):
    """Test that session persists across requests."""
    print("\n" + "="*70)
    print("TEST 3: Session Persistence")
    print("="*70)
    
    try:
        # Step 1: Login
        login_response = client.post(
            "/test/login",
            params={"username": "section_admin", "password": "section123"}
        )
        
        if login_response.status_code != 200:
            results.add_fail("Session persistence - login", f"Login failed: {login_response.status_code}")
            return
        
        results.add_pass("Step 1: Login successful")
        
        # Step 2: Get current user (should use same session)
        user_response = client.get("/test/me")
        
        if user_response.status_code != 200:
            results.add_fail(
                "Session persistence - get user",
                f"Expected 200, got {user_response.status_code}: {user_response.text}"
            )
            return
        
        user_data = user_response.json()
        
        if user_data.get("username") != "section_admin":
            results.add_fail(
                "Session persistence - verify username",
                f"Expected 'section_admin', got '{user_data.get('username')}'"
            )
        else:
            results.add_pass("Step 2: Get current user from session")
        
        # Step 3: Verify user data structure
        if "user_id" not in user_data:
            results.add_fail("Session data - user_id", "Missing user_id")
        elif "scopes" not in user_data:
            results.add_fail("Session data - scopes", "Missing scopes")
        elif not isinstance(user_data["scopes"], list):
            results.add_fail("Session data - scopes type", f"Scopes not list: {type(user_data['scopes'])}")
        elif len(user_data["scopes"]) == 0:
            results.add_fail("Session data - scopes empty", "User has no scopes")
        else:
            results.add_pass("Step 3: User data structure valid")
            
            # Verify scope structure
            scope = user_data["scopes"][0]
            if all(key in scope for key in ["role_code", "org_unit_id", "org_unit_type"]):
                results.add_pass(f"Step 4: Scope structure valid ({scope['role_code']})")
            else:
                results.add_fail("Scope structure", f"Missing keys in scope: {scope}")
        
    except Exception as e:
        results.add_fail("Session persistence test", str(e))


def test_get_current_user_without_session(results):
    """Test getting current user without logging in."""
    print("\n" + "="*70)
    print("TEST 4: get_current_user_from_session() - No Session")
    print("="*70)
    
    try:
        # Create new client (no session)
        new_client = TestClient(app)
        
        response = new_client.get("/test/me")
        
        if response.status_code != 401:
            results.add_fail(
                "Get user without session",
                f"Expected 401, got {response.status_code}"
            )
        else:
            results.add_pass("Get user without session returns 401")
    except Exception as e:
        results.add_fail("Get user without session", str(e))


def test_logout(results):
    """Test logout functionality."""
    print("\n" + "="*70)
    print("TEST 5: logout()")
    print("="*70)
    
    try:
        # Step 1: Login
        login_response = client.post(
            "/test/login",
            params={"username": "worker", "password": "worker123"}
        )
        
        if login_response.status_code != 200:
            results.add_fail("Logout test - login", f"Login failed: {login_response.status_code}")
            return
        
        results.add_pass("Step 1: Login before logout")
        
        # Step 2: Verify we're logged in
        user_response = client.get("/test/me")
        
        if user_response.status_code != 200:
            results.add_fail("Logout test - verify login", f"User not authenticated: {user_response.status_code}")
            return
        
        results.add_pass("Step 2: Verified authenticated")
        
        # Step 3: Logout
        logout_response = client.post("/test/logout")
        
        if logout_response.status_code != 200:
            results.add_fail("Logout test - logout", f"Logout failed: {logout_response.status_code}")
            return
        
        results.add_pass("Step 3: Logout successful")
        
        # Step 4: Verify session cleared
        user_response_after = client.get("/test/me")
        
        if user_response_after.status_code != 401:
            results.add_fail(
                "Logout test - verify cleared",
                f"Expected 401 after logout, got {user_response_after.status_code}"
            )
        else:
            results.add_pass("Step 4: Session cleared after logout")
    
    except Exception as e:
        results.add_fail("Logout test", str(e))


def test_multiple_users(results):
    """Test multiple users can login independently."""
    print("\n" + "="*70)
    print("TEST 6: Multiple User Sessions")
    print("="*70)
    
    try:
        # Create two separate clients (different sessions)
        client1 = TestClient(app)
        client2 = TestClient(app)
        
        # Login user 1
        response1 = client1.post(
            "/test/login",
            params={"username": "software_admin", "password": "admin123"}
        )
        
        if response1.status_code != 200:
            results.add_fail("Multi-user - user1 login", f"Failed: {response1.status_code}")
            return
        
        results.add_pass("User 1 login (software_admin)")
        
        # Login user 2
        response2 = client2.post(
            "/test/login",
            params={"username": "department_admin", "password": "dept123"}
        )
        
        if response2.status_code != 200:
            results.add_fail("Multi-user - user2 login", f"Failed: {response2.status_code}")
            return
        
        results.add_pass("User 2 login (department_admin)")
        
        # Verify user 1 session
        user1_data = client1.get("/test/me").json()
        if user1_data.get("username") != "software_admin":
            results.add_fail("Multi-user - user1 data", f"Wrong user: {user1_data.get('username')}")
        else:
            results.add_pass("User 1 session maintained (software_admin)")
        
        # Verify user 2 session
        user2_data = client2.get("/test/me").json()
        if user2_data.get("username") != "department_admin":
            results.add_fail("Multi-user - user2 data", f"Wrong user: {user2_data.get('username')}")
        else:
            results.add_pass("User 2 session maintained (department_admin)")
        
    except Exception as e:
        results.add_fail("Multiple users test", str(e))


def test_all_users_login(results):
    """Test all seed users can login."""
    print("\n" + "="*70)
    print("TEST 7: All Seed Users Login")
    print("="*70)
    
    all_users = [
        ("software_admin", "admin123", "SOFTWARE_ADMIN"),
        ("worker", "worker123", "WORKER"),
        ("complaint_supervisor", "sup123", "COMPLAINT_SUPERVISOR"),
        ("section_admin", "section123", "SECTION_ADMIN"),
        ("department_admin", "dept123", "DEPARTMENT_ADMIN"),
        ("administration_admin", "adminis123", "ADMINISTRATION_ADMIN"),
    ]
    
    for username, password, expected_role in all_users:
        try:
            # Create fresh client for each user
            test_client = TestClient(app)
            
            # Login
            login_resp = test_client.post(
                "/test/login",
                params={"username": username, "password": password}
            )
            
            if login_resp.status_code != 200:
                results.add_fail(f"Login {username}", f"Failed: {login_resp.status_code}")
                continue
            
            # Get user data
            user_resp = test_client.get("/test/me")
            
            if user_resp.status_code != 200:
                results.add_fail(f"Get user {username}", f"Failed: {user_resp.status_code}")
                continue
            
            user_data = user_resp.json()
            
            # Verify username
            if user_data.get("username") != username:
                results.add_fail(f"Verify {username}", f"Username mismatch: {user_data.get('username')}")
                continue
            
            # Verify has scopes
            if not user_data.get("scopes"):
                results.add_fail(f"Verify {username} scopes", "No scopes found")
                continue
            
            # Verify expected role exists
            roles = [s["role_code"] for s in user_data["scopes"]]
            if expected_role not in roles:
                results.add_fail(
                    f"Verify {username} role",
                    f"Expected {expected_role}, got {roles}"
                )
            else:
                results.add_pass(f"Login and verify {username} → {expected_role}")
        
        except Exception as e:
            results.add_fail(f"All users test - {username}", str(e))


# ==================== MAIN TEST RUNNER ====================

def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*70)
    print(" " * 12 + "AUTH SERVICE - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("\nTesting session-based authentication service...")
    print("Mode: SESSION-BASED (NO JWT, NO TOKENS)")
    print("Database: IncidentManager")
    print()
    
    results = TestResult()
    
    try:
        # Run all test suites
        test_login_valid_credentials(results)
        test_login_invalid_credentials(results)
        test_session_persistence(results)
        test_get_current_user_without_session(results)
        test_logout(results)
        test_multiple_users(results)
        test_all_users_login(results)
        
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
