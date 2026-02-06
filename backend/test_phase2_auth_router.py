"""
Phase 2 RBAC: Auth Router Integration Tests
Tests all auth endpoints with comprehensive coverage.
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


# ==================== TEST HELPERS ====================

def clear_all_sessions():
    """Clear all test client sessions."""
    client.cookies.clear()


def login_user(username: str, password: str):
    """Helper to login a user and return response."""
    return client.post(
        "/api/auth/login",
        json={"username": username, "password": password}
    )


def logout_user():
    """Helper to logout current user."""
    return client.post("/api/auth/logout")


def get_current_user():
    """Helper to get current user profile."""
    return client.get("/api/auth/me")


# ==================== LOGIN ENDPOINT TESTS ====================

def test_login_success_software_admin():
    """Test successful login with software_admin."""
    clear_all_sessions()
    
    response = login_user("software_admin", "admin123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["message"] == "Login successful"
    assert data["user"]["username"] == "software_admin"
    assert data["user"]["user_id"] == 1
    assert data["user"]["is_active"] is True
    assert len(data["user"]["scopes"]) == 1
    assert data["user"]["scopes"][0]["role_code"] == "SOFTWARE_ADMIN"
    assert data["user"]["scopes"][0]["org_unit_id"] == 0
    assert data["user"]["scopes"][0]["org_unit_type"] == "ADMINISTRATION"
    print("✓ Login success: software_admin")


def test_login_success_worker():
    """Test successful login with worker (department scoped)."""
    clear_all_sessions()
    
    response = login_user("worker", "worker123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["user"]["username"] == "worker"
    assert data["user"]["user_id"] == 2
    assert len(data["user"]["scopes"]) == 1
    assert data["user"]["scopes"][0]["role_code"] == "WORKER"
    assert data["user"]["scopes"][0]["org_unit_id"] == 10
    assert data["user"]["scopes"][0]["org_unit_type"] == "COMPLAINT"
    print("✓ Login success: worker with department scope")


def test_login_success_complaint_supervisor():
    """Test successful login with complaint_supervisor."""
    clear_all_sessions()
    
    response = login_user("complaint_supervisor", "sup123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["user"]["username"] == "complaint_supervisor"
    assert data["user"]["user_id"] == 3
    assert len(data["user"]["scopes"]) == 1
    assert data["user"]["scopes"][0]["role_code"] == "COMPLAINT_SUPERVISOR"
    print("✓ Login success: complaint_supervisor")


def test_login_success_section_admin():
    """Test successful login with section_admin."""
    clear_all_sessions()
    
    response = login_user("section_admin", "section123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["user"]["username"] == "section_admin"
    assert data["user"]["user_id"] == 4
    assert len(data["user"]["scopes"]) == 1
    assert data["user"]["scopes"][0]["role_code"] == "SECTION_ADMIN"
    assert data["user"]["scopes"][0]["org_unit_id"] == 10
    assert data["user"]["scopes"][0]["org_unit_type"] == "SECTION"
    print("✓ Login success: section_admin with section scope")


def test_login_success_department_admin():
    """Test successful login with department_admin."""
    clear_all_sessions()
    
    response = login_user("department_admin", "dept123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["user"]["username"] == "department_admin"
    assert data["user"]["user_id"] == 5
    assert len(data["user"]["scopes"]) == 1
    assert data["user"]["scopes"][0]["role_code"] == "DEPARTMENT_ADMIN"
    assert data["user"]["scopes"][0]["org_unit_id"] == 5
    assert data["user"]["scopes"][0]["org_unit_type"] == "DEPARTMENT"
    print("✓ Login success: department_admin with department scope")


def test_login_success_administration_admin():
    """Test successful login with administration_admin."""
    clear_all_sessions()
    
    response = login_user("administration_admin", "adminis123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["user"]["username"] == "administration_admin"
    assert data["user"]["user_id"] == 6
    assert len(data["user"]["scopes"]) == 1
    assert data["user"]["scopes"][0]["role_code"] == "ADMINISTRATION_ADMIN"
    assert data["user"]["scopes"][0]["org_unit_id"] == 1
    assert data["user"]["scopes"][0]["org_unit_type"] == "ADMINISTRATION"
    print("✓ Login success: administration_admin with administration scope")


def test_login_failure_invalid_username():
    """Test login failure with non-existent username."""
    clear_all_sessions()
    
    response = login_user("nonexistent_user", "password123")
    
    assert response.status_code == 401
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"] == "INVALID_CREDENTIALS"
    assert "Invalid username or password" in data["detail"]["message"]
    assert "message_ar" in data["detail"]
    print("✓ Login failure: invalid username")


def test_login_failure_invalid_password():
    """Test login failure with wrong password."""
    clear_all_sessions()
    
    response = login_user("software_admin", "wrong_password")
    
    assert response.status_code == 401
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"] == "INVALID_CREDENTIALS"
    assert "Invalid username or password" in data["detail"]["message"]
    print("✓ Login failure: invalid password")


def test_login_failure_empty_username():
    """Test login failure with empty username."""
    clear_all_sessions()
    
    response = client.post(
        "/api/auth/login",
        json={"username": "", "password": "password123"}
    )
    
    # Should fail validation (422) or authentication (401)
    assert response.status_code in [401, 422]
    print("✓ Login failure: empty username")


def test_login_failure_empty_password():
    """Test login failure with empty password."""
    clear_all_sessions()
    
    response = client.post(
        "/api/auth/login",
        json={"username": "software_admin", "password": ""}
    )
    
    # Should fail validation (422) or authentication (401)
    assert response.status_code in [401, 422]
    print("✓ Login failure: empty password")


def test_login_session_persistence():
    """Test that session persists across requests after login."""
    clear_all_sessions()
    
    # Login
    login_response = login_user("software_admin", "admin123")
    assert login_response.status_code == 200
    
    # Verify session cookie was set
    assert "incident_manager_session" in client.cookies
    
    # Make another request - session should persist
    me_response = get_current_user()
    assert me_response.status_code == 200
    data = me_response.json()
    assert data["user"]["username"] == "software_admin"
    print("✓ Session persistence: session maintained across requests")


def test_login_case_sensitivity():
    """Test username case handling (depends on DB collation)."""
    clear_all_sessions()
    
    response = login_user("SOFTWARE_ADMIN", "admin123")
    
    # SQL Server default collation is case-insensitive
    # Accept both 200 (case-insensitive) and 401 (case-sensitive)
    assert response.status_code in [200, 401]
    
    if response.status_code == 200:
        print("✓ Login case handling: database uses case-insensitive collation")
    else:
        data = response.json()
        assert data["detail"]["error"] == "INVALID_CREDENTIALS"
        print("✓ Login case sensitivity: username is case-sensitive")


# ==================== LOGOUT ENDPOINT TESTS ====================

def test_logout_success_after_login():
    """Test successful logout after login."""
    clear_all_sessions()
    
    # Login first
    login_response = login_user("software_admin", "admin123")
    assert login_response.status_code == 200
    
    # Logout
    logout_response = logout_user()
    assert logout_response.status_code == 200
    data = logout_response.json()
    assert data["success"] is True
    assert data["message"] == "Logout successful"
    
    # Verify session is cleared - /me should now fail
    me_response = get_current_user()
    assert me_response.status_code == 401
    print("✓ Logout success: session cleared after logout")


def test_logout_without_login():
    """Test logout when not logged in (should still succeed)."""
    clear_all_sessions()
    
    response = logout_user()
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["message"] == "Logout successful"
    print("✓ Logout without login: succeeds idempotently")


def test_logout_multiple_times():
    """Test that logout can be called multiple times."""
    clear_all_sessions()
    
    # Login
    login_user("software_admin", "admin123")
    
    # Logout multiple times
    response1 = logout_user()
    assert response1.status_code == 200
    
    response2 = logout_user()
    assert response2.status_code == 200
    
    response3 = logout_user()
    assert response3.status_code == 200
    print("✓ Logout idempotency: can be called multiple times")


# ==================== GET CURRENT USER (/me) ENDPOINT TESTS ====================

def test_me_success_after_login():
    """Test /me returns user profile after login."""
    clear_all_sessions()
    
    # Login
    login_user("software_admin", "admin123")
    
    # Get current user
    response = get_current_user()
    
    assert response.status_code == 200
    data = response.json()
    assert "user" in data
    assert data["user"]["username"] == "software_admin"
    assert data["user"]["user_id"] == 1
    assert data["user"]["is_active"] is True
    assert len(data["user"]["scopes"]) == 1
    assert data["user"]["scopes"][0]["role_code"] == "SOFTWARE_ADMIN"
    print("✓ /me success: returns user profile after login")


def test_me_failure_without_login():
    """Test /me returns 401 when not logged in."""
    clear_all_sessions()
    
    response = get_current_user()
    
    assert response.status_code == 401
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"] == "NOT_AUTHENTICATED"
    assert "No active session" in data["detail"]["message"] or "Authentication required" in data["detail"]["message"]
    assert "message_ar" in data["detail"]
    print("✓ /me failure: 401 when not authenticated")


def test_me_failure_after_logout():
    """Test /me returns 401 after logout."""
    clear_all_sessions()
    
    # Login
    login_user("software_admin", "admin123")
    
    # Verify logged in
    response1 = get_current_user()
    assert response1.status_code == 200
    
    # Logout
    logout_user()
    
    # Verify /me now fails
    response2 = get_current_user()
    assert response2.status_code == 401
    print("✓ /me after logout: returns 401 after session cleared")


def test_me_returns_correct_user():
    """Test /me returns the correct user based on session."""
    clear_all_sessions()
    
    # Login as worker
    login_user("worker", "worker123")
    
    # Get current user
    response = get_current_user()
    
    assert response.status_code == 200
    data = response.json()
    assert data["user"]["username"] == "worker"
    assert data["user"]["user_id"] == 2
    assert data["user"]["scopes"][0]["role_code"] == "WORKER"
    print("✓ /me correctness: returns correct user based on session")


# ==================== MULTIPLE USER SESSION TESTS ====================

def test_multiple_users_independent_sessions():
    """Test that different test clients can have independent sessions."""
    clear_all_sessions()
    
    # Create two independent clients
    client1 = TestClient(app)
    client2 = TestClient(app)
    
    # Login as software_admin in client1
    response1 = client1.post(
        "/api/auth/login",
        json={"username": "software_admin", "password": "admin123"}
    )
    assert response1.status_code == 200
    
    # Login as worker in client2
    response2 = client2.post(
        "/api/auth/login",
        json={"username": "worker", "password": "worker123"}
    )
    assert response2.status_code == 200
    
    # Verify client1 is still software_admin
    me1 = client1.get("/api/auth/me")
    assert me1.status_code == 200
    assert me1.json()["user"]["username"] == "software_admin"
    
    # Verify client2 is still worker
    me2 = client2.get("/api/auth/me")
    assert me2.status_code == 200
    assert me2.json()["user"]["username"] == "worker"
    print("✓ Multiple users: independent sessions maintained")


# ==================== SESSION SWITCHING TESTS ====================

def test_login_switches_user():
    """Test that logging in as a different user switches the session."""
    clear_all_sessions()
    
    # Login as software_admin
    login_user("software_admin", "admin123")
    
    # Verify logged in as software_admin
    response1 = get_current_user()
    assert response1.status_code == 200
    assert response1.json()["user"]["username"] == "software_admin"
    
    # Login as worker (should switch session)
    login_user("worker", "worker123")
    
    # Verify now logged in as worker
    response2 = get_current_user()
    assert response2.status_code == 200
    assert response2.json()["user"]["username"] == "worker"
    print("✓ Session switching: login switches to new user")


# ==================== REQUEST VALIDATION TESTS ====================

def test_login_missing_username():
    """Test login fails when username is missing."""
    clear_all_sessions()
    
    response = client.post(
        "/api/auth/login",
        json={"password": "password123"}
    )
    
    assert response.status_code == 422  # Validation error
    print("✓ Validation: missing username rejected")


def test_login_missing_password():
    """Test login fails when password is missing."""
    clear_all_sessions()
    
    response = client.post(
        "/api/auth/login",
        json={"username": "software_admin"}
    )
    
    assert response.status_code == 422  # Validation error
    print("✓ Validation: missing password rejected")


def test_login_invalid_json():
    """Test login fails with invalid JSON."""
    clear_all_sessions()
    
    response = client.post(
        "/api/auth/login",
        data="not json"
    )
    
    assert response.status_code == 422  # Validation error
    print("✓ Validation: invalid JSON rejected")


# ==================== RESPONSE FORMAT TESTS ====================

def test_login_response_structure():
    """Test login response has correct structure."""
    clear_all_sessions()
    
    response = login_user("software_admin", "admin123")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check top-level keys
    assert "success" in data
    assert "message" in data
    assert "user" in data
    
    # Check user structure
    user = data["user"]
    assert "user_id" in user
    assert "username" in user
    assert "is_active" in user
    assert "scopes" in user
    
    # Check scopes structure
    assert isinstance(user["scopes"], list)
    assert len(user["scopes"]) > 0
    scope = user["scopes"][0]
    assert "role_code" in scope
    assert "org_unit_id" in scope
    assert "org_unit_type" in scope
    print("✓ Response format: login response structure correct")


def test_me_response_structure():
    """Test /me response has correct structure."""
    clear_all_sessions()
    
    # Login first
    login_user("software_admin", "admin123")
    
    response = get_current_user()
    
    assert response.status_code == 200
    data = response.json()
    
    # Check top-level keys
    assert "user" in data
    
    # Check user structure
    user = data["user"]
    assert "user_id" in user
    assert "username" in user
    assert "is_active" in user
    assert "scopes" in user
    print("✓ Response format: /me response structure correct")


def test_logout_response_structure():
    """Test logout response has correct structure."""
    clear_all_sessions()
    
    response = logout_user()
    
    assert response.status_code == 200
    data = response.json()
    
    assert "success" in data
    assert "message" in data
    print("✓ Response format: logout response structure correct")


def test_error_response_structure():
    """Test error responses have correct structure."""
    clear_all_sessions()
    
    response = login_user("invalid_user", "password")
    
    assert response.status_code == 401
    data = response.json()
    
    assert "detail" in data
    detail = data["detail"]
    assert "error" in detail
    assert "message" in detail
    assert "message_ar" in detail
    print("✓ Response format: error response structure correct")


# ==================== MAIN TEST RUNNER ====================

def run_all_tests():
    """Run all tests and report results."""
    test_functions = [
        # Login tests
        test_login_success_software_admin,
        test_login_success_worker,
        test_login_success_complaint_supervisor,
        test_login_success_section_admin,
        test_login_success_department_admin,
        test_login_success_administration_admin,
        test_login_failure_invalid_username,
        test_login_failure_invalid_password,
        test_login_failure_empty_username,
        test_login_failure_empty_password,
        test_login_session_persistence,
        test_login_case_sensitivity,
        
        # Logout tests
        test_logout_success_after_login,
        test_logout_without_login,
        test_logout_multiple_times,
        
        # /me tests
        test_me_success_after_login,
        test_me_failure_without_login,
        test_me_failure_after_logout,
        test_me_returns_correct_user,
        
        # Multiple user tests
        test_multiple_users_independent_sessions,
        
        # Session switching tests
        test_login_switches_user,
        
        # Validation tests
        test_login_missing_username,
        test_login_missing_password,
        test_login_invalid_json,
        
        # Response format tests
        test_login_response_structure,
        test_me_response_structure,
        test_logout_response_structure,
        test_error_response_structure,
    ]
    
    print("\n" + "="*70)
    print("PHASE 2 RBAC: AUTH ROUTER INTEGRATION TESTS")
    print("="*70 + "\n")
    
    passed = 0
    failed = 0
    errors = []
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append(f"{test_func.__name__}: {str(e)}")
            print(f"✗ FAILED: {test_func.__name__}")
        except Exception as e:
            failed += 1
            errors.append(f"{test_func.__name__}: {str(e)}")
            print(f"✗ ERROR: {test_func.__name__}: {str(e)}")
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {len(test_functions)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(test_functions)*100):.1f}%")
    
    if errors:
        print("\n" + "="*70)
        print("FAILURES")
        print("="*70)
        for error in errors:
            print(f"  - {error}")
    
    print("="*70 + "\n")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    sys.exit(0 if failed == 0 else 1)
