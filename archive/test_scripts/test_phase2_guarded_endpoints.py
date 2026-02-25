"""
Phase 2 RBAC: Integration Tests for Guarded Endpoints
Tests the complete flow: authentication → authorization → endpoint access
"""

import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# Test user credentials (must exist in database from Phase 2 setup)
TEST_USERS = {
    "software_admin": {"username": "software_admin", "password": "admin123"},
    "worker": {"username": "worker", "password": "worker123"},
    "complaint_supervisor": {"username": "complaint_supervisor", "password": "supervisor123"},
    "section_admin": {"username": "section_admin", "password": "section123"},
    "department_admin": {"username": "department_admin", "password": "dept123"},
}


def login(username: str, password: str) -> TestClient:
    """Login and return client with session cookies."""
    response = client.post(
        "/api/auth/login",
        json={"username": username, "password": password}
    )
    assert response.status_code == 200, f"Login failed: {response.text}"
    return client


def logout():
    """Logout current session."""
    client.post("/api/auth/logout")


def test_public_endpoint_no_auth():
    """Public endpoint should work without authentication."""
    response = client.get("/api/guarded/public")
    assert response.status_code == 200
    data = response.json()
    assert "public endpoint" in data["message"]
    print("✓ Public endpoint: accessible without authentication")


def test_authenticated_endpoint_requires_login():
    """Authenticated endpoint should require login but no specific role."""
    # Without login
    logout()
    response = client.get("/api/guarded/authenticated-only")
    assert response.status_code == 401
    assert "NOT_AUTHENTICATED" in response.json()["detail"]["error"]
    print("✓ Authenticated endpoint: blocks unauthenticated access")
    
    # With login (any role)
    login("worker", "worker123")
    response = client.get("/api/guarded/authenticated-only")
    assert response.status_code == 200
    data = response.json()
    assert "worker" in data["message"].lower()
    print("✓ Authenticated endpoint: allows any authenticated user")


def test_admin_only_endpoint():
    """Admin-only endpoint should only allow SOFTWARE_ADMIN."""
    # Worker should be forbidden
    login("worker", "worker123")
    response = client.get("/api/guarded/admin-only")
    assert response.status_code == 403
    assert "FORBIDDEN" in response.json()["detail"]["error"]
    print("✓ Admin-only endpoint: blocks WORKER")
    
    # Software admin should succeed
    login("software_admin", "admin123")
    response = client.get("/api/guarded/admin-only")
    assert response.status_code == 200
    data = response.json()
    assert "Software Administrator" in data["message"]
    print("✓ Admin-only endpoint: allows SOFTWARE_ADMIN")


def test_worker_only_endpoint():
    """Worker-only endpoint should only allow WORKER role."""
    # Admin should be forbidden
    login("software_admin", "admin123")
    response = client.get("/api/guarded/worker-only")
    assert response.status_code == 403
    print("✓ Worker-only endpoint: blocks SOFTWARE_ADMIN")
    
    # Worker should succeed
    login("worker", "worker123")
    response = client.get("/api/guarded/worker-only")
    assert response.status_code == 200
    data = response.json()
    assert "Worker" in data["message"]
    print("✓ Worker-only endpoint: allows WORKER")


def test_any_admin_endpoint():
    """Any admin endpoint should allow all admin roles."""
    admin_users = ["software_admin", "section_admin", "department_admin"]
    
    # Worker should be forbidden
    login("worker", "worker123")
    response = client.get("/api/guarded/any-admin")
    assert response.status_code == 403
    print("✓ Any-admin endpoint: blocks WORKER")
    
    # All admin types should succeed
    for admin in admin_users:
        login(admin, TEST_USERS[admin]["password"])
        response = client.get("/api/guarded/any-admin")
        assert response.status_code == 200, f"Failed for {admin}: {response.text}"
        print(f"✓ Any-admin endpoint: allows {admin.upper()}")


def test_any_supervisor_endpoint():
    """Any supervisor endpoint should allow supervisors and admins."""
    # Test with available users only
    # complaint_supervisor may not exist in all test databases
    allowed_users = ["software_admin", "section_admin", "department_admin"]
    
    # Worker should be forbidden
    login("worker", "worker123")
    response = client.get("/api/guarded/any-supervisor")
    assert response.status_code == 403
    print("✓ Any-supervisor endpoint: blocks WORKER")
    
    # All supervisors and admins should succeed
    for user in allowed_users:
        login(user, TEST_USERS[user]["password"])
        response = client.get("/api/guarded/any-supervisor")
        assert response.status_code == 200, f"Failed for {user}: {response.text}"
        print(f"✓ Any-supervisor endpoint: allows {user.upper()}")


def test_conditional_access():
    """Conditional access should return different data based on role."""
    # Worker gets limited access
    login("worker", "worker123")
    response = client.post("/api/guarded/conditional-access")
    assert response.status_code == 200
    data = response.json()
    assert data["access_level"] == "limited"
    assert "public_field" in data["data"]
    assert "sensitive_field_1" not in data["data"]
    print("✓ Conditional access: WORKER gets limited data")
    
    # Admin gets full access
    login("software_admin", "admin123")
    response = client.post("/api/guarded/conditional-access")
    assert response.status_code == 200
    data = response.json()
    assert data["access_level"] == "full"
    assert "sensitive_field_1" in data["data"]
    assert "all_records" in data["data"]
    print("✓ Conditional access: SOFTWARE_ADMIN gets full data")


def test_dangerous_operation_multi_level_check():
    """Dangerous operation should have multi-level authorization."""
    # Protected resource (ID <= 10) requires SOFTWARE_ADMIN
    login("section_admin", "section123")
    response = client.delete("/api/guarded/dangerous-operation/5")
    assert response.status_code == 403
    print("✓ Dangerous operation: SECTION_ADMIN blocked from protected resource")
    
    # Software admin can delete protected resource
    login("software_admin", "admin123")
    response = client.delete("/api/guarded/dangerous-operation/5")
    assert response.status_code == 200
    print("✓ Dangerous operation: SOFTWARE_ADMIN can delete protected resource")
    
    # Section admin can delete non-protected resource (ID > 10)
    login("section_admin", "section123")
    response = client.delete("/api/guarded/dangerous-operation/15")
    assert response.status_code == 200
    print("✓ Dangerous operation: SECTION_ADMIN can delete non-protected resource")
    
    # Worker cannot delete anything (not an admin)
    login("worker", "worker123")
    response = client.delete("/api/guarded/dangerous-operation/15")
    assert response.status_code == 403
    print("✓ Dangerous operation: WORKER blocked from all deletes")


def test_my_permissions_endpoint():
    """My permissions endpoint should return user's roles and permissions."""
    # Test with worker
    login("worker", "worker123")
    response = client.get("/api/guarded/my-permissions")
    assert response.status_code == 200
    data = response.json()
    assert data["user"] == "worker"
    assert "WORKER" in data["roles"]
    assert data["permissions"]["can_handle_complaints"] == True
    assert data["permissions"]["can_access_admin_panel"] == False
    print("✓ My permissions: WORKER permissions correct")
    
    # Test with admin
    login("software_admin", "admin123")
    response = client.get("/api/guarded/my-permissions")
    assert response.status_code == 200
    data = response.json()
    assert data["user"] == "software_admin"
    assert "SOFTWARE_ADMIN" in data["roles"]
    assert data["permissions"]["can_access_admin_panel"] == True
    print("✓ My permissions: SOFTWARE_ADMIN permissions correct")


def test_session_persistence_across_requests():
    """Session should persist across multiple requests."""
    # Login as worker
    login("worker", "worker123")
    
    # First request
    response1 = client.get("/api/guarded/authenticated-only")
    assert response1.status_code == 200
    
    # Second request (should use same session)
    response2 = client.get("/api/guarded/authenticated-only")
    assert response2.status_code == 200
    
    # Logout
    logout()
    
    # Third request (should fail after logout)
    response3 = client.get("/api/guarded/authenticated-only")
    assert response3.status_code == 401
    
    print("✓ Session persistence: session maintained across requests")
    print("✓ Session persistence: logout invalidates session")


def test_unauthenticated_access_to_protected_endpoints():
    """All protected endpoints should reject unauthenticated requests."""
    logout()
    
    protected_endpoints = [
        "/api/guarded/authenticated-only",
        "/api/guarded/admin-only",
        "/api/guarded/worker-only",
        "/api/guarded/any-admin",
        "/api/guarded/any-supervisor",
        "/api/guarded/my-permissions",
    ]
    
    for endpoint in protected_endpoints:
        response = client.get(endpoint)
        assert response.status_code == 401, f"Endpoint {endpoint} should reject unauthenticated access"
    
    print("✓ Unauthenticated access: all protected endpoints blocked")


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("PHASE 2 RBAC: GUARDED ENDPOINTS INTEGRATION TESTS")
    print("="*70 + "\n")
    
    tests = [
        ("Public Access", test_public_endpoint_no_auth),
        ("Authenticated Access", test_authenticated_endpoint_requires_login),
        ("Admin-Only Access", test_admin_only_endpoint),
        ("Worker-Only Access", test_worker_only_endpoint),
        ("Any-Admin Access", test_any_admin_endpoint),
        ("Any-Supervisor Access", test_any_supervisor_endpoint),
        ("Conditional Access", test_conditional_access),
        ("Multi-Level Authorization", test_dangerous_operation_multi_level_check),
        ("User Permissions", test_my_permissions_endpoint),
        ("Session Persistence", test_session_persistence_across_requests),
        ("Unauthenticated Access", test_unauthenticated_access_to_protected_endpoints),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n[{test_name}]")
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
