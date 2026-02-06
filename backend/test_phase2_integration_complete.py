"""
Integration Test: Complete Authentication Flow
Tests the entire authentication system end-to-end.
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_complete_authentication_flow():
    """
    Test complete flow:
    1. Access public endpoint (no auth)
    2. Try to access protected endpoint (fail)
    3. Login
    4. Access protected endpoint (success)
    5. Check user info
    6. Logout
    7. Try to access protected endpoint again (fail)
    """
    print("\n" + "="*70)
    print("INTEGRATION TEST: Complete Authentication Flow")
    print("="*70 + "\n")
    
    # Clear any existing sessions
    client.cookies.clear()
    
    # Step 1: Access public endpoint
    print("Step 1: Access public endpoint (no auth required)...")
    response = client.get("/api/example/public")
    assert response.status_code == 200
    assert response.json()["authentication"] == "not required"
    print("✓ Public endpoint accessible")
    
    # Step 2: Try to access protected endpoint without auth
    print("\nStep 2: Try to access protected endpoint without auth...")
    response = client.get("/api/example/protected")
    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "NOT_AUTHENTICATED"
    print("✓ Protected endpoint properly blocked")
    
    # Step 3: Login
    print("\nStep 3: Login as software_admin...")
    response = client.post(
        "/api/auth/login",
        json={"username": "software_admin", "password": "admin123"}
    )
    assert response.status_code == 200
    assert response.json()["success"] is True
    print(f"✓ Login successful: {response.json()['user']['username']}")
    
    # Step 4: Access protected endpoint after login
    print("\nStep 4: Access protected endpoint with authentication...")
    response = client.get("/api/example/protected")
    assert response.status_code == 200
    data = response.json()
    assert "Hello software_admin" in data["message"]
    assert data["authentication"] == "required and successful"
    print(f"✓ Protected endpoint accessible: {data['message']}")
    
    # Step 5: Check user info
    print("\nStep 5: Get detailed user info...")
    response = client.get("/api/example/user-info")
    assert response.status_code == 200
    data = response.json()
    assert data["user"]["username"] == "software_admin"
    assert len(data["access"]["roles"]) > 0
    print(f"✓ User info retrieved: {data['user']['username']}")
    print(f"  Roles: {[r['role'] for r in data['access']['roles']]}")
    
    # Step 6: Check specific role
    print("\nStep 6: Check if user has SOFTWARE_ADMIN role...")
    response = client.get("/api/example/check-role/SOFTWARE_ADMIN")
    assert response.status_code == 200
    data = response.json()
    assert data["has_role"] is True
    print(f"✓ Role check successful: has SOFTWARE_ADMIN = {data['has_role']}")
    
    # Step 7: Logout
    print("\nStep 7: Logout...")
    response = client.post("/api/auth/logout")
    assert response.status_code == 200
    assert response.json()["success"] is True
    print("✓ Logout successful")
    
    # Step 8: Try to access protected endpoint after logout
    print("\nStep 8: Try to access protected endpoint after logout...")
    response = client.get("/api/example/protected")
    assert response.status_code == 401
    print("✓ Protected endpoint properly blocked after logout")
    
    print("\n" + "="*70)
    print("✓ ALL INTEGRATION TESTS PASSED")
    print("="*70 + "\n")


def test_multiple_users_workflow():
    """Test that different users can work independently."""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Multiple Users Workflow")
    print("="*70 + "\n")
    
    # Create two independent clients
    client1 = TestClient(app)
    client2 = TestClient(app)
    
    # Client 1: Login as software_admin
    print("Client 1: Login as software_admin...")
    response1 = client1.post(
        "/api/auth/login",
        json={"username": "software_admin", "password": "admin123"}
    )
    assert response1.status_code == 200
    print("✓ Client 1 logged in as software_admin")
    
    # Client 2: Login as worker
    print("\nClient 2: Login as worker...")
    response2 = client2.post(
        "/api/auth/login",
        json={"username": "worker", "password": "worker123"}
    )
    assert response2.status_code == 200
    print("✓ Client 2 logged in as worker")
    
    # Verify Client 1 still sees software_admin
    print("\nVerifying Client 1 session...")
    response1 = client1.get("/api/example/protected")
    assert response1.status_code == 200
    assert "software_admin" in response1.json()["message"]
    print("✓ Client 1 correctly authenticated as software_admin")
    
    # Verify Client 2 sees worker
    print("\nVerifying Client 2 session...")
    response2 = client2.get("/api/example/protected")
    assert response2.status_code == 200
    assert "worker" in response2.json()["message"]
    print("✓ Client 2 correctly authenticated as worker")
    
    # Check roles are different
    print("\nChecking role differences...")
    role1 = client1.get("/api/example/user-info").json()["access"]["roles"][0]["role"]
    role2 = client2.get("/api/example/user-info").json()["access"]["roles"][0]["role"]
    assert role1 == "SOFTWARE_ADMIN"
    assert role2 == "WORKER"
    print(f"✓ Client 1 has role: {role1}")
    print(f"✓ Client 2 has role: {role2}")
    
    print("\n" + "="*70)
    print("✓ MULTIPLE USERS WORKFLOW PASSED")
    print("="*70 + "\n")


def test_dependency_vs_direct_calls():
    """Test that dependency gives same results as direct auth service calls."""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Dependency vs Direct Service Calls")
    print("="*70 + "\n")
    
    client.cookies.clear()
    
    # Login
    print("Login as section_admin...")
    client.post(
        "/api/auth/login",
        json={"username": "section_admin", "password": "section123"}
    )
    print("✓ Logged in")
    
    # Get user via /me endpoint (direct service call)
    print("\nGet user via /api/auth/me (direct service)...")
    me_response = client.get("/api/auth/me")
    assert me_response.status_code == 200
    me_user = me_response.json()["user"]
    print(f"✓ /me returned: {me_user['username']}")
    
    # Get user via dependency-protected endpoint
    print("\nGet user via dependency-protected endpoint...")
    dep_response = client.get("/api/example/user-info")
    assert dep_response.status_code == 200
    dep_user = dep_response.json()["user"]
    print(f"✓ Dependency returned: {dep_user['username']}")
    
    # Compare results
    print("\nComparing results...")
    assert me_user["user_id"] == dep_user["id"]
    assert me_user["username"] == dep_user["username"]
    assert me_user["is_active"] == dep_user["active"]
    print("✓ Both methods return identical user data")
    
    print("\n" + "="*70)
    print("✓ DEPENDENCY CONSISTENCY TEST PASSED")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        test_complete_authentication_flow()
        test_multiple_users_workflow()
        test_dependency_vs_direct_calls()
        
        print("\n" + "="*70)
        print("🎉 ALL INTEGRATION TESTS PASSED SUCCESSFULLY! 🎉")
        print("="*70 + "\n")
        print("Summary:")
        print("  ✓ Complete authentication flow works end-to-end")
        print("  ✓ Multiple independent user sessions work correctly")
        print("  ✓ Dependency returns consistent data with direct calls")
        print("  ✓ Session management works across all endpoints")
        print("  ✓ get_current_user() dependency fully functional")
        print("\n" + "="*70 + "\n")
        
        sys.exit(0)
    
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {str(e)}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
